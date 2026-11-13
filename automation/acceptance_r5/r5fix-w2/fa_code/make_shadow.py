"""make_shadow -- generate shadow_alpha.py.

The generated file is self-contained: it embeds the bc_features.py
source verbatim (so the features at inference are computed by exactly
the code that produced the training set) plus a NumPy forward pass over
shadow_weights.npz. Run after train_bc.py.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

AGENT_TAIL = '''

# ----------------------------------------------------------------------
# Shadow Alpha agent: NumPy inference over shadow_weights.npz
# ----------------------------------------------------------------------
import os as _os
import re as _re

_WPATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "shadow_weights.npz")
_W = np.load(_WPATH)

_SUFFIX = _re.compile(r"(\\d+)$")


def _suf(k):
    m = _SUFFIX.search(str(k))
    return int(m.group(1)) if m else 0


def _forward(prefix, x):
    h = (x - _W[prefix + "mean"]) / _W[prefix + "std"]
    for i in (1, 2):
        h = h @ _W[f"{prefix}W{i}"].T + _W[f"{prefix}b{i}"]
        h = np.maximum(h, 0.0)
    return h @ _W[prefix + "W3"].T + _W[prefix + "b3"]


def _canonical(observation):
    """Parse a live observation into the canonical arrays bc_features
    expects, using the same slot ordering as extract_bc.py (numeric id
    suffix). Returns (player, arrays..., sorted worker/base id lists
    for the acting player)."""
    p = int(observation["player"])
    players = observation["players"]
    cash = np.zeros(2)
    bases = np.full((2, 2), -1, dtype=np.int64)
    workers = np.zeros((2, 10, 3))
    wmask = np.zeros((2, 10), dtype=bool)
    trows = []
    my_wids, my_bids = [], []
    for q in range(2):
        cash[q] = float(players[q][0])
        centers = players[q][1] or {}
        bids = sorted(centers, key=_suf)[:2]
        for i, bid in enumerate(bids):
            bases[q, i] = int(centers[bid])
        ws = players[q][2] or {}
        wids = sorted(ws, key=_suf)[:10]
        for i, wid in enumerate(wids):
            pos, cargo, typ = ws[wid]
            workers[q, i] = [float(pos), float(cargo),
                             0.0 if typ == "COLLECTOR" else 1.0]
            wmask[q, i] = True
        ts = players[q][3] or {}
        for tid in ts:
            pos, age = ts[tid]
            trows.append((q, _suf(tid), float(pos), float(age)))
        if q == p:
            my_wids, my_bids = wids, bids
    trows.sort()
    trees = np.zeros((30, 3))
    tmask = np.zeros(30, dtype=bool)
    for i, (q, _s, pos, age) in enumerate(trows[:30]):
        trees[i] = [pos, age, q]
        tmask[i] = True
    carbon = np.asarray(observation["carbon"], dtype=np.float64)
    step = int(observation.get("step", 0))
    return p, carbon, cash, bases, workers, wmask, trees, tmask, step, \\
        my_wids, my_bids


_MOVE = {1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
_REC = {1: "RECCOLLECTOR", 2: "RECPLANTER"}


class _SeqState:
    """Trajectory state for the base head's seq features, maintained
    exactly as train_bc.build_dataset computes them from the replay
    sample sequence: previous own worker count, workers lost during
    the previous step, steps since the last recruit order. Resets
    whenever the step counter does not advance by exactly 1."""

    def __init__(self):
        self.prev_n = None
        self.prev_rec = 0
        self.last_rec_step = None
        self.last_step = None

    def seq_for(self, n_now, step):
        if self.prev_n is None or self.last_step is None \\
                or step != self.last_step + 1:
            return {"prev_n": float(n_now), "lost": 0.0, "since": 60.0}
        lost = max(0, self.prev_n + self.prev_rec - n_now)
        since = 60 if self.last_rec_step is None \\
            else step - self.last_rec_step
        return {"prev_n": float(self.prev_n), "lost": float(lost),
                "since": float(since)}

    def update(self, n_now, step, n_rec):
        self.prev_n = n_now
        self.prev_rec = n_rec
        self.last_step = step
        if n_rec > 0:
            self.last_rec_step = step


def _act(observation, state, rng):
    (p, carbon, cash, bases, workers, wmask, trees, tmask, step,
     my_wids, my_bids) = _canonical(observation)
    n_now = int(np.sum(wmask[p]))
    seq = state.seq_for(n_now, step)
    W, B = featurize(carbon, cash, bases, workers, wmask, trees, tmask,
                     p, step, seq=seq)
    actions = {}
    if my_wids:
        # Worker head: SAMPLE from the learned distribution (T=1).
        # Argmax collapses in closed loop (stay is the plurality
        # class; the clone parks on drained cells and its economy
        # deadlocks). Sampling restores the demonstrator's marginals.
        # See bc_shadow_report.md.
        logits = _forward("w_", W[:len(my_wids)])
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        for j, wid in enumerate(my_wids):
            a = int(rng.choice(5, p=probs[j]))
            if a in _MOVE:
                actions[wid] = _MOVE[a]
    n_rec = 0
    if my_bids:
        blogits = _forward("b_", B[:len(my_bids)])
        for i, bid in enumerate(my_bids):
            a = int(np.argmax(blogits[i]))
            if a in _REC:
                actions[bid] = _REC[a]
                n_rec += 1
    state.update(n_now, step, n_rec)
    return actions


def make_agent():
    state = _SeqState()
    rng = np.random.default_rng(20261103)
    return lambda observation, configuration: _act(observation, state,
                                                   rng)


_GLOBAL_STATE = _SeqState()
_GLOBAL_RNG = np.random.default_rng(20261103)


def agent(observation, configuration):
    return _act(observation, _GLOBAL_STATE, _GLOBAL_RNG)
'''


def main():
    with open(os.path.join(HERE, "bc_features.py")) as f:
        feat_src = f.read()
    header = ('"""shadow_alpha -- Shadow Alpha: behaviour clone of the '
              "Claude-family\n(AlphaCarbon-2-led) policies, trained on "
              "bc_data/claude_family_fixed.npz\n(action labels shifted "
              "+1 step; the raw extract misaligns them -- see\n"
              "bc_shadow_report.md).\n\nGenerated by make_shadow.py: "
              "the featurization below is the bc_features.py\nsource "
              "verbatim; inference is a NumPy MLP over "
              "shadow_weights.npz.\nWorker actions are sampled from the "
              "learned distribution (argmax\ncollapses in closed loop); "
              "base actions are argmax. Features use\nONLY live-legal "
              "information (own visibility-filtered carbon + the\nshared "
              'players block)."""\n')
    out = header + "\n" + feat_src + AGENT_TAIL
    path = os.path.join(HERE, "shadow_alpha.py")
    with open(path, "w") as f:
        f.write(out)
    print("wrote", path, len(out), "bytes")


if __name__ == "__main__":
    main()
