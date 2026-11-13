"""make_shadow_ds -- generate shadow_deepseek.py.

Same construction as make_shadow.py (bc_features.py source embedded
verbatim + NumPy forward over the weights npz), with two changes:

  * weights come from shadow_deepseek_weights.npz (BC trained on the
    NaMnSi DeepSeek-family corpus, bc_data/deepseek_family.npz);
  * worker sampling is DECORRELATED within each step to fix v1's top
    distortion -- own workers stacking on one cell and killing each
    other (v1: 60-76 deaths/game vs the demonstrator's ~34):

    Workers are decided sequentially in canonical slot order. Each
    sampled destination cell is reserved; if a later worker's sampled
    action would land it on a cell already reserved by an own worker
    this step, that action is masked out of its distribution and the
    worker resamples from the renormalized remainder (repeat as
    needed; if every action is masked -- practically impossible --
    the original sample is kept and counted as a fallback).
    Own BASE cells are exempt: collisions there resolve before the
    deposit phase and the winning collector keeps the merged cargo,
    so base stacking consolidates instead of losing carbon -- it is
    part of the demonstrator's endgame pattern, not the bug.

    Cost accounting: the agent counts conflicts / resamples /
    fallbacks in `state.stats`, exposed as `fn.stats` on the closure
    returned by make_agent() (and `_GLOBAL_STATE.stats` for the
    module-level agent).

Run after train_bc.py + calibrate_*.py with BC_WEIGHTS=shadow_deepseek_weights.npz.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

AGENT_TAIL = '''

# ----------------------------------------------------------------------
# Shadow DeepSeek agent: NumPy inference over shadow_deepseek_weights.npz,
# worker sampling with same-step destination reservation (see
# make_shadow_ds.py header for the decorrelation design).
# ----------------------------------------------------------------------
import os as _os
import re as _re

_WPATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "shadow_deepseek_weights.npz")
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
# action index -> (dx, dy); 0 = stay. Matches fastsim _MOVE_OFFSETS.
_DELTA = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))


def _dest(pos, a):
    dx, dy = _DELTA[a]
    return int(_idx(int(_xs[pos]) + dx, int(_ys[pos]) + dy))


class _SeqState:
    """Trajectory state for the base head's seq features, maintained
    exactly as train_bc.build_dataset computes them from the replay
    sample sequence: previous own worker count, workers lost during
    the previous step, steps since the last recruit order. Resets
    whenever the step counter does not advance by exactly 1.
    Also carries the decorrelation counters (stats)."""

    def __init__(self):
        self.prev_n = None
        self.prev_rec = 0
        self.last_rec_step = None
        self.last_step = None
        self.stats = {"conflicts": 0, "resamples": 0, "fallbacks": 0,
                      "worker_decisions": 0}

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
        # Worker head: SAMPLE from the learned distribution (T=1), as
        # in v1 -- but sequentially, with same-step destination
        # reservation (decorrelation, see file header).
        logits = _forward("w_", W[:len(my_wids)])
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        own_bases = {int(b) for b in bases[p] if int(b) >= 0}
        reserved = {}
        stats = state.stats
        for j, wid in enumerate(my_wids):
            pos = int(workers[p, j, 0])
            pr = probs[j].copy()
            a = int(rng.choice(5, p=pr))
            while True:
                dest = _dest(pos, a)
                if dest in reserved and dest not in own_bases:
                    stats["conflicts"] += 1
                    pr[a] = 0.0
                    tot = pr.sum()
                    if tot <= 1e-12:
                        stats["fallbacks"] += 1
                        break
                    pr = pr / tot
                    a = int(rng.choice(5, p=pr))
                    stats["resamples"] += 1
                    continue
                break
            reserved[dest] = reserved.get(dest, 0) + 1
            stats["worker_decisions"] += 1
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
    fn = lambda observation, configuration: _act(observation, state, \\
                                                 rng)
    fn.stats = state.stats
    return fn


_GLOBAL_STATE = _SeqState()
_GLOBAL_RNG = np.random.default_rng(20261103)


def agent(observation, configuration):
    return _act(observation, _GLOBAL_STATE, _GLOBAL_RNG)
'''


def main():
    with open(os.path.join(HERE, "bc_features.py")) as f:
        feat_src = f.read()
    header = ('"""shadow_deepseek -- Shadow DeepSeek: behaviour clone '
              "of the NaMnSi DeepSeek family (ladder #2/#3: "
              "DeepSeek-RL-policy-1 / -light-1, plus NN-light siblings), "
              "trained on bc_data/deepseek_family.npz\n(public snapshot + "
              "harvested accumulated games, deduped by match id;\naction "
              "labels shifted +1 step -- the raw extract misaligns\nthem, "
              "see bc_shadow_deepseek_report.md). SPARRING ONLY --\nnot a "
              "submission lineage.\n\n"
              "Generated by make_shadow_ds.py: the featurization below "
              "is the bc_features.py\nsource verbatim; inference is a "
              "NumPy MLP over shadow_deepseek_weights.npz.\nWorker actions are "
              "sampled from the learned distribution (argmax\ncollapses "
              "in closed loop) WITH same-step destination reservation:\n"
              "own workers no longer stack onto one cell and kill each\n"
              "other (own base cells exempt -- stacking there "
              "consolidates\ncargo before deposit). Base actions are "
              "argmax. Features use\nONLY live-legal information (own "
              "visibility-filtered carbon + the\nshared players "
              'block)."""\n')
    out = header + "\n" + feat_src + AGENT_TAIL
    path = os.path.join(HERE, "shadow_deepseek.py")
    with open(path, "w") as f:
        f.write(out)
    print("wrote", path, len(out), "bytes")


if __name__ == "__main__":
    main()
