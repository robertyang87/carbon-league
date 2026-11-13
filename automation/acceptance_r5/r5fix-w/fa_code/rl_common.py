"""rl_common -- shared plumbing for the Carbon League RL line (stage 1).

Contents:
  * CFG: league configuration (same construction as the eval scripts).
  * canonical_from_sim(sim, player): build the exact canonical arrays
    that shadow_alpha._canonical(sim.obs(player)) produces, but read
    straight from FastCarbon internals (no obs-dict round trip). This
    is the rollout fast path; test_rl_common.py proves bit-equality
    against the obs-dict path on live game states.
  * SeqState: trajectory state for the base head's seq features
    (identical semantics to shadow_alpha._SeqState).
  * NumpyActor: a policy with shadow_alpha's exact inference protocol
    (worker head SAMPLED at T=1 with a per-instance RNG, base head
    argmax) parameterized by an arbitrary weights dict -- used for the
    frozen BC shadow, self-play snapshots, and (via generated code) the
    exported rl_policy.
  * load_agent_fn: module loader identical to the eval scripts'.
"""

import importlib.util
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

_spec = json.load(open(os.path.join(
    HERE, "env_src", "carbon_challenge-master",
    "zerosum_env", "envs", "carbon", "carbon.json")))
CFG = {}
for _k, _v in _spec["configuration"].items():
    CFG[_k] = _v.get("default") if isinstance(_v, dict) else _v

_SUFFIX = re.compile(r"(\d+)$")


def _suf(k):
    m = _SUFFIX.search(str(k))
    return int(m.group(1)) if m else 0


# ----------------------------------------------------------------------
# Fast canonical builder (must mirror shadow_alpha._canonical exactly)
# ----------------------------------------------------------------------
def canonical_from_sim(sim, player):
    """Return (player, carbon, cash, bases, workers, wmask, trees,
    tmask, step, my_wids, my_bids) identical to
    shadow_alpha._canonical(sim.obs(player))."""
    p = int(player)
    carbon = np.zeros(sim.n_cells, dtype=np.float64)
    for wid in sim.player_workers[p]:
        w = sim.workers[wid]
        if w.is_collector:
            carbon[w.pos] = sim.carbon[w.pos]
        else:
            for nb in sim._nbr4[w.pos]:
                carbon[nb] = sim.carbon[nb]
    cash = np.zeros(2)
    bases = np.full((2, 2), -1, dtype=np.int64)
    workers = np.zeros((2, 10, 3))
    wmask = np.zeros((2, 10), dtype=bool)
    trows = []
    my_wids, my_bids = [], []
    for q in range(2):
        cash[q] = float(sim.cash[q])
        bids = sorted((bid for bid, _ in sim.bases[q]), key=_suf)[:2]
        bpos = dict(sim.bases[q])
        for i, bid in enumerate(bids):
            bases[q, i] = int(bpos[bid])
        wids = sorted(sim.player_workers[q], key=_suf)[:10]
        for i, wid in enumerate(wids):
            w = sim.workers[wid]
            workers[q, i] = [float(w.pos), float(w.carbon),
                             0.0 if w.is_collector else 1.0]
            wmask[q, i] = True
        for tid in sim.player_trees[q]:
            t = sim.trees[tid]
            trows.append((q, _suf(tid), float(t.pos), float(t.age)))
        if q == p:
            my_wids, my_bids = wids, bids
    trows.sort()
    trees = np.zeros((30, 3))
    tmask = np.zeros(30, dtype=bool)
    for i, (q, _s, pos, age) in enumerate(trows[:30]):
        trees[i] = [pos, age, q]
        tmask[i] = True
    step = int(sim.step_count)
    return p, carbon, cash, bases, workers, wmask, trees, tmask, step, \
        my_wids, my_bids


# ----------------------------------------------------------------------
# Seq state (shadow_alpha._SeqState semantics, verbatim logic)
# ----------------------------------------------------------------------
class SeqState:
    def __init__(self):
        self.prev_n = None
        self.prev_rec = 0
        self.last_rec_step = None
        self.last_step = None

    def seq_for(self, n_now, step):
        if self.prev_n is None or self.last_step is None \
                or step != self.last_step + 1:
            return {"prev_n": float(n_now), "lost": 0.0, "since": 60.0}
        lost = max(0, self.prev_n + self.prev_rec - n_now)
        since = 60 if self.last_rec_step is None \
            else step - self.last_rec_step
        return {"prev_n": float(self.prev_n), "lost": float(lost),
                "since": float(since)}

    def update(self, n_now, step, n_rec):
        self.prev_n = n_now
        self.prev_rec = n_rec
        self.last_step = step
        if n_rec > 0:
            self.last_rec_step = step


# ----------------------------------------------------------------------
# NumPy actor with shadow_alpha's inference protocol
# ----------------------------------------------------------------------
_MOVE = {1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
_REC = {1: "RECCOLLECTOR", 2: "RECPLANTER"}


def np_forward(weights, prefix, x):
    h = (x - weights[prefix + "mean"]) / weights[prefix + "std"]
    for i in (1, 2):
        h = h @ weights[f"{prefix}W{i}"].T + weights[f"{prefix}b{i}"]
        h = np.maximum(h, 0.0)
    return h @ weights[prefix + "W3"].T + weights[prefix + "b3"]


class NumpyActor:
    """Callable agent(observation, configuration) with shadow_alpha's
    protocol: worker actions sampled from the softmax (T=1) with a
    per-instance RNG, base actions argmax. Featurization is the
    bc_features module (same source the BC training used)."""

    def __init__(self, weights, seed=20261103):
        from bc_features import featurize
        self._featurize = featurize
        self.weights = weights
        self.state = SeqState()
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        self.state = SeqState()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act_canonical(self, canon):
        (p, carbon, cash, bases, workers, wmask, trees, tmask, step,
         my_wids, my_bids) = canon
        n_now = int(np.sum(wmask[p]))
        seq = self.state.seq_for(n_now, step)
        W, B = self._featurize(carbon, cash, bases, workers, wmask,
                               trees, tmask, p, step, seq=seq)
        actions = {}
        if my_wids:
            logits = np_forward(self.weights, "w_", W[:len(my_wids)])
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            for j, wid in enumerate(my_wids):
                a = int(self.rng.choice(5, p=probs[j]))
                if a in _MOVE:
                    actions[wid] = _MOVE[a]
        n_rec = 0
        if my_bids:
            blogits = np_forward(self.weights, "b_", B[:len(my_bids)])
            for i, bid in enumerate(my_bids):
                a = int(np.argmax(blogits[i]))
                if a in _REC:
                    actions[bid] = _REC[a]
                    n_rec += 1
        self.state.update(n_now, step, n_rec)
        return actions

    def __call__(self, observation, configuration):
        import shadow_alpha
        canon = shadow_alpha._canonical(observation)
        return self.act_canonical(canon)


def load_weights_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


_counter = [0]


def load_agent_fn(path, kind):
    """Identical to the eval scripts': policies via make_agent(),
    opponents via their module-level agent()."""
    _counter[0] += 1
    name = f"rlmod_{kind}_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if kind == "policy":
        return m.make_agent()
    return m.agent
