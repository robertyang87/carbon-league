"""rl2_common -- stage-2 shared machinery: hard-rule shields, action
masks, decorrelated sequential sampling, and the NumPy actor for the
belief-feature NN (featurize2 dims).

The ACTOR SECTION between the markers is written against only:
numpy, bc_features' geometry names (_idx/_xs/_ys/DIST) and belief's
names (BeliefTracker, featurize2, _NBR4) -- gen_rl_policy2.py embeds
it verbatim into the deployable rl_policy_v2.py.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bc_features import DIST, _idx, _xs, _ys  # noqa: E402
from belief import _NBR4, BeliefTracker, featurize2  # noqa: E402

import rl_common  # noqa: E402

# === DEPLOYABLE ACTOR BEGIN ===
_MOVE = {1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
_REC = {1: "RECCOLLECTOR", 2: "RECPLANTER"}
_DELTA = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))

DANGER_MIN_CARGO = 15.0     # v0.5 guard_danger_min_cargo
ENDGAME_STEP = 270          # last-30-steps return scheduling
OPENING_LAST_STEP = 40      # opening shield hard-off
OPENING_PLANTER_FORCE_STEP = 6
OPENING_SECOND_COLLECTOR_STEP = 2
OPENING_THROTTLE_CASH = 90.0


def _dest(pos, a):
    dx, dy = _DELTA[a]
    return int(_idx(int(_xs[pos]) + dx, int(_ys[pos]) + dy))


def _np_forward(weights, prefix, x):
    h = (x - weights[prefix + "mean"]) / weights[prefix + "std"]
    for i in (1, 2):
        h = h @ weights[f"{prefix}W{i}"].T + weights[f"{prefix}b{i}"]
        h = np.maximum(h, 0.0)
    return h @ weights[prefix + "W3"].T + weights[prefix + "b3"]


def danger_allowed(canon, j):
    """Shield (a): action mask for one own worker.

    Returns a bool (5,) of allowed actions, or None for no mask.
    Only collectors carrying >= DANGER_MIN_CARGO are masked: landing
    cells reachable next step by an enemy collector carrying LESS
    (it would win the collision) are forbidden -- v0.5's hard danger
    set. Landing on an own base cell is always allowed (depositing
    must never be masked off). If every action would be masked, the
    mask is dropped entirely (fallback)."""
    (p, carbon, cash, bases, workers, wmask, trees, tmask, step,
     my_wids, my_bids) = canon
    if int(workers[p, j, 2]) != 0:
        return None
    cargo = float(workers[p, j, 1])
    if cargo < DANGER_MIN_CARGO:
        return None
    opp = 1 - p
    hard = set()
    for k in range(workers.shape[1]):
        if wmask[opp, k] and int(workers[opp, k, 2]) == 0 \
                and float(workers[opp, k, 1]) < cargo:
            epos = int(workers[opp, k, 0])
            hard.add(epos)
            for nb in _NBR4[epos]:
                hard.add(int(nb))
    if not hard:
        return None
    own_bases = {int(b) for b in bases[p] if int(b) >= 0}
    pos = int(workers[p, j, 0])
    allowed = np.ones(5, dtype=bool)
    for a in range(5):
        d = _dest(pos, a)
        if d in hard and d not in own_bases:
            allowed[a] = False
    if not np.any(allowed):
        return None
    return allowed


def sequential_sample(probs, positions, own_bases, allowed_list, rng):
    """Decorrelated sequential sampling (shadow v2's protocol, exact-
    mask form): workers are sampled in slot order; worker j's
    distribution is its probs restricted to (danger-allowed) actions
    whose landing cell is not already reserved by an earlier own
    worker this step (own base cells exempt from reservation), then
    renormalized. Returns (actions int array, masks bool (n,5),
    n_fallbacks). The returned mask is exactly the support the action
    was sampled from, so logp under the masked distribution is the
    true behaviour logp (what PPO needs)."""
    n = len(positions)
    acts = np.zeros(n, dtype=np.int64)
    masks = np.ones((n, 5), dtype=bool)
    reserved = set()
    fallbacks = 0
    for j in range(n):
        mask = np.ones(5, dtype=bool)
        if allowed_list is not None and allowed_list[j] is not None:
            mask &= allowed_list[j]
        pos = int(positions[j])
        for a in range(5):
            if mask[a] and _dest(pos, a) in reserved \
                    and _dest(pos, a) not in own_bases:
                mask[a] = False
        if not np.any(mask):
            mask = np.ones(5, dtype=bool)
            fallbacks += 1
        pr = np.where(mask, probs[j], 0.0)
        if not np.isfinite(pr).all():
            # robustness (2026-11-03): a resumed-training rollout died
            # on NaN probabilities from an extreme standardized
            # feature (BC std floor 1e-6 on a train-constant feature,
            # amplified by PPO-updated weights). Sanitize: without
            # this the sampler raises and the whole game is lost;
            # with it, behaviour differs only in states that would
            # otherwise crash.
            pr = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
            fallbacks += 1
        tot = pr.sum()
        if tot <= 1e-12:
            pr = mask.astype(np.float64)
            tot = pr.sum()
            fallbacks += 1
        pr = pr / tot
        a = int(rng.choice(5, p=pr))
        acts[j] = a
        masks[j] = mask
        reserved.add(_dest(pos, a))
    return acts, masks, fallbacks


def endgame_action(canon, j):
    """Shield (b): in the last 30 steps a loaded collector is routed
    straight home along the toroidal distance field (BFS == Dijkstra
    on this unweighted grid); on an own base cell it stays. Returns an
    action index or None."""
    (p, carbon, cash, bases, workers, wmask, trees, tmask, step,
     my_wids, my_bids) = canon
    if step < ENDGAME_STEP:
        return None
    if int(workers[p, j, 2]) != 0 or float(workers[p, j, 1]) <= 0:
        return None
    own_bases = [int(b) for b in bases[p] if int(b) >= 0]
    if not own_bases:
        return None
    pos = int(workers[p, j, 0])
    if pos in own_bases:
        return 0
    cur = min(int(DIST[pos, b]) for b in own_bases)
    best_a, best_d = None, cur
    for a in (1, 2, 3, 4):
        d = min(int(DIST[_dest(pos, a), b]) for b in own_bases)
        if d < best_d:
            best_a, best_d = a, d
    return best_a


class OpeningShield:
    """Shield (c): opening cash-chain protection.

    Until the first income lands (own cash ever exceeds the 80 start)
    AND a planter is alive -- and never past step 40 -- base decisions
    are overridden:
      step 0: base slot 0 -> RECCOLLECTOR, slot 1 -> RECPLANTER
              (the v0.5-proven 1C+1P opening);
      step >= 2: a second collector is forced while collectors < 2
              and cash >= 30 (deposit-chain throughput);
      step >= 6: a planter is forced while no planter is alive and
              cash >= 30 (the stage-1 cliff was the planter timing
              being left to NN chance);
      any other base while the shield is active and cash < 90 is
      throttled to NONE (recruit throttle floor: do not let random
      recruiting drain the buffer the chain depends on).
    Returns per-slot directives: ('free',) | ('force', action) |
    ('none',)."""

    def __init__(self):
        self.cash_max = None
        self.active = True

    def reset(self):
        self.__init__()

    def directives(self, canon):
        (p, carbon, cash, bases, workers, wmask, trees, tmask, step,
         my_wids, my_bids) = canon
        n_bases = len(my_bids)
        free = [("free",)] * n_bases
        if not self.active or n_bases == 0:
            return free
        my_cash = float(cash[p])
        self.cash_max = my_cash if self.cash_max is None \
            else max(self.cash_max, my_cash)
        n_coll = 0
        n_plan = 0
        for j in range(workers.shape[1]):
            if wmask[p, j]:
                if int(workers[p, j, 2]) == 0:
                    n_coll += 1
                else:
                    n_plan += 1
        deposit_seen = self.cash_max > 80.0 + 1e-9
        if step >= OPENING_LAST_STEP or (deposit_seen and n_plan > 0):
            self.active = False
            return free
        out = []
        forced_used = False
        for i in range(n_bases):
            if step == 0:
                out.append(("force", 1 if i == 0 else 2))
                forced_used = True
            elif not forced_used and n_plan == 0 \
                    and step >= OPENING_PLANTER_FORCE_STEP \
                    and my_cash >= 30.0:
                out.append(("force", 2))
                forced_used = True
            elif not forced_used and n_coll < 2 \
                    and step >= OPENING_SECOND_COLLECTOR_STEP \
                    and my_cash >= 30.0 and not deposit_seen:
                out.append(("force", 1))
                forced_used = True
            elif my_cash < OPENING_THROTTLE_CASH:
                out.append(("none",))
            else:
                out.append(("free",))
        return out


class _SeqState2:
    """rl_common.SeqState semantics (base-head seq features)."""

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


class NumpyActor2:
    """Belief-feature NN actor with shields + decorrelated sampling.

    shields: dict with keys 'danger', 'endgame', 'opening' (all True
    by default). act_canonical returns (actions, info) where info
    carries per-unit masks and shield-exclusion flags for PPO."""

    def __init__(self, weights, seed=20261103, shields=None):
        self.weights = weights
        self.rng = np.random.default_rng(seed)
        self.tracker = BeliefTracker()
        self.seq = _SeqState2()
        self.opening = OpeningShield()
        self.shields = {"danger": True, "endgame": True,
                        "opening": True}
        if shields:
            self.shields.update(shields)
        self.stats = {"fallbacks": 0, "endgame_overrides": 0,
                      "opening_forced": 0, "opening_throttled": 0}

    def reset(self, seed=None):
        self.tracker.reset()
        self.seq = _SeqState2()
        self.opening.reset()
        self.stats = {"fallbacks": 0, "endgame_overrides": 0,
                      "opening_forced": 0, "opening_throttled": 0}
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act_canonical(self, canon):
        (p, carbon, cash, bases, workers, wmask, trees, tmask, step,
         my_wids, my_bids) = canon
        belief = self.tracker.update(carbon, workers, wmask, trees,
                                     tmask, p, step)
        n_now = int(np.sum(wmask[p]))
        seq = self.seq.seq_for(n_now, step)
        W, B = featurize2(carbon, cash, bases, workers, wmask, trees,
                          tmask, p, step, belief, seq=seq)
        actions = {}
        info = {"worker_mask": None, "worker_excluded": None,
                "base_excluded": None}
        if my_wids:
            logits = _np_forward(self.weights, "w_",
                                 W[:len(my_wids)])
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            own_bases = {int(b) for b in bases[p] if int(b) >= 0}
            allowed = None
            if self.shields["danger"]:
                allowed = [danger_allowed(canon, j)
                           for j in range(len(my_wids))]
            positions = [int(workers[p, j, 0])
                         for j in range(len(my_wids))]
            acts, masks, fb = sequential_sample(
                probs, positions, own_bases, allowed, self.rng)
            self.stats["fallbacks"] += fb
            excluded = np.zeros(len(my_wids), dtype=bool)
            if self.shields["endgame"]:
                for j in range(len(my_wids)):
                    ea = endgame_action(canon, j)
                    if ea is not None and ea != int(acts[j]):
                        acts[j] = ea
                        excluded[j] = True
                        self.stats["endgame_overrides"] += 1
                    elif ea is not None:
                        excluded[j] = True
            for j, wid in enumerate(my_wids):
                if int(acts[j]) in _MOVE:
                    actions[wid] = _MOVE[int(acts[j])]
            info["worker_mask"] = masks
            info["worker_excluded"] = excluded
        n_rec = 0
        if my_bids:
            blogits = _np_forward(self.weights, "b_",
                                  B[:len(my_bids)])
            chosen = [int(np.argmax(blogits[i]))
                      for i in range(len(my_bids))]
            bexcl = np.zeros(len(my_bids), dtype=bool)
            if self.shields["opening"]:
                dirs = self.opening.directives(canon)
                for i, d in enumerate(dirs):
                    if d[0] == "force":
                        chosen[i] = d[1]
                        bexcl[i] = True
                        self.stats["opening_forced"] += 1
                    elif d[0] == "none":
                        chosen[i] = 0
                        bexcl[i] = True
                        self.stats["opening_throttled"] += 1
            for i, bid in enumerate(my_bids):
                if chosen[i] in _REC:
                    actions[bid] = _REC[chosen[i]]
                    n_rec += 1
            info["base_excluded"] = bexcl
        self.seq.update(n_now, step, n_rec)
        return actions, info

    def __call__(self, observation, configuration):
        actions, _info = self.act_canonical(_canonical_obs(observation))
        return actions


def _canonical_obs(observation):
    """obs-dict -> canonical arrays. In the deployable generated file
    a local _canonical (from shadow_alpha) is in globals; in the
    training workspace, defer to shadow_alpha's."""
    g = globals().get("_canonical")
    if g is not None:
        return g(observation)
    import shadow_alpha
    return shadow_alpha._canonical(observation)

# === DEPLOYABLE ACTOR END ===


class NumpyActorD:
    """Frozen-opponent actor: v1-feature weights (e.g. shadow v2) with
    decorrelated sequential sampling on the canonical fast path.
    Optional stay-logit penalty (shadow v2's calibration: 0.4)."""

    def __init__(self, weights, seed=20261103, stay_penalty=0.0):
        self.weights = weights
        self.rng = np.random.default_rng(seed)
        self.state = rl_common.SeqState()
        self.stay_penalty = stay_penalty

    def reset(self, seed=None):
        self.state = rl_common.SeqState()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act_canonical(self, canon):
        from bc_features import featurize
        (p, carbon, cash, bases, workers, wmask, trees, tmask, step,
         my_wids, my_bids) = canon
        n_now = int(np.sum(wmask[p]))
        seq = self.state.seq_for(n_now, step)
        W, B = featurize(carbon, cash, bases, workers, wmask, trees,
                         tmask, p, step, seq=seq)
        actions = {}
        if my_wids:
            logits = rl_common.np_forward(self.weights, "w_",
                                          W[:len(my_wids)])
            logits = logits - logits.max(axis=1, keepdims=True)
            if self.stay_penalty:
                logits[:, 0] -= self.stay_penalty
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            own_bases = {int(b) for b in bases[p] if int(b) >= 0}
            positions = [int(workers[p, j, 0])
                         for j in range(len(my_wids))]
            acts, _m, _fb = sequential_sample(
                probs, positions, own_bases, None, self.rng)
            for j, wid in enumerate(my_wids):
                if int(acts[j]) in _MOVE:
                    actions[wid] = _MOVE[int(acts[j])]
        n_rec = 0
        if my_bids:
            blogits = rl_common.np_forward(self.weights, "b_",
                                           B[:len(my_bids)])
            for i, bid in enumerate(my_bids):
                a = int(np.argmax(blogits[i]))
                if a in _REC:
                    actions[bid] = _REC[a]
                    n_rec += 1
        self.state.update(n_now, step, n_rec)
        return actions
