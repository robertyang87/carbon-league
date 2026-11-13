"""rl_policy_v2 -- RL policy rl-v0.2 (Carbon League NN line,
stage 2).

PPO (stage-1 recipe: clipped surrogate + GAE, per-unit baseline
attribution rewards) warm-started from a stage-2 BC trained on the
53-game Claude-family corpus with belief-augmented features
(four-fold-symmetric belief carbon map, juzi-style reconstruction --
see belief.py docstring), self-played against {shadow v2,
self-snapshots, policy_v0_5, example}; juzi was never in training
(probe/final-eval only). Checkpoint selection: fixed probe set every
iteration (see rl_stage2_report.md). Inference: worker actions are
sampled (T=1) through the decorrelated sequential sampler with the
three hard-rule shields active (danger mask / endgame return /
opening cash-chain protection); base actions argmax with the opening
shield's overrides. Weights: rl2_weights.npz in this directory.
"""

import os as _os
import re as _re

import numpy as _np
np = _np


# ===== bc_features.py (verbatim) =====
import numpy as np

SIZE = 15
N = SIZE * SIZE

_xs = np.arange(N) % SIZE
_ys = SIZE - 1 - np.arange(N) // SIZE


def _idx(x, y):
    return (SIZE - 1 - (y % SIZE)) * SIZE + (x % SIZE)


_dxa = np.abs(_xs[:, None] - _xs[None, :])
_dya = np.abs(_ys[:, None] - _ys[None, :])
DIST = np.minimum(_dxa, SIZE - _dxa) + np.minimum(_dya, SIZE - _dya)

# 5x5 patch (dy +2..-2, dx -2..+2) and 3x3 patch (dy +1..-1, dx -1..+1)
PATCH = np.zeros((N, 25), dtype=np.int64)
PATCH3 = np.zeros((N, 9), dtype=np.int64)
for _i in range(N):
    _x, _y = int(_xs[_i]), int(_ys[_i])
    _k = 0
    for _dy in (2, 1, 0, -1, -2):
        for _dx in (-2, -1, 0, 1, 2):
            PATCH[_i, _k] = _idx(_x + _dx, _y + _dy)
            _k += 1
    _k = 0
    for _dy in (1, 0, -1):
        for _dx in (-1, 0, 1):
            PATCH3[_i, _k] = _idx(_x + _dx, _y + _dy)
            _k += 1

GLOBAL_DIM = 13
WORKER_LOCAL_DIM = 111
WORKER_DIM = GLOBAL_DIM + WORKER_LOCAL_DIM
BASE_LOCAL_DIM = 12
BASE_DIM = GLOBAL_DIM + BASE_LOCAL_DIM


def _global_feats(carbon, cash, bases, workers, wmask, trees, tmask,
                  player, step):
    s, o = player, 1 - player
    n_coll = np.zeros(2)
    n_plan = np.zeros(2)
    carried = np.zeros(2)
    for q in (0, 1):
        m = wmask[q]
        t = workers[q, :, 2]
        n_coll[q] = np.sum(m & (t == 0))
        n_plan[q] = np.sum(m & (t == 1))
        carried[q] = np.sum(workers[q, :, 1] * m)
    n_tree = np.zeros(2)
    if np.any(tmask):
        own = trees[:, 2]
        n_tree[0] = np.sum(tmask & (own == 0))
        n_tree[1] = np.sum(tmask & (own == 1))
    g = np.array([
        step / 300.0,
        np.clip(cash[s] / 1000.0, 0, 5),
        np.clip(cash[o] / 1000.0, 0, 5),
        np.clip((cash[s] - cash[o]) / 2000.0, -2, 2),
        n_coll[s] / 10.0,
        n_plan[s] / 10.0,
        n_coll[o] / 10.0,
        n_plan[o] / 10.0,
        n_tree[s] / 15.0,
        n_tree[o] / 15.0,
        np.clip(carried[s] / 1000.0, 0, 3),
        np.clip(np.sum(carbon) / 3000.0, 0, 3),
        np.sum(wmask[s]) / 10.0,
    ], dtype=np.float64)
    return g


def _board_maps(workers, wmask, trees, tmask, player):
    """Per-cell owner/presence maps used by the local patches."""
    s, o = player, 1 - player
    tree_map = np.zeros(N)          # +1 own tree, -1 enemy tree
    tree_age = np.zeros(N)
    if np.any(tmask):
        for pos, age, owner in trees[tmask]:
            tree_map[int(pos)] = 1.0 if int(owner) == s else -1.0
            tree_age[int(pos)] = age
    enemy_pres = np.zeros(N)
    friend_pres = np.zeros(N)
    for q, mp in ((s, friend_pres), (o, enemy_pres)):
        for j in range(workers.shape[1]):
            if wmask[q, j]:
                mp[int(workers[q, j, 0])] += 1.0
    return tree_map, tree_age, enemy_pres, friend_pres


def _nearest(pos, targets):
    """Toroidal manhattan distance from pos to nearest of targets."""
    if len(targets) == 0:
        return 14.0, -1
    d = DIST[pos, np.asarray(targets, dtype=np.int64)]
    k = int(np.argmin(d))
    return float(d[k]), int(targets[k])


def _dirvec(pos, target):
    """Signed toroidal offset (dx, dy) from pos to target, /7.
    Returns (0, 0) when target is -1 (absent)."""
    if target < 0:
        return 0.0, 0.0
    dx = (int(_xs[target]) - int(_xs[pos]) + 7) % SIZE - 7
    dy = (int(_ys[target]) - int(_ys[pos]) + 7) % SIZE - 7
    return dx / 7.0, dy / 7.0


def featurize(carbon, cash, bases, workers, wmask, trees, tmask,
              player, step, seq=None):
    """Return (W, B): worker features (10, WORKER_DIM) for the player's
    own workers (rows where wmask[player] is False are zeroed) and base
    features (2, BASE_DIM) for the player's own bases.

    seq: optional dict of trajectory-state features for the base head,
    all computable live by an agent that remembers its own previous
    observation and its own previous actions (no hidden information):
      prev_n   -- own worker count at the previous step
      lost     -- own workers lost during the previous step
                  (prev_n + own recruits ordered last step - n now)
      since    -- steps since the player last ordered any recruit
    Defaults to zeros (used for step 0 / stateless calls)."""
    carbon = np.asarray(carbon, dtype=np.float64).ravel()
    cash = np.asarray(cash, dtype=np.float64)
    workers = np.asarray(workers, dtype=np.float64)
    trees = np.asarray(trees, dtype=np.float64)
    s, o = player, 1 - player
    g = _global_feats(carbon, cash, bases, workers, wmask, trees, tmask,
                      player, step)
    tree_map, tree_age, enemy_pres, friend_pres = _board_maps(
        workers, wmask, trees, tmask, player)

    own_bases = [int(b) for b in bases[s] if int(b) >= 0]
    opp_bases = [int(b) for b in bases[o] if int(b) >= 0]
    own_trees = [int(t[0]) for t in trees[tmask] if int(t[2]) == s]
    ene_trees = [int(t[0]) for t in trees[tmask] if int(t[2]) == o]
    ene_pos = [int(workers[o, j, 0]) for j in range(workers.shape[1])
               if wmask[o, j]]

    W = np.zeros((workers.shape[1], WORKER_DIM))
    for j in range(workers.shape[1]):
        if not wmask[s, j]:
            continue
        pos = int(workers[s, j, 0])
        cargo = float(workers[s, j, 1])
        wtype = int(workers[s, j, 2])
        d_ob, ob_pos = _nearest(pos, own_bases)
        d_eb, _ = _nearest(pos, opp_bases)
        d_ot, ot_pos = _nearest(pos, own_trees)
        d_et, et_pos = _nearest(pos, ene_trees)
        dir_ob = _dirvec(pos, ob_pos)
        dir_ot = _dirvec(pos, ot_pos)
        dir_et = _dirvec(pos, et_pos)
        et_age = tree_age[et_pos] / 50.0 if et_pos >= 0 else 0.0
        d_ew, _ = _nearest(pos, ene_pos)
        fri_pos = [int(workers[s, k, 0]) for k in range(workers.shape[1])
                   if wmask[s, k] and k != j]
        d_fw, _ = _nearest(pos, fri_pos)
        # enemy cargo mass near this worker (collision stakes)
        near_enemy_cargo = 0.0
        for k in range(workers.shape[1]):
            if wmask[o, k] and DIST[pos, int(workers[o, k, 0])] <= 2:
                near_enemy_cargo += float(workers[o, k, 1])
        own_trees_near = sum(1 for tp in own_trees if DIST[pos, tp] <= 3)
        local = np.concatenate([
            [1.0 if wtype == 0 else 0.0,
             1.0 if wtype == 1 else 0.0,
             np.clip(cargo / 500.0, 0, 2) / 2.0,
             1.0 if cargo > 0 else 0.0,
             1.0 if cargo >= 150 else 0.0],
            np.clip(carbon[PATCH[pos]] / 100.0, 0, 1),
            tree_map[PATCH[pos]],
            enemy_pres[PATCH[pos]],
            friend_pres[PATCH3[pos]],
            [min(d_ob, 14) / 14.0, min(d_eb, 14) / 14.0,
             min(d_ot, 14) / 14.0, min(d_et, 14) / 14.0, et_age,
             1.0 if pos in own_bases else 0.0,
             1.0 if pos in opp_bases else 0.0,
             1.0 if tree_map[pos] > 0 else 0.0,
             1.0 if tree_map[pos] < 0 else 0.0,
             tree_age[pos] / 50.0,
             min(d_ew, 14) / 14.0, min(d_fw, 14) / 14.0,
             np.clip(near_enemy_cargo / 500.0, 0, 1),
             min(own_trees_near, 10) / 10.0,
             _xs[pos] / 14.0, _ys[pos] / 14.0,
             dir_ob[0], dir_ob[1], dir_ot[0], dir_ot[1],
             dir_et[0], dir_et[1]],
        ])
        W[j] = np.concatenate([g, local])
    if seq is None:
        seq = {"prev_n": 0.0, "lost": 0.0, "since": 0.0}
    own_pos = [int(workers[s, j, 0]) for j in range(workers.shape[1])
               if wmask[s, j]]
    B = np.zeros((2, BASE_DIM))
    for bi in range(2):
        bpos = int(bases[s, bi])
        if bpos < 0:
            continue
        d_ew, _ = _nearest(bpos, ene_pos)
        d_ot, _ = _nearest(bpos, own_trees)
        own_near = sum(1 for wp in own_pos if DIST[bpos, wp] <= 2)
        ene_near = sum(1 for ep in ene_pos if DIST[bpos, ep] <= 3)
        local = np.array([
            1.0 if bi == 1 else 0.0,
            1.0 if cash[s] >= 30 else 0.0,
            1.0 if cash[s] >= 60 else 0.0,
            (10 - int(np.sum(wmask[s]))) / 10.0,
            min(d_ew, 14) / 14.0,
            1.0 if step < 60 else 0.0,
            min(d_ot, 14) / 14.0,
            min(float(seq["prev_n"]), 10) / 10.0,
            min(float(seq["lost"]), 5) / 5.0,
            min(float(seq["since"]), 60) / 60.0,
            min(own_near, 5) / 5.0,
            min(ene_near, 5) / 5.0,
        ])
        B[bi] = np.concatenate([g, local])
    return W, B

# ===== belief.py (verbatim) =====
import numpy as np


SIZE = 15
N = SIZE * SIZE

# 4-neighbours in flat index space (UP, RIGHT, DOWN, LEFT as (dx, dy))
_NBR4 = np.zeros((N, 4), dtype=np.int64)
_MIRROR = np.zeros((N, 4), dtype=np.int64)
for _i in range(N):
    _x, _y = int(_xs[_i]), int(_ys[_i])
    for _k, (_dx, _dy) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        _NBR4[_i, _k] = _idx(_x + _dx, _y + _dy)
    _MIRROR[_i] = [_idx(_x, _y), _idx(14 - _x, _y), _idx(_x, 14 - _y),
                   _idx(14 - _x, 14 - _y)]

BASE_CELLS = frozenset(int(_idx(x, y)) for x, y in
                       ((4, 4), (4, 10), (10, 4), (10, 10)))

UNKNOWN_ENC = -0.25

WORKER_DIM2 = WORKER_DIM + 32
BASE_DIM2 = BASE_DIM + 3


class BeliefTracker:
    """Stateful belief map for ONE player in ONE game."""

    def __init__(self):
        self.belief = np.full(N, -1.0)
        for b in BASE_CELLS:
            self.belief[b] = 0.0
        self.prev_enemy_collectors = {}   # pos -> cargo
        self.last_step = None

    def reset(self):
        self.__init__()

    def update(self, carbon, workers, wmask, trees, tmask, player,
               step):
        """Advance the belief with one canonical observation (the acting
        player's view at `step`). Must be called once per step, in
        order, before featurize2 for that step."""
        if self.last_step is not None and step != self.last_step + 1:
            # sequence break (new game sliced in / missing step): the
            # stay-inference would be garbage; restart conservatively.
            self.prev_enemy_collectors = {}
        belief = self.belief
        updated = np.zeros(N, dtype=bool)
        carbon = np.asarray(carbon, dtype=np.float64)

        # --- own observations (+ symmetric mirrors into unknown) ---
        for j in range(workers.shape[1]):
            if not wmask[player, j]:
                continue
            pos = int(workers[player, j, 0])
            if int(workers[player, j, 2]) == 0:      # collector
                cells = (pos,)
            else:                                    # planter: 4-nbrs
                cells = tuple(int(c) for c in _NBR4[pos])
            for cell in cells:
                v = float(carbon[cell])
                belief[cell] = v
                updated[cell] = True
                for m in _MIRROR[cell]:
                    m = int(m)
                    if belief[m] < 0:
                        belief[m] = v
                        updated[m] = True

        # --- enemy collectors that stayed put ---
        opp = 1 - player
        new_enemy = {}
        for j in range(workers.shape[1]):
            if not wmask[opp, j] or int(workers[opp, j, 2]) != 0:
                continue
            pos = int(workers[opp, j, 0])
            cargo = float(workers[opp, j, 1])
            new_enemy[pos] = cargo
            prev = self.prev_enemy_collectors.get(pos)
            if prev is not None:
                if belief[pos] > 0:
                    belief[pos] *= 0.7
                inc = cargo - prev
                if inc > 0:
                    belief[pos] = inc / 0.3 * 0.7
                updated[pos] = True
        self.prev_enemy_collectors = new_enemy

        # --- trees: cell value + absorption decay on neighbours ---
        if np.any(tmask):
            collector_cells = set()
            for q in range(2):
                for j in range(workers.shape[1]):
                    if wmask[q, j] and int(workers[q, j, 2]) == 0:
                        collector_cells.add(int(workers[q, j, 0]))
            for pos, age, _owner in trees[tmask]:
                pos = int(pos)
                age = float(age)
                belief[pos] = 0.0 if age < 50 else 30.0 / 1.03
                updated[pos] = True
                if age > 1:
                    for nb in _NBR4[pos]:
                        nb = int(nb)
                        if nb not in collector_cells and belief[nb] > 0:
                            belief[nb] *= (1.0 - 0.0375)
                        updated[nb] = True

        # --- regrowth on everything not refreshed ---
        mask = (belief > 0) & (~updated)
        belief[mask] = np.minimum(belief[mask] * 1.03, 100.0)
        self.last_step = step
        return belief


def belief_globals(belief):
    known = belief >= 0
    frac = float(np.mean(known))
    total = float(np.sum(belief[known])) if np.any(known) else 0.0
    if np.any(known):
        vals = np.sort(belief[known])[::-1]
        theta = float(vals[min(6, len(vals) - 1)])
    else:
        theta = 0.0
    return frac, total / 3000.0, theta / 100.0


def featurize2(carbon, cash, bases, workers, wmask, trees, tmask,
               player, step, belief, seq=None):
    """bc_features.featurize + belief channels. `belief` is the (225,)
    array AFTER BeliefTracker.update for this step."""
    W, B = featurize(carbon, cash, bases, workers, wmask, trees, tmask,
                     player, step, seq=seq)
    belief = np.asarray(belief, dtype=np.float64)
    known = belief >= 0
    enc = np.where(known, np.clip(belief, 0, 100) / 100.0, UNKNOWN_ENC)
    gfrac, gsum, gtheta = belief_globals(belief)
    globals3 = np.array([gfrac, gsum, gtheta])
    if np.any(known):
        rich = int(np.argmax(np.where(known, belief, -1.0)))
    else:
        rich = -1
    W2 = np.zeros((W.shape[0], WORKER_DIM2))
    W2[:, :WORKER_DIM] = W
    for j in range(W.shape[0]):
        if not wmask[player, j]:
            continue
        pos = int(workers[player, j, 0])
        d_rich = float(DIST[pos, rich]) if rich >= 0 else 14.0
        dir_rich = _dirvec(pos, rich)
        W2[j, WORKER_DIM:] = np.concatenate([
            enc[PATCH[pos]],
            [enc[pos], min(d_rich, 14) / 14.0, dir_rich[0], dir_rich[1]],
            globals3,
        ])
    B2 = np.zeros((B.shape[0], BASE_DIM2))
    B2[:, :BASE_DIM] = B
    for bi in range(B.shape[0]):
        if int(bases[player, bi]) >= 0:
            B2[bi, BASE_DIM:] = globals3
    return W2, B2

# ===== rl2_common.py actor section (verbatim) =====
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



# ----------------------------------------------------------------------
# weights + agent factory (stage-2 glue). _suf/_canonical are
# shadow_alpha.py's, verbatim -- the same parsing the BC extraction
# and stage-1/2 training used.
# ----------------------------------------------------------------------
_SUFFIX = _re.compile(r"(\d+)$")


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
    return p, carbon, cash, bases, workers, wmask, trees, tmask, step, \
        my_wids, my_bids


_W2 = None


def _load_w2():
    global _W2
    if _W2 is None:
        z = np.load(_os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "rl2_weights.npz"))
        _W2 = {k: z[k] for k in z.files}
    return _W2


SHIELDS_DEFAULT = {'danger': True, 'endgame': True, 'opening': True}


def make_agent():
    actor = NumpyActor2(_load_w2(), shields=dict(SHIELDS_DEFAULT))
    fn = lambda observation, configuration: actor(observation,
                                                  configuration)
    fn.actor = actor
    return fn


_GLOBAL_ACTOR = None


def agent(observation, configuration):
    global _GLOBAL_ACTOR
    if _GLOBAL_ACTOR is None:
        _GLOBAL_ACTOR = NumpyActor2(_load_w2(),
                                    shields=dict(SHIELDS_DEFAULT))
    return _GLOBAL_ACTOR(observation, configuration)
