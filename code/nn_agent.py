"""nn_agent -- NumPy serving of the egocentric CNN policy (nn_train_bc.Net).

make_agent(weights_path or dict) -> agent(observation, configuration).
Action selection on top of the policy:
  * own-collision guard: units claim target cells in order of confidence;
    a unit takes its best action whose target cell is not already claimed
    by an own unit (staying units claim their cell);
  * recruitment only if cash >= cost, worker cap not reached and the base
    cell is not claimed by an own unit after moves.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nnfeat as F  # noqa: E402

SHIFTS = [(ki - 1, kj - 1) for ki in range(3) for kj in range(3)]
TTA = 1   # test-time augmentation: 1 (off), 2 or 4 symmetric views averaged


def _wmat(w):
    """torch [O,C,3,3] -> [(ki*3+kj)*C + c, O] for channels-last im2col."""
    O, C = w.shape[:2]
    return np.ascontiguousarray(w.transpose(2, 3, 1, 0).reshape(9 * C, O))


def conv(x, wm, b):
    """circular 3x3 conv, channels-last. x [B,15,15,C], wm from _wmat -> [B,15,15,O]."""
    B, H, W_, C = x.shape
    cols = np.empty((B, H, W_, 9 * C), dtype=np.float32)
    for k, (dr, dc) in enumerate(SHIFTS):
        cols[..., k * C:(k + 1) * C] = np.roll(x, shift=(-dr, -dc), axis=(1, 2))
    out = cols.reshape(B * H * W_, 9 * C) @ wm
    out += b
    return out.reshape(B, H, W_, wm.shape[1])


class Policy:
    def __init__(self, sd):
        self.sd = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items() if k != "meta"}
        self.nb = len({k.split(".")[1] for k in self.sd if k.startswith("blocks.")})
        self.wm = {k: _wmat(v) for k, v in self.sd.items() if k.endswith(".weight") and v.ndim == 4}

    def forward(self, x, s):
        """logits; with TTA > 1, average over dihedral symmetries of the torus view
        (centre (7,7) is fixed by the flips; move logits are mapped back)."""
        if TTA <= 1:
            return self._forward(x, s)
        outs = []
        for k, (fr, fc) in enumerate(((False, False), (True, False), (False, True), (True, True))[:TTA]):
            xt = x
            if fr:
                xt = xt[:, :, ::-1, :]
            if fc:
                xt = xt[:, :, :, ::-1]
            lg = self._forward(np.ascontiguousarray(xt), s)
            perm = list(range(8))
            if fr:                      # UP<->DOWN
                perm[1], perm[3] = perm[3], perm[1]
            if fc:                      # RIGHT<->LEFT
                perm[2], perm[4] = perm[4], perm[2]
            outs.append(lg[:, perm])
        return np.mean(outs, axis=0)

    def _forward(self, x, s):
        p = self.sd; wm = self.wm
        x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        h = np.maximum(conv(x, wm["stem.weight"], p["stem.bias"]), 0)
        for i in range(self.nb):
            t = np.maximum(conv(h, wm[f"blocks.{i}.0.weight"], p[f"blocks.{i}.0.bias"]), 0)
            h = np.maximum(h + conv(t, wm[f"blocks.{i}.1.weight"], p[f"blocks.{i}.1.bias"]), 0)
        patch = h[:, 6:9, 6:9, :].transpose(0, 3, 1, 2).reshape(h.shape[0], -1)
        g = h.mean(axis=(1, 2))
        z = np.concatenate([patch, g, s], axis=1)
        z = np.maximum(z @ p["fc.weight"].T + p["fc.bias"], 0)
        return z @ p["pi.weight"].T + p["pi.bias"]


def load_weights(src):
    if isinstance(src, dict):
        return src
    d = np.load(src, allow_pickle=False)
    return {k: d[k] for k in d.files}


def make_agent(weights=None):
    if weights is None:
        weights = os.path.join(HERE, "nn_bc_weights.npz")
    pol = Policy(load_weights(weights))
    state = {"tr": None}

    def agent(obs, config):
        me = obs["player"]
        if obs["step"] == 0 or state["tr"] is None:
            state["tr"] = F.Tracker(me)
        tr = state["tr"]
        tr.update(obs)
        G, scal, info = tr.global_planes(obs)
        info["bel"] = tr.bel
        mine = obs["players"][me]
        units = []
        xs, ss = [], []
        for wid, w in mine[2].items():
            k = 0 if w[2] == "COLLECTOR" else 1
            x, s = F.unit_inputs(G, scal, w[0], k, w[1])
            units.append((wid, k, w[0])); xs.append(x); ss.append(s)
        for bid, b in mine[1].items():
            x, s = F.unit_inputs(G, scal, b, 2, 0.0)
            units.append((bid, 2, b)); xs.append(x); ss.append(s)
        if not units:
            return {}
        logits = pol.forward(np.stack(xs).astype(np.float32), np.stack(ss).astype(np.float32))
        acts, _ = select_actions(units, logits, obs, info, _cfg(config, "workerLimit", 10))
        return acts

    return agent


ENDGAME_SLACK = 1      # start the forced return when moves left <= dist + slack
SHIELD = True          # loaded collectors avoid cells a lighter/equal enemy collector can reach
BASE_PILE_FROM = 15    # in the last N moves own units may share an own base cell
# v3 anti-camping (ladder loss vs ds_cov18_m_v1e: 2 empty enemy collectors on the diagonals of each
# base threatened every approach cell, SHIELD kept loaded collectors parked at d=2 for 180 steps,
# and own planters idled on the base cells).
STALL_K = 3            # loaded collector within STALL_R of a base and not getting closer for K steps
STALL_R = 3            #   -> forced home run with SHIELD off (0 disables)
ENDGAME_BRAVE = True   # endgame forced return ignores SHIELD (undelivered cargo scores 0)
PLANTER_OFF_BASE = True  # planters never stay on / step onto an own base cell (before the pile window)
_stall = {}            # (player, uid) -> (last dist, count, step)
# --- tree manager (rule-computed planter control; off = v8 behaviour) ---
TREE_MGR = False
TOLL_AVOID = False     # loaded collectors prefer cells outside enemy trees' 8-neighbourhood (10%/step toll)
TM_RADIUS = 4          # planter target search radius (torus manhattan)
TM_STEP_COST = 8.0     # value of one planter step
TM_DENY = 0.7          # weight of denying an enemy tree's future income when seizing
TM_SHARED = 0.5        # discount for neighbour cells already drained by another tree
ABS_CELL, LIFESPAN, SEIZE_COST = 0.0375, 50, 20.0


def _tree_value(cell, L, bel, tree_cells, shared, bases):
    if L <= 0:
        return 0.0
    keep = 1.0 - (1.0 - ABS_CELL) ** L
    v = 0.0
    for n in F.NB4[cell]:
        n = int(n)
        if n in tree_cells or n in bases:
            continue
        x = float(bel[n])
        if n in shared:
            x *= TM_SHARED
        v += x * keep
    return v


_tm_commit = {}        # (player, uid) -> committed target cell


def tree_plan(obs, info, left):
    """Return {planter uid: target cell} chosen greedily by rule-computed value.
    Targets are committed until reached (planted/seized) or invalid; new targets only when cash covers the price."""
    me = obs["player"]
    if obs["step"] <= 1:
        for k_ in [k_ for k_ in _tm_commit if k_[0] == me]:
            del _tm_commit[k_]
    P = obs["players"]
    bel = info["bel"]
    price = info["tree_price"]
    bases = set(P[0][1].values()) | set(P[1][1].values())
    tree_cells, my_tree_cells, en_trees = set(), set(), {}
    for p in (0, 1):
        for tid, tv in P[p][3].items():
            tree_cells.add(tv[0])
            if p == me:
                my_tree_cells.add(tv[0])
            else:
                en_trees[tv[0]] = tv[1]
    shared = set()
    for c in tree_cells:
        shared.update(int(n) for n in F.NB4[c])
    planters = [(uid, w[0]) for uid, w in P[me][2].items() if w[2] != "COLLECTOR"]
    alive = {uid for uid, _ in planters}
    for k_ in [k_ for k_ in _tm_commit if k_[0] == me and k_[1] not in alive]:
        del _tm_commit[k_]
    plan, taken = {}, set()
    for uid, pos in planters:                      # keep valid commitments
        c = _tm_commit.get((me, uid))
        if c is None:
            continue
        valid = (c in en_trees) or (c not in tree_cells and c not in bases)
        if pos == c and c in my_tree_cells:        # done: planted / seized
            valid = False
        if valid:
            plan[uid] = c; taken.add(c); taken.update(int(n) for n in F.NB4[c])
        else:
            del _tm_commit[(me, uid)]
    if info["cash"] < min(price, SEIZE_COST):
        return plan
    cands = []
    for uid, pos in planters:
        if uid in plan:
            continue
        for c in range(F.N):
            d = int(F.tdist(pos, c))
            if d > TM_RADIUS or c in bases:
                continue
            if c in en_trees:
                age = en_trees[c]
                L = min(LIFESPAN - age - d, left - d)
                v = _tree_value(c, L, bel, tree_cells, shared, bases) * (1.0 + TM_DENY) - SEIZE_COST
            elif c in tree_cells:
                continue
            else:
                L = min(LIFESPAN, left - d - 1)
                if L < 8:
                    continue
                v = _tree_value(c, L, bel, tree_cells, shared, bases) - price
            sc = v - TM_STEP_COST * d
            if sc > 0:
                cands.append((sc, uid, c))
    cands.sort(reverse=True)
    for sc, uid, c in cands:
        if uid in plan or c in taken:
            continue
        plan[uid] = c
        _tm_commit[(me, uid)] = c
        taken.add(c)
        taken.update(int(n) for n in F.NB4[c])     # keep new trees from sharing neighbour cells
    return plan



def _toward(pos, target):
    """moves (indices into F.MOVES) that reduce torus distance pos->target."""
    r, c = divmod(pos, F.S); tr_, tc = divmod(target, F.S)
    dr = (tr_ - r) % F.S; dc = (tc - c) % F.S
    out = []
    if dr:
        out.append(3 if dr <= F.S // 2 else 1)   # DOWN / UP
    if dc:
        out.append(2 if dc <= F.S // 2 else 4)   # RIGHT / LEFT
    return out


def select_actions(units, logits, obs, info, cap, rng=None, temp=1.0):
    """Shared by nn_agent (rng=None: greedy) and nn_rl (sampling).
    units: list of (uid, kind, pos); logits [n, 8].
    Returns (acts dict, chosen list of (unit index, move index, from_policy) for workers)."""
    me = obs["player"]
    mine = obs["players"][me]
    my_bases = list(mine[1].values())
    left = 299 - obs["step"]                       # moves still to be made
    pile = left <= BASE_PILE_FROM
    acts, chosen = {}, []
    claimed = set()
    forced, order = [], []
    # threat[c] = lightest enemy collector cargo that can be on cell c next step
    thr = np.full(F.N, np.inf)
    if SHIELD:
        for w in obs["players"][1 - me][2].values():
            if w[2] == "COLLECTOR":
                for c in (w[0], *F.NB4[w[0]]):
                    if w[1] < thr[c]:
                        thr[c] = w[1]
    cargo_of = {uid: (w[1] if w[2] == "COLLECTOR" else None) for uid, w in mine[2].items()}
    toll = set()
    if TOLL_AVOID:
        for tv in obs["players"][1 - me][3].values():
            r0, c0 = divmod(tv[0], F.S)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    toll.add(((r0 + dr) % F.S) * F.S + (c0 + dc) % F.S)
    plan = tree_plan(obs, info, left) if (TREE_MGR and "bel" in info) else {}
    en_reach = set()
    if plan:
        for w in obs["players"][1 - me][2].values():
            if w[2] == "COLLECTOR":
                en_reach.add(w[0]); en_reach.update(int(n) for n in F.NB4[w[0]])
    step = obs["step"]
    if step <= 1:
        for key in [k_ for k_ in _stall if k_[0] == me]:
            del _stall[key]
    brave = set()

    def safe_first(uid, cand):
        c = cargo_of.get(uid)
        if not SHIELD or c is None or c <= 0 or uid in brave:
            return cand
        pos = mine[2][uid][0]
        if toll and c > 50:
            return sorted(cand, key=lambda a: (thr[F.nb(pos, F.MOVES[a])] <= c, F.nb(pos, F.MOVES[a]) in toll))
        return sorted(cand, key=lambda a: thr[F.nb(pos, F.MOVES[a])] <= c)   # stable: safe moves first
    for i, (uid, k, pos) in enumerate(units):
        if k == 2:
            continue
        z = logits[i, 0:5] / temp
        pr = np.exp(z - z.max()); pr /= pr.sum()
        cargo = mine[2][uid][1] if uid in mine[2] else 0.0
        if k == 0 and cargo > 0 and my_bases:
            b = min(my_bases, key=lambda q: int(F.tdist(pos, q)))
            d = int(F.tdist(pos, b))
            if STALL_K:
                key = (me, uid)
                pd, cnt, ps = _stall.get(key, (99, 0, -9))
                if ps == step - 1 and cnt >= STALL_K and d > 0:
                    pass                                   # stay brave until the cargo is home
                elif (ps == step - 1 and 0 < d <= STALL_R and d >= pd and SHIELD
                      and any(thr[F.nb(pos, F.MOVES[a])] <= cargo for a in _toward(pos, b))):
                    cnt += 1
                else:
                    cnt = 0
                _stall[key] = (d, cnt, step)
                if cnt >= STALL_K:
                    brave.add(uid)
                    forced.append((d, i, pr, b))
                    continue
            if d > 0 and left <= d + ENDGAME_SLACK:
                if ENDGAME_BRAVE:
                    brave.add(uid)
                forced.append((d, i, pr, b))
                continue
        a0 = int(rng.choice(5, p=pr)) if rng is not None else int(np.argmax(pr))
        order.append((-pr[a0], i, pr, a0))
    forced.sort(key=lambda t: t[0])
    for d, i, pr, b in forced:
        uid, k, pos = units[i]
        cand = safe_first(uid, sorted(_toward(pos, b), key=lambda a: -pr[a]) + [int(a) for a in np.argsort(-pr)])
        for a in cand:
            tgt = F.nb(pos, F.MOVES[a])
            if tgt not in claimed or (pile and tgt in my_bases):
                claimed.add(tgt)
                if a:
                    acts[uid] = F.MOVES[a]
                chosen.append((i, a, False))
                break
    order.sort(key=lambda t: t[0])
    for _, i, pr, a0 in order:
        uid, k, pos = units[i]
        cand = safe_first(uid, [a0] + [int(a) for a in np.argsort(-pr) if a != a0])
        if k == 1 and uid in plan:
            tc = plan[uid]
            mv = [0] if pos == tc else sorted(_toward(pos, tc), key=lambda a: F.nb(pos, F.MOVES[a]) in en_reach)
            cand = mv + [a for a in cand if a not in mv]
        for a in cand:
            tgt = F.nb(pos, F.MOVES[a])
            if PLANTER_OFF_BASE and k == 1 and not pile and tgt in my_bases:
                continue
            if tgt not in claimed or (pile and k == 0 and tgt in my_bases):
                claimed.add(tgt)
                if a:
                    acts[uid] = F.MOVES[a]
                chosen.append((i, a, True))
                break
        else:
            claimed.add(pos)
    cash = info["cash"]
    nworkers = len(mine[2])
    n_pl = sum(1 for w in mine[2].values() if w[2] != "COLLECTOR")
    for i, (uid, k, pos) in enumerate(units):
        if k != 2:
            continue
        a = recruit_choice(logits[i, 5:8], cash, nworkers, cap, n_pl, info["tree_price"],
                           obs["step"], pos in claimed)
        if a == 0:
            continue
        acts[uid] = F.BASE_ACTS[a]
        cash -= 30; nworkers += 1; claimed.add(pos)
        n_pl += (a == 2)
    return acts, chosen


RECRUIT_UNTIL = 285   # steps; later recruits cannot pay back
MIN_COLLECTORS = 2
MAX_PLANTERS = 4
EARLY_COL_UNTIL = 0     # before this step: recruit collectors only (no tree reserve) while collectors < EARLY_COL_N
EARLY_COL_N = 6
PLANTER_FLOOR = 0       # recruit a planter while planters < floor (collector floor already met)
COL_MARGIN = 0         # collector floor = max(MIN_COLLECTORS, planters + COL_MARGIN)


def recruit_choice(lg, cash, nworkers, cap, n_pl, tree_price, step, blocked):
    """Rule-gated recruitment (BC argmax almost never recruits: class imbalance).
    Recruit whenever a slot is free and cash covers the cost plus a planting
    reserve while we field planters; the policy picks collector vs planter."""
    if blocked or nworkers >= cap or step >= RECRUIT_UNTIL:
        return 0
    n_col = nworkers - n_pl
    if step < EARLY_COL_UNTIL and n_col < EARLY_COL_N:   # opening collector rush (Godslayer-style economy)
        return 1 if cash >= 30 else 0
    if n_col == 0:                      # no income without collectors: buy one at any cash >= 30
        return 1 if cash >= 30 else 0
    reserve = tree_price if n_pl > 0 else 0.0
    if cash < 30 + min(reserve, 60.0):
        return 0
    # collector floor / planter cap: a policy that only ever picks planters starves (seen in RL v5b,
    # seed 640002: 0 collectors from step ~50, lost by 16.8k)
    if n_col < max(MIN_COLLECTORS, n_pl + COL_MARGIN) or n_pl >= MAX_PLANTERS:
        return 1
    if n_pl < PLANTER_FLOOR:
        return 2
    return 1 if lg[1] >= lg[2] else 2


_AGENT = None


def agent(observation, configuration):
    global _AGENT
    if _AGENT is None or observation["step"] == 0:
        _AGENT = make_agent()
    return _AGENT(observation, configuration)


def _cfg(config, key, default):
    try:
        v = config.get(key) if hasattr(config, "get") else getattr(config, key, None)
    except Exception:
        v = None
    return int(v) if v is not None else default
