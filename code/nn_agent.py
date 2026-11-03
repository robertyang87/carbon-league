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
BASE_PILE_FROM = 15    # in the last N moves own units may share an own base cell


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
    for i, (uid, k, pos) in enumerate(units):
        if k == 2:
            continue
        z = logits[i, 0:5] / temp
        pr = np.exp(z - z.max()); pr /= pr.sum()
        cargo = mine[2][uid][1] if uid in mine[2] else 0.0
        if k == 0 and cargo > 0 and my_bases:
            b = min(my_bases, key=lambda q: int(F.tdist(pos, q)))
            d = int(F.tdist(pos, b))
            if d > 0 and left <= d + ENDGAME_SLACK:
                forced.append((d, i, pr, b))
                continue
        a0 = int(rng.choice(5, p=pr)) if rng is not None else int(np.argmax(pr))
        order.append((-pr[a0], i, pr, a0))
    forced.sort(key=lambda t: t[0])
    for d, i, pr, b in forced:
        uid, k, pos = units[i]
        cand = sorted(_toward(pos, b), key=lambda a: -pr[a]) + [int(a) for a in np.argsort(-pr)]
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
        cand = [a0] + [int(a) for a in np.argsort(-pr) if a != a0]
        for a in cand:
            tgt = F.nb(pos, F.MOVES[a])
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


def recruit_choice(lg, cash, nworkers, cap, n_pl, tree_price, step, blocked):
    """Rule-gated recruitment (BC argmax almost never recruits: class imbalance).
    Recruit whenever a slot is free and cash covers the cost plus a planting
    reserve while we field planters; the policy picks collector vs planter."""
    if blocked or nworkers >= cap or step >= RECRUIT_UNTIL:
        return 0
    reserve = tree_price if n_pl > 0 else 0.0
    if cash < 30 + min(reserve, 60.0):
        return 0
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
