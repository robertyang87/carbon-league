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


def conv(x, w, b):
    """circular 3x3 conv. x [B,C,15,15], w [O,C,3,3] -> [B,O,15,15]."""
    B, C = x.shape[:2]
    cols = np.empty((B, C, 9, 15, 15), dtype=np.float32)
    for k, (dr, dc) in enumerate(SHIFTS):
        cols[:, :, k] = np.roll(x, shift=(-dr, -dc), axis=(2, 3))
    cols = cols.reshape(B, C * 9, 225)
    wm = w.reshape(w.shape[0], C, 9).reshape(w.shape[0], C * 9)
    out = np.einsum("ok,bkn->bon", wm, cols, optimize=True) + b[None, :, None]
    return out.reshape(B, w.shape[0], 15, 15)


class Policy:
    def __init__(self, sd):
        self.sd = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items() if k != "meta"}
        self.nb = len({k.split(".")[1] for k in self.sd if k.startswith("blocks.")})

    def forward(self, x, s):
        p = self.sd
        h = np.maximum(conv(x, p["stem.weight"], p["stem.bias"]), 0)
        for i in range(self.nb):
            t = np.maximum(conv(h, p[f"blocks.{i}.0.weight"], p[f"blocks.{i}.0.bias"]), 0)
            h = np.maximum(h + conv(t, p[f"blocks.{i}.1.weight"], p[f"blocks.{i}.1.bias"]), 0)
        patch = h[:, :, 6:9, 6:9].reshape(h.shape[0], -1)
        g = h.mean(axis=(2, 3))
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
        acts = {}
        claimed = set()
        order = []
        for i, (uid, k, pos) in enumerate(units):
            if k == 2:
                continue
            lg = logits[i, 0:5]
            pr = np.exp(lg - lg.max()); pr /= pr.sum()
            order.append((-pr.max(), i, pr))
        order.sort(key=lambda t: t[0])
        for _, i, pr in order:
            uid, k, pos = units[i]
            for a in np.argsort(-pr):
                tgt = F.nb(pos, F.MOVES[a])
                if tgt not in claimed:
                    claimed.add(tgt)
                    if a != 0:
                        acts[uid] = F.MOVES[a]
                    break
            else:
                claimed.add(pos)
        cash = info["cash"]
        nworkers = len(mine[2])
        cap = _cfg(config, "workerLimit", 10)
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
            if a == 2:
                n_pl += 1
        return acts

    return agent


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
