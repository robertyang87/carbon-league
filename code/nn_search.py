"""nn_search -- one-step lookahead on top of the NN policy, scored by a learned state-value net.

Per step:
  1. NN policy + rule layer give default actions for us (acts0) and, from the opponent's point of view,
     a prediction of the opponent's actions (acts_opp).
  2. Key units (loaded collectors near enemy units, empty collectors near loaded enemy collectors,
     planters near enemy planters/trees) try each of their 5 moves: the joint action is simulated
     exactly for one step in FastCarbon (built from our observation + belief carbon map) and the
     resulting position is scored by the value net from our point of view.
  3. A move replaces the default only if it beats the current choice by > SEARCH_EPS.
Time-guarded: search stops when SEARCH_BUDGET seconds are used.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nnfeat as F  # noqa: E402
import nn_agent as A  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

SEARCH_EPS = float(os.environ.get("SEARCH_EPS", "0.002"))    # value units (margin / 5000): 10 points
SEARCH_BUDGET = float(os.environ.get("SEARCH_BUDGET", "0.30"))
SCORE = os.environ.get("SEARCH_SCORE", "value")   # "value" (learned value net) or "material" (exact 1-step material)
CARGO_W = 0.9          # material value of carried carbon relative to cash
KEY_R = 2
MAX_KEY = 6


class ValueNet:
    def __init__(self, sd):
        self.sd = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items() if k != "meta"}
        self.nb = len({k.split(".")[1] for k in self.sd if k.startswith("blocks.")})
        self.wm = {k: A._wmat(v) for k, v in self.sd.items() if k.endswith(".weight") and v.ndim == 4}

    def __call__(self, G, S):
        """G [B, C, 15, 15], S [B, N_SCAL] -> [B]"""
        p, wm = self.sd, self.wm
        x = np.ascontiguousarray(G.transpose(0, 2, 3, 1).astype(np.float32))
        h = np.maximum(A.conv(x, wm["stem.weight"], p["stem.bias"]), 0)
        for i in range(self.nb):
            t = np.maximum(A.conv(h, wm[f"blocks.{i}.0.weight"], p[f"blocks.{i}.0.bias"]), 0)
            h = np.maximum(h + A.conv(t, wm[f"blocks.{i}.1.weight"], p[f"blocks.{i}.1.bias"]), 0)
        g = h.mean(axis=(1, 2))
        z = np.maximum(np.concatenate([g, S.astype(np.float32)], 1) @ p["fc.weight"].T + p["fc.bias"], 0)
        return (z @ p["out.weight"].T + p["out.bias"])[:, 0]


def _policy_actions(pol, tr, obs, cfg):
    G, scal, info = tr.global_planes(obs)
    me = obs["player"]
    mine = obs["players"][me]
    units, xs, ss = [], [], []
    for wid, w in mine[2].items():
        k = 0 if w[2] == "COLLECTOR" else 1
        x, s = F.unit_inputs(G, scal, w[0], k, w[1])
        units.append((wid, k, w[0])); xs.append(x); ss.append(s)
    for bid, b in mine[1].items():
        x, s = F.unit_inputs(G, scal, b, 2, 0.0)
        units.append((bid, 2, b)); xs.append(x); ss.append(s)
    if not units:
        return {}, info
    logits = pol.forward(np.stack(xs).astype(np.float32), np.stack(ss).astype(np.float32))
    acts, _ = A.select_actions(units, logits, obs, info, A._cfg(cfg, "workerLimit", 10))
    return acts, info


def _key_units(obs):
    me = obs["player"]
    P = obs["players"]
    en = list(P[1 - me][2].values())
    en_trees = [tv[0] for tv in P[1 - me][3].values()]
    keys = []
    for uid, w in P[me][2].items():
        pos, cargo, kind = w
        near = [e for e in en if int(F.tdist(pos, e[0])) <= KEY_R]
        if kind == "COLLECTOR":
            if cargo > 0 and near:
                keys.append((cargo, uid))
            elif cargo == 0 and any(e[2] == "COLLECTOR" and e[1] >= 50 for e in near):
                keys.append((max(e[1] for e in near), uid))
        else:
            if any(e[2] != "COLLECTOR" for e in near) or any(int(F.tdist(pos, t)) <= 1 for t in en_trees):
                keys.append((1.0, uid))
    keys.sort(reverse=True)
    return [u for _, u in keys[:MAX_KEY]]


def make_search_agent(weights, vweights):
    pol = A.Policy(A.load_weights(weights))
    vnet = ValueNet(A.load_weights(vweights))
    st = {"tr": None, "tro": None}
    stats = {"steps": 0, "changed": 0, "evals": 0}

    def agent(obs, config):
        t0 = time.time()
        me = obs["player"]
        if obs["step"] == 0 or st["tr"] is None:
            st["tr"] = F.Tracker(me); st["tro"] = F.Tracker(1 - me)
        tr, tro = st["tr"], st["tro"]
        tr.update(obs)
        acts, info = _policy_actions(pol, tr, obs, config)
        keys = _key_units(obs)
        if SCORE == "material":   # 1-step material would just veto planting/seizing (cash spent now, value later)
            keys = [u for u in keys if obs["players"][me][2][u][2] == "COLLECTOR"]
        if not keys or obs["step"] >= 298:
            return acts
        # opponent prediction from its own point of view (approximate visibility)
        obs_o = dict(obs); obs_o["player"] = 1 - me
        tro.update(obs_o)
        acts_o, _ = _policy_actions(pol, tro, obs_o, config)
        full = {"step": obs["step"], "players": obs["players"], "trees": obs.get("trees") or {},
                "full_carbon": [float(v) for v in tr.bel], "carbon": obs["carbon"],
                "remainingOverageTime": obs.get("remainingOverageTime", 5)}
        for b in obs["players"][0][1].values():
            full["full_carbon"][b] = 0.0
        for b in obs["players"][1][1].values():
            full["full_carbon"][b] = 0.0
        sim = FastCarbon()

        def material():
            s_ = sim.get_state()
            m = [s_["cash"][p] for p in (0, 1)]
            for w in sim.workers.values():
                m[w.player] += CARGO_W * w.carbon
            return (m[me] - m[1 - me]) / 5000.0

        def value_of(cands):
            Gs, Ss = [], []
            if SCORE == "material":
                out = []
                for a in cands:
                    sim.load_observation(full)
                    if me == 0:
                        sim.step(a, acts_o)
                    else:
                        sim.step(acts_o, a)
                    out.append(material())
                stats["evals"] += len(cands)
                return np.array(out)
            for a in cands:
                sim.load_observation(full)
                if me == 0:
                    sim.step(a, acts_o)
                else:
                    sim.step(acts_o, a)
                nob = sim.obs(me)
                G, scal, _ = tr.global_planes(nob)
                Gs.append(G.reshape(F.C_PLANES, 15, 15)); Ss.append(scal)
            stats["evals"] += len(cands)
            return vnet(np.stack(Gs), np.stack(Ss))

        cur = dict(acts)
        best_v = float(value_of([cur])[0])
        for uid in keys:
            if time.time() - t0 > SEARCH_BUDGET:
                break
            cands, moves = [], []
            for m in F.MOVES:
                a = dict(cur)
                if m == "NONE":
                    a.pop(uid, None)
                else:
                    a[uid] = m
                if a == cur:
                    continue
                cands.append(a); moves.append(m)
            if not cands:
                continue
            vs = value_of(cands)
            j = int(np.argmax(vs))
            if vs[j] > best_v + SEARCH_EPS:
                cur = cands[j]; best_v = float(vs[j]); stats["changed"] += 1
        stats["steps"] += 1
        return cur

    agent.stats = stats
    return agent
