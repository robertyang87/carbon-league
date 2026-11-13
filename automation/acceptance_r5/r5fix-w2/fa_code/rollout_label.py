"""rollout_label -- counterfactual (exact) move evaluation by branching the simulator: policy-improvement labels.

python3 rollout_label.py OUT_DIR N_SHARDS [GAMES_PER_SHARD=8] [SEED0=3000000]

Local games are deterministic (FastCarbon + greedy NN agents). In a game of v9 vs an NN opponent, at
sampled key moments (a collector of ours carrying cargo near enemy units, or empty next to a loaded enemy
collector), we clone simulator + both agents and try each of the 5 moves of that unit; both sides then play
on with their own policies for H steps and we score the branch by material margin
(cash + CARGO_W * carried carbon, ours minus theirs). The recorded sample is the unit's network input
(x, s), its policy logits and the 5 branch scores -> the best move is an exact improvement target.
Resumable by shard.
"""
import copy
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V9 = "/home/user/carbon-league/data/weights/nn_master64.npz"
V6 = "/home/user/carbon-league/data/weights/nn_alpha2_v1.npz"
POOL = [("SELF", 0.6), (V6, 0.4)]
H = int(os.environ.get("RL_H", "25"))
P_DECIDE = float(os.environ.get("RL_PDECIDE", "0.35"))
CARGO_W = 0.9


class NNAgent:
    """make_agent equivalent with copyable state (Tracker)."""

    def __init__(self, pol, me):
        import nnfeat as F
        self.pol, self.me, self.tr = pol, me, F.Tracker(me)
        self.last = None   # (units, xs, ss, logits) of the latest act()

    def clone(self):
        c = NNAgent.__new__(NNAgent)
        c.pol, c.me, c.tr, c.last = self.pol, self.me, copy.deepcopy(self.tr), None
        return c

    def act(self, obs, cfg, override=None):
        import nn_agent as A
        import nnfeat as F
        self.tr.update(obs)
        G, scal, info = self.tr.global_planes(obs)
        info["bel"] = self.tr.bel
        mine = obs["players"][self.me]
        units, xs, ss = [], [], []
        for wid, w in mine[2].items():
            k = 0 if w[2] == "COLLECTOR" else 1
            x, s = F.unit_inputs(G, scal, w[0], k, w[1])
            units.append((wid, k, w[0])); xs.append(x); ss.append(s)
        for bid, b in mine[1].items():
            x, s = F.unit_inputs(G, scal, b, 2, 0.0)
            units.append((bid, 2, b)); xs.append(x); ss.append(s)
        if not units:
            return {}
        X, S = np.stack(xs).astype(np.float32), np.stack(ss).astype(np.float32)
        logits = self.pol.forward(X, S)
        acts, _ = A.select_actions(units, logits, obs, info, 10)
        self.last = (units, X, S, logits)
        if override:
            uid, m = override
            if m == "NONE":
                acts.pop(uid, None)
            else:
                acts[uid] = m
        return acts


def material(sim, me):
    m = list(sim.cash)
    for w in sim.workers.values():
        m[w.player] += CARGO_W * w.carbon
    return m[me] - m[1 - me]


def play(task):
    seed, side, opp = task
    os.environ["OMP_NUM_THREADS"] = "1"
    import eval_v53
    import nn_agent as A
    import nn_search as NS
    import nnfeat as F
    from fastsim import FastCarbon
    A.ENDGAME_BRAVE = False
    A.STALL_OFF_LEFT = 30
    cfg = eval_v53.CFG
    rnd = random.Random(seed * 13 + side)
    random.seed(eval_v53.rng_seed(seed, side))
    sim = FastCarbon(cfg).reset(seed=seed)
    pol9 = A.Policy(A.load_weights(V9))
    polo = pol9 if opp == "SELF" else A.Policy(A.load_weights(opp))
    ag = [None, None]
    ag[side] = NNAgent(pol9, side); ag[1 - side] = NNAgent(polo, 1 - side)
    Xs, Ss, Ks, Ls, Vs, Ts = [], [], [], [], [], []
    for t in range(300):
        obs = [sim.obs(0), sim.obs(1)]
        keys = []
        if 5 <= t <= 299 - H and rnd.random() < P_DECIDE:
            keys = [u for u in NS._key_units(obs[side]) if obs[side]["players"][side][2][u][2] == "COLLECTOR"]
        if keys:
            u = keys[0]
            base_sim, base_ag, base_stall = copy.deepcopy(sim), [a.clone() for a in ag], dict(A._stall)
            scores = []
            for m in F.MOVES:
                s2 = copy.deepcopy(base_sim); a2 = [a.clone() for a in base_ag]
                A._stall.clear(); A._stall.update(base_stall)
                o2 = [s2.obs(0), s2.obs(1)]
                acts = [a2[p].act(o2[p], cfg, override=(u, m) if p == side else None) for p in (0, 1)]
                s2.step(acts[0], acts[1])
                for _ in range(H - 1):
                    o2 = [s2.obs(0), s2.obs(1)]
                    acts = [a2[p].act(o2[p], cfg) for p in (0, 1)]
                    s2.step(acts[0], acts[1])
                scores.append(material(s2, side))
            A._stall.clear(); A._stall.update(base_stall)
            acts = [ag[p].act(obs[p], cfg) for p in (0, 1)]
            units, X, S, logits = ag[side].last
            i = [uu[0] for uu in units].index(u)
            Xs.append(X[i]); Ss.append(S[i]); Ks.append(units[i][1]); Ls.append(logits[i]); Vs.append(scores); Ts.append(t)
        else:
            acts = [ag[p].act(obs[p], cfg) for p in (0, 1)]
        sim.step(acts[0], acts[1])
    margin = sim.cash[side] - sim.cash[1 - side]
    if not Xs:
        return None, margin
    return (np.stack(Xs).astype(np.float16), np.stack(Ss).astype(np.float16), np.array(Ks, np.int8),
            np.stack(Ls).astype(np.float32), np.array(Vs, np.float32), np.array(Ts, np.int16)), margin


def main(out_dir, n_shards, per=8, seed0=3000000):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed0)
    names, probs = zip(*POOL)
    for sh in range(n_shards):
        tasks = [(seed0 + sh * per + i, i % 2, rng.choices(names, probs)[0]) for i in range(per)]
        path = os.path.join(out_dir, f"shard_{sh:04d}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=int(os.environ.get("RL_WORKERS", "4"))) as ex:
            res = list(ex.map(play, tasks))
        parts = [r for r, _ in res if r is not None]
        X, S, K, L, V, T = (np.concatenate([p[i] for p in parts]) for i in range(6))
        np.savez(path + ".tmp.npz", X=X, Sc=S, kind=K, logits=L, scores=V, step=T)
        os.replace(path + ".tmp.npz", path)
        gap = V.max(1) - V[np.arange(len(V)), L[:, :5].argmax(1)]
        print(f"shard {sh}: {len(V)} decisions, policy-argmax already best {np.mean(gap < 1):.0%}, "
              f"mean gain if switched {gap.mean():.1f}, >50: {np.mean(gap > 50):.0%}, "
              f"margins {[round(m) for _, m in res]}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]) if len(a) > 2 else 8, int(a[3]) if len(a) > 3 else 3000000)
