"""gen_value2 -- large self-play / league data for the value net v2 (resumable, sharded).

python3 gen_value2.py OUT_DIR N_SHARDS [GAMES_PER_SHARD=16] [SEED0=5000000]
env: SP_WORKERS (default 2), VAL_STRIDE (default 2)

Our side = v9 (nn_master64 + rule layer, ENDGAME_BRAVE=False, STALL_OFF_LEFT=30). Opponent drawn from POOL.
Every NN side records every STRIDE-th step from its own partial observation (nnfeat.Tracker, exactly as at
inference): global planes G, scalars Sc. Labels: final margin from that side's view / 5000.
Extra per-state columns: pol (0 = v9, 1 = alpha2, 2 = wide64, 3 = other), mat = true material
[cash_me, cash_en, cargo_me, cargo_en] (for residual / horizon targets), gid, step.
Per-game: cash trajectory traj [n_games, 301, 4] (cash0, cash1, cargo0, cargo1 after each step).
Exploration: in half of the games the v9 side replaces one random unit's move by a random move with
probability EPS per step (states just off the policy distribution, as visited by search branches).
"""
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
WD = os.path.join(ROOT, "data", "weights")
V9 = os.path.join(WD, "nn_master64.npz")
POOL = [("SELF", 0.35), ("NN:" + os.path.join(WD, "nn_alpha2_v1.npz"), 0.15),
        ("NN:" + os.path.join(WD, "nn_wide64_alpha.npz"), 0.15), ("nocamp.py", 0.2), ("base_v53.py", 0.15)]
STRIDE = int(os.environ.get("VAL_STRIDE", "2"))
EPS = 0.03


def pol_id(name):
    if name in ("V9", "SELF"):
        return 0
    if "alpha2" in name:
        return 1
    if "wide64" in name:
        return 2
    return 3


def play(task):
    seed, side, opp, explore = task
    import eval_v53
    import nn_agent as A
    import nnfeat as F
    from fastsim import FastCarbon
    A.ENDGAME_BRAVE = False
    A.STALL_OFF_LEFT = 30
    random.seed(eval_v53.rng_seed(seed, side))
    rnd = np.random.default_rng(seed * 7 + side)
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    agents, pids = [None, None], [None, None]
    agents[side] = A.make_agent(A.load_weights(V9)); pids[side] = 0
    if opp == "SELF":
        agents[1 - side] = A.make_agent(A.load_weights(V9)); pids[1 - side] = 0
    elif opp.startswith("NN:"):
        agents[1 - side] = A.make_agent(A.load_weights(opp[3:])); pids[1 - side] = pol_id(opp)
    else:
        agents[1 - side] = eval_v53.load_agent_fn(os.path.join(HERE, opp))
    trs = [F.Tracker(p) if pids[p] is not None else None for p in (0, 1)]
    rec = {p: ([], [], [], []) for p in (0, 1)}
    traj = np.zeros((301, 4), np.float32)
    traj[0, :2] = sim.cash
    for t in range(300):
        obs = [sim.obs(0), sim.obs(1)]
        for p in (0, 1):
            if trs[p] is None:
                continue
            trs[p].update(obs[p])
            if t % STRIDE == 0:
                G, scal, _ = trs[p].global_planes(obs[p])
                rec[p][0].append(G.reshape(F.C_PLANES, 15, 15).astype(np.float16))
                rec[p][1].append(scal.astype(np.float16))
                rec[p][2].append(t)
                cg = [0.0, 0.0]
                for w in sim.workers.values():
                    cg[w.player] += w.carbon
                rec[p][3].append([sim.cash[p], sim.cash[1 - p], cg[p], cg[1 - p]])
        acts = [agents[p](obs[p], eval_v53.CFG) for p in (0, 1)]
        if explore and rnd.random() < EPS:
            ws = [wid for wid, w in obs[side]["players"][side][2].items()]
            if ws:
                wid = ws[rnd.integers(len(ws))]
                m = F.MOVES[rnd.integers(5)]
                if m == "NONE":
                    acts[side].pop(wid, None)
                else:
                    acts[side][wid] = m
        sim.step(acts[0], acts[1])
        cg = [0.0, 0.0]
        for w in sim.workers.values():
            cg[w.player] += w.carbon
        traj[t + 1] = [sim.cash[0], sim.cash[1], cg[0], cg[1]]
    cash = sim.cash
    out = []
    for p in (0, 1):
        if trs[p] is None:
            continue
        y = (cash[p] - cash[1 - p]) / 5000.0
        n = len(rec[p][2])
        out.append((np.stack(rec[p][0]), np.stack(rec[p][1]), np.full(n, y, np.float32),
                    np.array(rec[p][2], np.int16), np.full(n, pids[p], np.int8),
                    np.array(rec[p][3], np.float32), np.full(n, p, np.int8)))
    return out, traj, (seed, side, opp, explore, cash[side] - cash[1 - side])


def main(out_dir, n_shards, per=16, seed0=5000000):
    os.makedirs(out_dir, exist_ok=True)
    names, probs = zip(*POOL)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SP_WORKERS", "2"))) as ex:
        for sh in range(n_shards):
            path = os.path.join(out_dir, f"shard_{sh:04d}.npz")
            if os.path.exists(path):
                continue
            rng = random.Random(seed0 + sh)
            tasks = [(seed0 + sh * per + i, i % 2, rng.choices(names, probs)[0], i % 4 >= 2) for i in range(per)]
            t0 = time.time()
            res = list(ex.map(play, tasks))
            cols = [[] for _ in range(8)]
            for k, (parts, _, _) in enumerate(res):
                for part in parts:
                    for i in range(7):
                        cols[i].append(part[i])
                    cols[7].append(np.full(len(part[2]), tasks[k][0], np.int32))
            G, S, y, st, pol, mat, persp, gid = (np.concatenate(c) for c in cols)
            traj = np.stack([r[1] for r in res])
            meta = np.array([[r[2][0], r[2][1], names.index(r[2][2]), int(r[2][3]), r[2][4]] for r in res], np.float64)
            np.savez_compressed(path + ".tmp.npz", G=G, Sc=S, y=y, step=st, pol=pol, mat=mat, persp=persp, gid=gid,
                                traj=traj, meta=meta)
            os.replace(path + ".tmp.npz", path)
            marg = [r[2][4] for r in res]
            print(f"shard {sh}: {len(y)} states, {len(res)} games, mean margin {np.mean(marg):+.0f}, "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]) if len(a) > 2 else 16, int(a[3]) if len(a) > 3 else 5000000)
