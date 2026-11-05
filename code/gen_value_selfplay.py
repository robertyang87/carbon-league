"""gen_value_selfplay -- self-play games on FastCarbon -> (global planes, scalars, final margin) shards.

python3 gen_value_selfplay.py OUT_DIR N_SHARDS [GAMES_PER_SHARD=16] [SEED0=2000000]
Our side is v9 (v8 network + endgame-shield rules). Opponent drawn from POOL. Every step of every NN
side is recorded from that side's own partial observation (nnfeat.Tracker, exactly as at inference);
label = final cash margin from that side's view / 5000. Resumable: existing shards are skipped.
"""
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
POOL = [("SELF", 0.4), ("NN:" + V6, 0.2), ("nocamp.py", 0.2), ("base_v53.py", 0.2)]


def play(task):
    seed, side, opp = task
    os.environ["OMP_NUM_THREADS"] = "1"
    import eval_v53
    import nn_agent as A
    import nnfeat as F
    from fastsim import FastCarbon
    A.ENDGAME_BRAVE = False
    A.STALL_OFF_LEFT = 30
    random.seed(eval_v53.rng_seed(seed, side))
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    agents, nn_side = [None, None], [False, False]
    agents[side] = A.make_agent(A.load_weights(V9)); nn_side[side] = True
    if opp == "SELF":
        agents[1 - side] = A.make_agent(A.load_weights(V9)); nn_side[1 - side] = True
    elif opp.startswith("NN:"):
        agents[1 - side] = A.make_agent(A.load_weights(opp[3:])); nn_side[1 - side] = True
    else:
        agents[1 - side] = eval_v53.load_agent_fn(os.path.join(HERE, opp))
    trs = [F.Tracker(p) if nn_side[p] else None for p in (0, 1)]
    rec = {0: ([], [], []), 1: ([], [], [])}
    for t in range(300):
        obs = [sim.obs(0), sim.obs(1)]
        for p in (0, 1):
            if trs[p] is not None:
                trs[p].update(obs[p])
                G, scal, _ = trs[p].global_planes(obs[p])
                rec[p][0].append(G.reshape(F.C_PLANES, 15, 15).astype(np.float16))
                rec[p][1].append(scal.astype(np.float16)); rec[p][2].append(t)
        acts = [agents[p](obs[p], eval_v53.CFG) for p in (0, 1)]
        sim.step(acts[0], acts[1])
    cash = sim.get_state()["cash"]
    out = []
    for p in (0, 1):
        if trs[p] is None:
            continue
        y = (cash[p] - cash[1 - p]) / 5000.0
        n = len(rec[p][2])
        out.append((np.stack(rec[p][0]), np.stack(rec[p][1]), np.full(n, y, np.float32),
                    np.array(rec[p][2], np.int16)))
    return out, (seed, side, opp, cash[side] - cash[1 - side])


def main(out_dir, n_shards, per=16, seed0=2000000):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed0)
    names, probs = zip(*POOL)
    for sh in range(n_shards):
        path = os.path.join(out_dir, f"shard_{sh:04d}.npz")
        tasks = [(seed0 + sh * per + i, i % 2, rng.choices(names, probs)[0]) for i in range(per)]
        if os.path.exists(path):
            continue
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=int(os.environ.get("SP_WORKERS", "4"))) as ex:
            res = list(ex.map(play, tasks))
        parts = [r for g, _ in res for r in g]
        G, S, y, st = (np.concatenate([p[i] for p in parts]) for i in range(4))
        gid = np.concatenate([np.full(len(p[2]), seed0 + sh * per + k, np.int32) for k, p in enumerate(parts)])
        np.savez(path + ".tmp.npz", G=G, Sc=S, y=y, step=st, gid=gid)
        os.replace(path + ".tmp.npz", path)
        marg = [m for _, (_, _, _, m) in res]
        print(f"shard {sh}: {len(y)} states, {len(res)} games, mean margin {np.mean(marg):+.0f}, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]) if len(a) > 2 else 16, int(a[3]) if len(a) > 3 else 2000000)
