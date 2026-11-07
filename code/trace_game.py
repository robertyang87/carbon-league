"""trace_game -- step trace of one FastCarbon game: python3 trace_game.py A.py B.py SEED A_SIDE [T_END] [WATCH_SIDE]"""
import os, random, sys
import eval_v53 as E
from fastsim import FastCarbon
a, b, seed, side = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
T = int(sys.argv[5]) if len(sys.argv) > 5 else 80
W = int(sys.argv[6]) if len(sys.argv) > 6 else side
random.seed(E.rng_seed(seed, side)); sim = FastCarbon(E.CFG).reset(seed=seed)
ag = [None, None]; ag[side] = E.load_agent_fn(os.path.join(E.HERE, a)); ag[1 - side] = E.load_agent_fn(os.path.join(E.HERE, b))
xy = lambda c: (c // 15, c % 15)
for step in range(1, T + 1):
    pre = sim.get_state()
    acts = [ag[p](sim.obs(p), E.CFG) for p in range(2)]
    sim.step(acts[0], acts[1]); post = sim.get_state()
    if step == 1:
        print("bases", pre["bases"])
    died = [(wid[-6:], pre['workers'][wid]['type'][0], pre['workers'][wid])
            for wid in pre['workers'] if wid not in post['workers'] and pre['workers'][wid]['player'] == W]
    mine = [(w['type'][0], round(w['carbon'])) for w in post['workers'].values() if w['player'] == W]
    print(f"t{step:3d} cash {post['cash'][W]:7.1f}/{post['cash'][1-W]:7.1f} units {len(mine)} acts {acts[W]} died {died}")
