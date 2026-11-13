"""mech_check A.py B.py SEED0 N -- per-game mechanism stats for side A: units at t30/t50, cash at t100, final margin."""
import os, random, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
def one(task):
    a, b, seed, side = task
    import eval_v53
    from fastsim import FastCarbon
    random.seed(eval_v53.rng_seed(seed, side))
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    fa = eval_v53.load_agent_fn(os.path.join(HERE, a)); fb = eval_v53.load_agent_fn(os.path.join(HERE, b))
    rec = {}
    for step in range(1, 301):
        oa, ob = sim.obs(side), sim.obs(1 - side)
        if oa["step"] in (30, 50, 100):
            rec[oa["step"]] = (len(oa["players"][side][2]), round(oa["players"][side][0]),
                               len(oa["players"][1 - side][2]), round(oa["players"][1 - side][0]))
        x, y = fa(oa, eval_v53.CFG), fb(ob, eval_v53.CFG)
        sim.step(*((x, y) if side == 0 else (y, x)))
    c = sim.get_state()["cash"]
    return {"seed": seed, "side": side, "rec": rec, "margin": round(c[side] - c[1 - side])}
if __name__ == "__main__":
    a, b, s0, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    tasks = [(a, b, s, sd) for s in range(s0, s0 + n) for sd in (0, 1)]
    with ProcessPoolExecutor(int(os.environ.get("EVO_WORKERS", "2"))) as ex:
        rs = list(ex.map(one, tasks))
    k = len(rs)
    for t in (30, 50, 100):
        print(f"t{t}: A units {sum(r['rec'][t][0] for r in rs)/k:.1f} cash {sum(r['rec'][t][1] for r in rs)/k:.0f} | "
              f"B units {sum(r['rec'][t][2] for r in rs)/k:.1f} cash {sum(r['rec'][t][3] for r in rs)/k:.0f}")
    m = [r["margin"] for r in rs]
    print(f"margin A-B mean {sum(m)/k:.0f} wins {sum(x>0 for x in m)}/{k}")
