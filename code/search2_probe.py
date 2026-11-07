"""search2_probe -- one game of nn_search2 agent vs v9 with search statistics and timing.

python3 search2_probe.py SEED SIDE 'CONF_DICT' [VWEIGHTS]
"""
import ast
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_v53  # noqa: E402
import nn_search2 as S  # noqa: E402
from fastsim import FastCarbon  # noqa: E402


def main(seed, side, conf, vw=None):
    random.seed(eval_v53.rng_seed(seed, side))
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    ag = S.make_search_agent(conf, vweights=vw)
    opp = eval_v53.load_agent_fn(os.path.join(HERE, "exp", "submission_nn_v9.py"))
    agents = [None, None]; agents[side] = ag; agents[1 - side] = opp
    times = []
    for t in range(300):
        acts = []
        for p in (0, 1):
            o = sim.obs(p)
            t0 = time.time()
            acts.append(agents[p](o, eval_v53.CFG))
            if p == side:
                times.append(time.time() - t0)
        sim.step(acts[0], acts[1])
    import numpy as np
    tt = np.array(times) * 1000
    out = dict(ag.stats)
    out.update(seed=seed, side=side, margin=round(sim.cash[side] - sim.cash[1 - side], 1),
               mean_ms=round(float(tt.mean()), 1), p99_ms=round(float(np.percentile(tt, 99)), 1),
               max_ms=round(float(tt.max()), 1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), ast.literal_eval(a[2]) if len(a) > 2 else {}, a[3] if len(a) > 3 else None)
