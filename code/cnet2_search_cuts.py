"""cnet2_search_cuts -- quiet-machine measurement: sims per searched step that each budget cut completes within
wall deadlines (no-new-round START, abort ABORT).  python3 cnet2_search_cuts.py START ABORT SEED"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, eval_v53, cnet2_search as S
from fastsim import FastCarbon
st_, ab_, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
BASE = {'MODE': 'mcts', 'CLOCK': 'wall', 'BUDGET': st_, 'HARD': ab_ + 0.01, 'START_DL': st_, 'ABORT_DL': ab_, 'ROOT_S': 4, 'MINC': 20.0,
        'SPLIT': True, 'PREY_MIN': 50.0, 'MAX_KEY': 2, 'MAX_BR': 9, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'MIX': True, 'GUARD': True,
        'RW': 'cnet2_s64_q8.npz'}
CUTS = {"base": {}, "a_rs2": {"ROOT_S": 2}, "b_key1": {"SPLIT": False, "MAX_KEY": 1, "MAX_BR": 5}, "c_h2": {"H": 2}}
for name, c in CUTS.items():
    ag = S.make_search_agent(dict(BASE, **c)); ob = eval_v53.load_agent_fn("exp/submission_cnet2_v13.py")
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed); W = []; SM = []; prev = (0, 0)
    for t in range(300):
        o = sim.obs(0); ts = time.time(); a = ag(o, eval_v53.CFG); W.append((time.time() - ts) * 1000)
        cur = (ag.stats.get("sims", 0), ag.stats["searched"])
        if cur[1] > prev[1]:
            SM.append(cur[0] - prev[0])
        prev = cur
        sim.step(a, ob(sim.obs(1), eval_v53.CFG))
    W = np.array(W); SM = np.array(SM)
    print(json.dumps({"cut": name, "sims_mean": round(SM.mean(), 1), "sims_p10": float(np.percentile(SM, 10)), "sims_p50": float(np.percentile(SM, 50)),
                      "sims_p90": float(np.percentile(SM, 90)), "zero_sim_frac": round(float((SM == 0).mean()), 3), "wall_max": round(W.max()),
                      "wall_p99": round(np.percentile(W, 99)), "aborted": ag.stats.get("aborted", 0), "searched": len(SM)}), flush=True)
