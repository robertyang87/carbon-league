"""cnet2_search_vetotrace -- per guard veto (v12g vs OPP on SEED/SIDE): context + what happened to the unit in
the next 10 steps.  python3 cnet2_search_vetotrace.py OPP SEED:SIDE [SEED:SIDE ...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v53, cnet2_search as S
from fastsim import FastCarbon
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "weights")
opp = sys.argv[1]
for arg in sys.argv[2:]:
    seed, side = map(int, arg.split(":"))
    ag = S.make_search_agent({'SEARCH': False, 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'GUARD_LOG': True},
                             weights=os.path.join(W, "cnet2_b128_q8.npz"), rweights=os.path.join(W, "cnet2_s64_q8.npz"))
    ob = eval_v53.load_agent_fn(opp); sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    follow = []; nlog = 0
    for t in range(300):
        o = [sim.obs(0), sim.obs(1)]
        a = ag(o[side], eval_v53.CFG); b = ob(o[1 - side], eval_v53.CFG)
        gl = ag.stats.get("guard_log", [])
        for e in gl[nlog:]:
            e["after"] = []; follow.append(e)
        nlog = len(gl)
        cash0 = sim.cash[side]
        sim.step(*((a, b) if side == 0 else (b, a)))
        for e in follow:
            if len(e["after"]) < 10 and e["step"] < sim.step_count:
                w = sim.workers.get(e["uid"])
                e["after"].append("dead" if w is None else round(w.carbon))
    print(json.dumps({"seed": seed, "side": side, "margin": round(sim.cash[side] - sim.cash[1 - side]), "vetoes": follow}), flush=True)
