"""cnet2_search_idtest -- online opponent modelling diagnostics (search off): opponent next-move
log-likelihood / accuracy of (a) fixed style 1, (b) style mixture with online weights, (c) mixture + online
temperature, plus the mixture-weight trajectory, vs each style stand-in, v12, v13, v5.
python3 cnet2_search_idtest.py SEED0 N [OPPKEYS]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v53, cnet2_search as S
from fastsim import FastCarbon
OPP = {0: "exp/cnet2_style_t0.py", 1: "exp/submission_cnet2_v12.py", 2: "exp/cnet2_style_t2.py",
       3: "exp/cnet2_style_t3.py", 4: "exp/cnet2_style_t4.py", "v13": "exp/submission_cnet2_v13.py",
       "v5": "exp/opp_1cz_v5.py"}
s0, n = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) > 3:
    OPP = {k: v for k, v in OPP.items() if str(k) in sys.argv[3].split(",")}
res = {}
for k, path in OPP.items():
    for seed in range(s0, s0 + n):
        side = seed % 2
        ag = S.make_search_agent({"SEARCH": False, "MIX": True, "ONLINE_T": True})
        ob = eval_v53.load_agent_fn(path)
        sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
        for t in range(300):
            o = [sim.obs(0), sim.obs(1)]
            a = ag(o[side], eval_v53.CFG); b = ob(o[1 - side], eval_v53.CFG)
            sim.step(*((a, b) if side == 0 else (b, a)))
        d = ag.stats.get("pred", {})
        res.setdefault(str(k), []).append(d)
        print(json.dumps({"opp": str(k), "seed": seed, "id": ag.stats.get("id"), "pred": d, "err": ag.stats["errors"],
                          "last_error": ag.stats.get("last_error", "")}), flush=True)
print("SUMMARY (per opponent unit: mean log-lik, accuracy)")
for k, rows in res.items():
    nu = sum(r["n_units"] for r in rows)
    print(k, "units", nu, {m: (round(sum(r[m + "_ll"] for r in rows) / nu, 3), round(sum(r[m + "_acc"] for r in rows) / nu, 3))
                           for m in ("a", "b", "c")}, flush=True)
