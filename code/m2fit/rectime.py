import os, sys, importlib.util
import numpy as np
sys.path.insert(0, ".")
TS = [10, 25, 50, 75, 100, 150, 200]
def tl(Ps, me):
    seen = {"PLANTER": set(), "COLLECTOR": set()}; out = {}
    alive = {}
    for t, P in enumerate(Ps):
        for u, w in P[me][2].items(): seen[w[2]].add(u)
        if t in TS:
            out[t] = (len(seen["COLLECTOR"]), len(seen["PLANTER"]), sum(w[2] == "COLLECTOR" for w in P[me][2].values()), round(P[me][0]))
    return out
if sys.argv[1] == "replay":
    from m2fit.load import load
    R = [tl(g["P"], g["me"]) for g in load()]
    RO = [tl(g["P"], 1 - g["me"]) for g in load(None) if "claude-opus55" in g["opp"]]
else:
    from fastsim import FastCarbon
    os.environ["M2BOT_CFG"] = sys.argv[3] if len(sys.argv) > 3 else "{}"
    R = []; RO = []
    for s in range(998000, 998000 + int(sys.argv[2])):
        me = s % 2
        def ld(p, nm):
            sp = importlib.util.spec_from_file_location(nm + str(s), p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
        ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld(sys.argv[1], "b")
        sim = FastCarbon({}).reset(seed=s); Ps = []
        for t in range(201):
            o = [sim.obs(p) for p in range(2)]; Ps.append(o[0]["players"])
            sim.step(ag[0](o[0], {}), ag[1](o[1], {}))
        R.append(tl(Ps, me)); RO.append(tl(Ps, 1 - me))
print("step: cumC cumP aliveC cash")
for t in TS:
    print(t, np.mean([r[t] for r in R], 0).round(1), " opp", np.mean([r[t] for r in RO], 0).round(1))
