"""early (t<100) and total unit losses per side; replay (M2 vs our nets) or sim (m2bot vs OPP)"""
import os, sys, importlib.util
import numpy as np
sys.path.insert(0, ".")
def ev(Ps, me):
    out = {}
    for side, q in (("me", me), ("op", 1 - me)):
        c = {"colD100": 0, "colD": 0, "plD100": 0, "plD": 0, "cargoD100": 0.0}
        for t in range(len(Ps) - 1):
            W, W1 = Ps[t][q][2], Ps[t + 1][q][2]
            for u, w in W.items():
                if u not in W1:
                    k = "col" if w[2] == "COLLECTOR" else "pl"
                    c[k + "D"] += 1
                    if t < 100: c[k + "D100"] += 1; c["cargoD100"] += w[1] if w[2] == "COLLECTOR" else 0
        out[side] = c
    return out
if sys.argv[1] == "replay":
    from m2fit.load import load
    R = [ev(g["P"], g["me"]) for g in load(None) if "claude-opus55" in g["opp"]]
else:
    from fastsim import FastCarbon
    os.environ["M2BOT_CFG"] = sys.argv[3] if len(sys.argv) > 3 else "{}"
    R = []
    for s in range(998000, 998000 + int(sys.argv[2])):
        me = s % 2
        def ld(p, nm):
            sp = importlib.util.spec_from_file_location(nm + str(s), p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
        ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld(sys.argv[1], "b")
        sim = FastCarbon({}).reset(seed=s); Ps = []
        for t in range(300):
            o = [sim.obs(p) for p in range(2)]; Ps.append(o[0]["players"])
            if t < 299: sim.step(ag[0](o[0], {}), ag[1](o[1], {}))
        R.append(ev(Ps, me))
for side in ("me", "op"):
    print(side, {k: round(float(np.mean([r[side][k] for r in R])), 1) for k in R[0][side]})
