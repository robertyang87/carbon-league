import os, sys, importlib.util
import numpy as np
sys.path.insert(0, "."); from fastsim import FastCarbon
from m2fit.feats import NBR, DIRS
s = int(sys.argv[1]); me = int(sys.argv[2])
def ld(p, nm):
    sp = importlib.util.spec_from_file_location(nm, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld("exp/cnet2_dg5e5s57.py", "b")
sim = FastCarbon({}).reset(seed=s); dep = 0.0; bad = 0
for t in range(299):
    o = [sim.obs(p) for p in range(2)]
    a = [ag[0](o[0], {}), ag[1](o[1], {})]
    P = o[me]["players"]; W = P[me][2]
    sim.step(a[0], a[1]); P1 = sim.obs(me)["players"]; W1 = P1[me][2]; bs = {int(x) for x in P1[me][1].values()}
    d = 0.0
    for u, w in W.items():
        if u in W1:
            mv = a[me].get(u); exp = int(w[0]) if mv not in DIRS else int(NBR[int(w[0])][DIRS.index(mv)])
            bad += exp != int(W1[u][0])
            if w[2] == "COLLECTOR" and int(W1[u][0]) in bs and int(W1[u][0]) != int(w[0]): d += w[1]
    dep += d
    if t < 50:
        print(t, "cash", round(P[me][0]), "ncol", sum(w[2] == "COLLECTOR" for w in W.values()), "npl", sum(w[2] == "PLANTER" for w in W.values()),
              "cargo", round(sum(w[1] for w in W.values())), "dep", round(d), {k[-9:]: v for k, v in a[me].items()})
print("final", sim.cash, "deposits", round(dep), "position mismatches", bad)
