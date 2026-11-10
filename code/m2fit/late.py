import os, sys, json, importlib.util
import numpy as np
sys.path.insert(0, "."); from fastsim import FastCarbon
from m2fit.feats import D
os.environ["M2BOT_CFG"] = sys.argv[2] if len(sys.argv) > 2 else "{}"
s = int(sys.argv[1]); me = s % 2
def ld(p, nm):
    sp = importlib.util.spec_from_file_location(nm, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld("exp/cnet2_dg5e5s57.py", "b")
sim = FastCarbon({}).reset(seed=s)
for t in range(299):
    o = [sim.obs(p) for p in range(2)]
    a = [ag[0](o[0], {}), ag[1](o[1], {})]
    P = o[0]["players"]; bases = [int(x) for x in P[me][1].values()]
    if t >= 240 and t % 4 == 0 or t >= 285:
        L = []
        for u, w in P[me][2].items():
            if w[2] == "COLLECTOR" and w[1] > 20:
                p = int(w[0]); en = sorted((int(D[int(x[0]), p]), round(x[1])) for x in P[1 - me][2].values() if x[2] == "COLLECTOR")[:2]
                L.append((u[-3:], round(w[1]), int(min(D[p, b] for b in bases)), a[me].get(u, "-"), en))
        print(t, L)
    sim.step(a[0], a[1])
