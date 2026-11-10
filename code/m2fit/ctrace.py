import os, sys, importlib.util
import numpy as np
sys.path.insert(0, "."); from fastsim import FastCarbon
from m2fit.feats import D
os.environ["M2BOT_CFG"] = sys.argv[2] if len(sys.argv) > 2 else "{}"
s = int(sys.argv[1]); side = sys.argv[3] if len(sys.argv) > 3 else "me"; me = s % 2
def ld(p, nm):
    sp = importlib.util.spec_from_file_location(nm, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld("exp/cnet2_dg5e5s57.py", "b")
q = me if side == "me" else 1 - me
sim = FastCarbon({}).reset(seed=s); hist = {}
for t in range(60):
    o = [sim.obs(p) for p in range(2)]
    a = [ag[0](o[0], {}), ag[1](o[1], {})]
    P = o[0]["players"]; car = np.array(sim.carbon); bs = [int(x) for x in P[q][1].values()]
    for u, w in P[q][2].items():
        if w[2] != "COLLECTOR": continue
        p = int(w[0])
        hist.setdefault(u, []).append(f"{a[q].get(u, 'S')[0]}{int(car[p])}/{int(w[1])}" + ("B" if p in bs else ""))
    sim.step(a[0], a[1])
print(side, "cash", sim.cash[q])
for u, h in list(hist.items())[:5]: print(u[-9:], " ".join(h))
