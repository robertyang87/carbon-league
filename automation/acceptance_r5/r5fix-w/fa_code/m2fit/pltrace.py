import os, sys, importlib.util
import numpy as np
sys.path.insert(0, "."); from fastsim import FastCarbon
from m2fit.feats import D
os.environ["M2BOT_CFG"] = sys.argv[2] if len(sys.argv) > 2 else "{}"
s = int(sys.argv[1]); me = s % 2
def ld(p, nm):
    sp = importlib.util.spec_from_file_location(nm, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld("exp/cnet2_dg5e5s57.py", "b")
sim = FastCarbon({}).reset(seed=s); hist = {}
for t in range(70):
    o = [sim.obs(p) for p in range(2)]
    a = [ag[0](o[0], {}), ag[1](o[1], {})]
    P = o[0]["players"]
    nb = np.array(sim.carbon); 
    for u, w in P[me][2].items():
        if w[2] != "PLANTER": continue
        p = int(w[0]); ep = [int(x[0]) for x in P[1 - me][2].values() if x[2] == "PLANTER"]
        occ = "T" if any(int(q) == p for q, _ in P[me][3].values()) else ("E" if any(int(q) == p for q, _ in P[1 - me][3].values()) else ".")
        hist.setdefault(u, []).append(f"{t}:{a[me].get(u, 'S')[0]}{occ}{min([int(D[p, e]) for e in ep] or [9])}")
    sim.step(a[0], a[1])
    P1 = sim.obs(0)["players"]
    for u in P[me][2]:
        if u not in P1[me][2] and u in hist: hist[u].append("DEAD")
print("cash", sim.cash, "own trees", len(sim.obs(0)["players"][me][3]))
for u, h in hist.items(): print(u[-9:], " ".join(h))
