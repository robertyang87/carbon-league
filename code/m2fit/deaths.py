"""where do m2bot's loaded collectors die? python m2fit/deaths.py OPP SEED N [cfg]"""
import os, sys, json, importlib.util
import numpy as np
sys.path.insert(0, "."); from fastsim import FastCarbon
from m2fit.feats import D, NBR, DIRS
opp, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if len(sys.argv) > 4: os.environ["M2BOT_CFG"] = sys.argv[4]
from collections import Counter
C = Counter(); lost = Counter()
for s in range(s0, s0 + n):
    def ld(p, nm):
        sp = importlib.util.spec_from_file_location(nm + str(s), p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
    me = s % 2; ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld(opp, "b")
    sim = FastCarbon({}).reset(seed=s)
    for t in range(299):
        o = [sim.obs(p) for p in range(2)]
        a = [ag[0](o[0], {}), ag[1](o[1], {})]
        P = o[0]["players"]
        sim.step(a[0], a[1])
        P1 = sim.obs(0)["players"]
        for u, w in P[me][2].items():
            if w[2] != "COLLECTOR" or u in P1[me][2] or w[1] <= 0: continue
            mv = a[me].get(u); pos = int(w[0]); nxt = pos if mv not in DIRS else int(NBR[pos][DIRS.index(mv)])
            thr = [x for x in P[1 - me][2].values() if x[2] == "COLLECTOR" and D[int(x[0]), nxt] <= 1 and x[1] <= w[1]]
            ncand = sum(1 for q in [pos] + [int(x) for x in NBR[pos]] if not any(D[int(x[0]), q] <= 1 for x in P[1 - me][2].values() if x[2] == "COLLECTOR" and x[1] <= w[1]))
            if os.environ.get("VERB"): print(t, u, round(w[1]), "safe options", ncand, "mv", mv, "thr", [(int(x[0]), round(x[1])) for x in thr], "pos", pos)
            k = ("moved" if mv in DIRS else "stayed") + ("/visible-threat" if thr else "/no-threat") + ("/late" if t > 250 else "")
            C[k] += 1; lost[k] += w[1]
print(C); print({k: round(v) for k, v in lost.items()})
