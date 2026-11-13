import pickle, pandas as pd, numpy as np, sys
sys.path.insert(0, "."); from mfit.extract import D, NBR
X = pickle.load(open("mfit/ex.pkl", "rb")); P = pd.DataFrame(X["pl"])
P = P.sort_values(["g", "uid", "step"])
# next stay cell per unit
tgt = []
for (g, u), grp in P.groupby(["g", "uid"]):
    nxt = None; out = []
    for i in reversed(range(len(grp))):
        r = grp.iloc[i]
        if r.y == "STAY": nxt = (r.pos, r.step, r.occ_here)
        out.append(nxt)
    tgt += list(zip(grp.index, reversed(out)))
T = dict(tgt)
P["tpos"] = [T[i][0] if T[i] else -1 for i in P.index]
P["tocc"] = [T[i][2] if T[i] else -1 for i in P.index]
P["tstep"] = [T[i][1] if T[i] else -1 for i in P.index]
m = P[(P.y != "STAY") & (P.tpos >= 0)].copy()
from mfit.extract import step as stp
m["dd"] = [D[stp(int(p), y), int(t)] - D[int(p), int(t)] for p, y, t in zip(m.pos, m.y, m.tpos)]
print("moves with later stay:", len(m), "of", (P.y != "STAY").sum(), "  dist change to next stay cell:", m.dd.value_counts().to_dict())
print("target occ of moves (0 free 1 own 2 enemy tree):", m.tocc.value_counts().to_dict())
m["dist"] = [D[int(p), int(t)] for p, t in zip(m.pos, m.tpos)]
print("dist to target at decision", m.dist.describe().round(2).to_dict())
