import pickle, numpy as np, pandas as pd, sys
sys.path.insert(0, "."); from mfit.extract import D, step as stp
X = pickle.load(open("mfit/ex.pkl", "rb")); P = pd.DataFrame(X["pl"]).sort_values(["g", "uid", "step"])
P["prev"] = P.groupby(["g", "uid"]).y.shift(1)
M = P[(P.y != "STAY") & P.prev.notna()]
print("same as prev move:", (M.y == M.prev).mean(), " prev was move:", (M.prev != "STAY").mean(),
      " same|prev move:", (M[M.prev != "STAY"].y == M[M.prev != "STAY"].prev).mean())
opp = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
mm = M[M.prev != "STAY"]
print("reverse of prev:", (mm.y == mm.prev.map(opp)).mean())
# direction preference when two dirs approach the next stay target: vertical vs horizontal
exec(open("mfit/explore2.py").read().split("m = P[")[0].replace("P = P.sort_values", "P = P.sort_values"))
m = P[(P.y != "STAY") & (P.tpos >= 0)]
both = 0; vert = 0; cnt = {}
for p, y, t in zip(m.pos, m.y, m.tpos):
    p, t = int(p), int(t)
    ap = [d for d in ["UP", "RIGHT", "DOWN", "LEFT"] if D[stp(p, d), t] < D[p, t]]
    if len(ap) == 2 and y in ap:
        both += 1; vert += y in ("UP", "DOWN")
        k = tuple(sorted(ap)); cnt.setdefault(k, {}); cnt[k][y] = cnt[k].get(y, 0) + 1
print("two approach dirs:", both, "vertical chosen", vert / both); print(cnt)
P["prev"] = P.groupby(["g", "uid"]).y.shift(1)
m = P[(P.y != "STAY") & (P.tpos >= 0)]
st = dict(larger_delta=0, prev_same=0, prev_n=0, nb_hi=0, nb_n=0, n=0, car_hi=0)
def dl(p, t):
    r0, c0 = divmod(p, 15); r1, c1 = divmod(t, 15)
    dr = min((r1 - r0) % 15, (r0 - r1) % 15); dc = min((c1 - c0) % 15, (c0 - c1) % 15); return dr, dc
for _, r in m.iterrows():
    p, t, y = int(r.pos), int(r.tpos), r.y
    ap = [d for d in ["UP", "RIGHT", "DOWN", "LEFT"] if D[stp(p, d), t] < D[p, t]]
    if len(ap) != 2 or y not in ap: continue
    st["n"] += 1
    dr, dc = dl(p, t)
    if dr != dc: st["larger_delta"] += (y in ("UP", "DOWN")) == (dr > dc)
    else: st["larger_delta"] += 0.5
    if r.prev in ap: st["prev_n"] += 1; st["prev_same"] += r.prev == y
    a, b = r[ap[0] + ":d_nb"], r[ap[1] + ":d_nb"]
    if a != b: st["nb_n"] += 1; st["nb_hi"] += (r[y + ":d_nb"] == max(a, b))
print(st)
