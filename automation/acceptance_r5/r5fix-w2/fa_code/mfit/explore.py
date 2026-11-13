import pickle, pandas as pd, numpy as np
X = pickle.load(open("mfit/ex.pkl", "rb")); B = pd.DataFrame(X["base"]); P = pd.DataFrame(X["pl"])
print(B.y.value_counts()); print(P.y.value_counts())
print(P.groupby("occ_here").y.value_counts().unstack().fillna(0))
s = P[P.y == "STAY"]
print("stay on free: nb_here quantiles", s[s.free_here == 1].nb_here.quantile([.05, .25, .5, .75]).values)
print("stay on free cash>=price frac", (s[s.free_here == 1].afford).mean())
print("move from free: nb_here q", P[(P.y != "STAY") & (P.free_here == 1)].nb_here.quantile([.05, .25, .5, .75, .95]).values)
# first recruits
fp = B[B.y == "P"].groupby("g").step.min(); fc = B[B.y == "C"].groupby("g").step.min()
print("first P step", fp.describe().round(1).to_dict()); print("first C", fc.describe().round(1).to_dict())
print(B[B.y != "N"].groupby([pd.cut(B.step, [-1, 10, 25, 50, 75, 100]), "y"]).size().unstack() / B.g.nunique())
print("chosen dir: is it the max d_nb among dirs?")
m = P[P.y != "STAY"]
for k in ["d_nb", "d_best1", "d_best2"]:
    arr = np.stack([m[d + ":" + k].values for d in ["UP", "RIGHT", "DOWN", "LEFT"]], 1)
    ch = m.y.map({"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}).values
    print(k, (arr[np.arange(len(m)), ch] == arr.max(1)).mean())
print(P.groupby("name").size(), B.groupby("name").size())
