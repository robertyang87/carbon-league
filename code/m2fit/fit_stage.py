import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from m2fit.stage import rows, RNAMES
from m2fit.feats import CNAMES
R = pickle.load(open(os.path.join(HERE, "ex_m2.pkl"), "rb"))["C"]
fn = os.path.join(HERE, "rows_m2.pkl")
if os.path.exists(fn): Df = pickle.load(open(fn, "rb"))
else: Df = rows(R); pickle.dump(Df, open(fn, "wb"))
g = np.array([r["g"] for r in R]); y = np.array([r["y"] for r in R])
isb = np.array([r["tgt"] in set(r["cells"][r["X"][:, 2] == 1].tolist()) for r in R])
here = np.array([r["tgt"] == r["pos"] for r in R])
def cv(mask, lab, depth, msl=100):
    Xm = Df[RNAMES].values[mask]; yy = lab[mask]; gg = g[mask]; p = np.empty(len(yy), dtype=yy.dtype)
    for tr, te in GroupKFold(5).split(Xm, yy, gg):
        p[te] = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(Xm[tr], yy[tr]).predict(Xm[te])
    maj = pd.Series(yy).value_counts().iloc[0] / len(yy)
    return (p == yy).mean(), maj
onc = (Df.on_base.values == 0) & (Df.car_here.values > 0)
for d in (4, 6, 8):
    print("STAY (on carbon cell) d", d, cv(onc, here, d))
for d in (4, 6, 8):
    print("HOME (not staying) d", d, cv(~here, isb, d))
