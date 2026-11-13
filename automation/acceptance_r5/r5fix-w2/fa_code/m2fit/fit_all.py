"""fit all whole-game M2 decision trees, report held-out-by-game accuracy, dump code/m2_bot_trees.json
   python m2fit/fit_all.py [dump]"""
import os, sys, pickle, json
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from m2fit.stage import rows, RNAMES
from m2fit.feats import CNAMES, D, NBR, DIRS
from mfit.fit import tree_json
TAG = os.environ.get("EXTAG", "m2")
EX = pickle.load(open(os.path.join(HERE, f"ex_{TAG}.pkl"), "rb"))
R = EX["C"]
fn = os.path.join(HERE, f"rows_{TAG}.pkl")
if os.path.exists(fn): Df = pickle.load(open(fn, "rb"))
else: Df = rows(R); pickle.dump(Df, open(fn, "wb"))
g = np.array([r["g"] for r in R]); y = np.array([r["y"] for r in R]); pos = np.array([r["pos"] for r in R])
isb = np.array([bool(r["X"][r["cells"] == r["tgt"], 2][0]) for r in R])
here = np.array([r["tgt"] == r["pos"] for r in R])
onc = (Df.on_base.values == 0) & (Df.car_here.values > 0)
XR = Df[RNAMES].values
DS, DH, DT = [int(x) for x in os.environ.get("DEPTHS", "8,6,8").split(",")]
TF = [f for f in CNAMES if f not in ("c_isbase", "c_here")]
TI = [CNAMES.index(f) for f in TF]

def tmask(r):
    return (r["X"][:, 2] == 0) & (r["X"][:, 3] == 0)

def tbuild(idx):
    Xs, ys, ls = [], [], []
    for i in idx:
        r = R[i]; m = tmask(r); Xs.append(r["X"][m][:, TI]); ys.append((r["cells"][m] == r["tgt"]).astype(np.int8))
    return np.concatenate(Xs), np.concatenate(ys)

def tpick(m, r, excl=None):
    msk = tmask(r)
    if not msk.any():
        return None
    X = r["X"][msk]; c = r["cells"][msk]
    p = m.predict_proba(X[:, TI])[:, 1]
    key = p + 1e-4 * X[:, CNAMES.index("c_rate")] / 100 - 1e-7 * X[:, 1]
    return int(c[np.argmax(key)])

def move_ok(p, tg, yy):
    if tg == p: return float(yy == "STAY")
    ap = [DIRS[k] for k in range(4) if D[NBR[p][k], tg] < D[p, tg]]
    return (yy in ap) / len(ap)

def fit_stages(tr):
    ms = DecisionTreeClassifier(max_depth=DS, min_samples_leaf=100, random_state=0).fit(XR[tr & onc], here[tr & onc])
    nh = tr & ~here
    mh = DecisionTreeClassifier(max_depth=DH, min_samples_leaf=100, random_state=0).fit(XR[nh], isb[nh])
    ti = np.nonzero(tr & ~here & ~isb)[0]
    Xt, yt = tbuild(ti)
    mt = DecisionTreeClassifier(max_depth=DT, min_samples_leaf=200, random_state=0).fit(Xt, yt)
    return ms, mh, mt

def pipeline(ms, mh, mt, te_idx):
    st = ms.predict(XR[te_idx]); ho = mh.predict(XR[te_idx]); acc = []; tacc = []
    for k, i in enumerate(te_idx):
        r = R[i]
        if onc[i] and st[k]:
            tg = r["pos"]
        elif ho[k]:
            bs = r["cells"][r["X"][:, 2] == 1]; tg = int(bs[np.argmin(D[r["pos"], bs])])
        else:
            tg = tpick(mt, r)
            if tg is None: tg = r["pos"]
        acc.append(move_ok(r["pos"], tg, r["y"])); tacc.append(tg == r["tgt"])
    return np.array(acc), np.array(tacc)

def cv():
    acc = np.zeros(len(R)); tacc = np.zeros(len(R)); sacc = []; hacc = []; tsub = []
    for tr_i, te_i in GroupKFold(5).split(XR, y, g):
        tr = np.zeros(len(R), bool); tr[tr_i] = True
        ms, mh, mt = fit_stages(tr)
        a, t = pipeline(ms, mh, mt, te_i); acc[te_i] = a; tacc[te_i] = t
        te = ~tr
        sacc.append(((ms.predict(XR[te & onc]) == here[te & onc]).mean(), (te & onc).sum()))
        hacc.append(((mh.predict(XR[te & ~here]) == isb[te & ~here]).mean(), (te & ~here).sum()))
        ti = np.nonzero(te & ~here & ~isb)[0]
        tsub.append((np.mean([tpick(mt, R[i]) == R[i]["tgt"] for i in ti]), len(ti)))
    w = lambda L: sum(a * n for a, n in L) / sum(n for _, n in L)
    maj = lambda v: max(v.mean(), 1 - v.mean())
    print(f"collector STAY-here (on carbon cell) heldout {w(sacc):.3f}  majority {maj(here[onc]):.3f}")
    print(f"collector HOME (when not staying)   heldout {w(hacc):.3f}  majority {maj(isb[~here]):.3f}")
    print(f"collector TARGET top1 (non-base, moving) heldout {w(tsub):.3f}")
    print(f"collector pipeline: next-stop {tacc.mean():.3f}  action (exp.) {acc.mean():.3f}  majority action {pd.Series(y).value_counts().iloc[0] / len(y):.3f}")

def dump(path):
    out = {}
    ms, mh, mt = fit_stages(np.ones(len(R), bool))
    out["c_stay"] = tree_json(ms, RNAMES); out["c_home"] = tree_json(mh, RNAMES); out["c_tgt"] = tree_json(mt, TF)
    for nm, m, f in (("STAY", ms, RNAMES), ("HOME", mh, RNAMES)):
        print("=====", nm); print(export_text(m, feature_names=f, decimals=1, max_depth=5))
    json.dump(out, open(path, "w"))

if __name__ == "__main__":
    if "cv" in sys.argv: cv()
    if "dump" in sys.argv: dump(os.path.join(HERE, "..", "m2_bot_ctrees.json"))
