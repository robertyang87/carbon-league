"""fit collector next-stop tree (candidate formulation); held-out by game"""
import os, sys, pickle, json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from m2fit.feats import CNAMES, D, NBR, DIRS
from mfit.fit import tree_json

def build(R):
    X = np.concatenate([r["X"] for r in R]); y = np.concatenate([(r["cells"] == r["tgt"]).astype(np.int8) for r in R])
    g = np.concatenate([np.full(len(r["cells"]), r["g"]) for r in R]); st = np.r_[0, np.cumsum([len(r["cells"]) for r in R])]
    return X, y, g, st

def pick(sc, r):
    X = r["X"]; key = sc + 1e-4 * X[:, 0] / 100 - 1e-6 * X[:, 1]
    return int(r["cells"][np.argmax(key)])

def evaluate(sc, R, st):
    ok_t = ok_s = ok_a = ok_h = 0
    for i, r in enumerate(R):
        tg = pick(sc[st[i]:st[i + 1]], r); pos = r["pos"]
        ok_t += tg == r["tgt"]
        ps = tg == pos; ok_s += ps == (r["y"] == "STAY")
        isb = lambda c: c in set(r["cells"][r["X"][:, 2] == 1].tolist())
        ok_h += isb(tg) == isb(r["tgt"])
        if ps:
            ok_a += r["y"] == "STAY"
        else:
            ap = [DIRS[k] for k in range(4) if D[NBR[pos][k], tg] < D[pos, tg]]
            ok_a += (r["y"] in ap) / len(ap)
    n = len(R); return ok_t / n, ok_s / n, ok_h / n, ok_a / n

def run(R, depth, msl=200, feats=None):
    X, y, g, st = build(R)
    fi = [CNAMES.index(f) for f in (feats or CNAMES)]
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(X[tr][:, fi], y[tr]); sc[te] = m.predict_proba(X[te][:, fi])[:, 1]
    res = evaluate(sc, R, st)
    print(f"depth {depth} msl {msl}: target {res[0]:.3f} stay {res[1]:.3f} home {res[2]:.3f} action {res[3]:.3f}", flush=True)
    return res

def final(R, depth, msl=200, feats=None):
    X, y, g, st = build(R); feats = feats or CNAMES; fi = [CNAMES.index(f) for f in feats]
    m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(X[:, fi], y)
    return m, tree_json(m, feats)

if __name__ == "__main__":
    R = pickle.load(open(os.path.join(HERE, "ex_m2.pkl"), "rb"))["C"]
    ys = np.array([r["y"] for r in R])
    from collections import Counter; c = Counter(ys); print("n", len(R), "majority action", max(c.values()) / len(R), c)
    for d in map(int, sys.argv[1:]):
        run(R, d)
