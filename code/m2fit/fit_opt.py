"""option trees (stay + 4 moves, argmax of a binary tree over option rows); held-out by game; dump m2_bot_otrees.json"""
import os, sys, pickle, json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from m2fit.policy import ONAMES_C, ONAMES_P
from mfit.fit import tree_json
TAG = os.environ.get("EXTAG", "m2")
O = pickle.load(open(os.path.join(HERE, f"opt_{TAG}.pkl"), "rb"))

def run(R, names, depth, msl=100, dump=False):
    X = np.concatenate([r["X"] for r in R]); y = np.concatenate([(np.arange(5) == r["y"]).astype(int) for r in R])
    g = np.repeat([r["g"] for r in R], 5); lab = np.array([r["y"] for r in R])
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        sc[te] = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    S = sc.reshape(-1, 5)
    ap = X[:, names.index("o_appr")].reshape(-1, 5); st = X[:, 0].reshape(-1, 5)
    tie = -1e-3 * ap - 1e-6 * np.arange(5)
    acc = (np.argmax(S + tie, 1) == lab).mean()
    base = (np.argmax(-ap - 0.5 * st * (1 - X[:, names.index("o_tgt_here")].reshape(-1, 5)) * 0 + 10 * st * X[:, names.index("o_tgt_here")].reshape(-1, 5) - 1e-6 * np.arange(5), 1) == lab).mean()
    maj = np.bincount(lab).max() / len(lab)
    print(f"  depth {depth}: heldout action {acc:.3f}  target-following {base:.3f}  majority {maj:.3f}  n={len(lab)}", flush=True)
    if dump:
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(X, y)
        print(export_text(m, feature_names=names, decimals=1, max_depth=4))
        return tree_json(m, names)

if __name__ == "__main__":
    dc, dp = [int(x) for x in os.environ.get("ODEPTHS", "8,8").split(",")]
    out = {}
    print("COLLECTOR options"); out["c_opt"] = run(O["C"], ONAMES_C, dc, dump="dump" in sys.argv)
    print("PLANTER options"); out["p_opt"] = run(O["PL"], ONAMES_P, dp, dump="dump" in sys.argv)
    if "dump" in sys.argv:
        json.dump(out, open(os.path.join(HERE, "..", "m2_bot_otrees.json"), "w"))
