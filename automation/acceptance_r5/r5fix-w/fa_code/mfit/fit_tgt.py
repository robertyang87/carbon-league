import pickle, sys, json
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "."); from mfit.extract import D, NBR, DIRS
from mfit.fit import tree_json
R = pickle.load(open("mfit/tgt.pkl", "rb"))
names = list(R[0]["cols"].keys())
def build(R):
    X = np.concatenate([np.stack([r["cols"][k] for k in names], 1) for r in R]).astype(float)
    y = np.concatenate([(r["cells"] == r["tgt"]).astype(int) for r in R])
    sid = np.concatenate([np.full(len(r["cells"]), i) for i, r in enumerate(R)])
    g = np.concatenate([np.full(len(r["cells"]), r["g"]) for r in R])
    return X, y, sid, g
X, y, sid, g = build(R)
def move_toward(pos, tg, nb):
    if tg == pos: return "STAY"
    ap = [k for k in range(4) if D[NBR[pos][k], tg] < D[pos, tg]]
    return DIRS[max(ap, key=lambda k: nb[NBR[pos][k]] if False else 0)]
def evaluate(sc, feats_used):
    ok_t = ok_a = ok_s = 0; n = len(R)
    starts = np.r_[0, np.cumsum([len(r["cells"]) for r in R])]
    for i, r in enumerate(R):
        s = sc[starts[i]:starts[i + 1]]; c = r["cells"]
        key = s - 1e-4 * r["cols"]["c_dist"] + 1e-7 * r["cols"]["c_nb"]
        tg = int(c[np.argmax(key)])
        ok_t += tg == r["tgt"]
        a = "STAY" if tg == r["pos"] else None
        ok_s += (a == "STAY") == (r["y"] == "STAY")
        if a is None:
            ap = [DIRS[k] for k in range(4) if D[NBR[r["pos"]][k], tg] < D[r["pos"], tg]]
            ok_a += r["y"] in ap and (len(ap) == 1 and 1 or 0.5)   # expected acc with a fair coin between 2 approach dirs
        else:
            ok_a += r["y"] == "STAY"
    return ok_t / n, ok_s / n, ok_a / n
def run(depth, msl=50, final=False):
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(X[tr], y[tr]); sc[te] = m.predict_proba(X[te])[:, 1]
    res = evaluate(sc, names)
    print(f"depth {depth}: heldout target top1 {res[0]:.3f}  stay-vs-move {res[1]:.3f}  action(exp.) {res[2]:.3f}", flush=True)
    if final:
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(X, y)
        print(export_text(m, feature_names=names, decimals=1)); return m
if __name__ == "__main__":
    print("rows", len(R), "cands", len(y), "stay frac", np.mean([r["y"] == "STAY" for r in R]))
    for d in map(int, sys.argv[1:]): run(d)

def save(depth, path="m_rules_trees.json"):
    m = run(depth, final=True)
    out = json.load(open(path)); out["target"] = tree_json(m, names); json.dump(out, open(path, "w"))
