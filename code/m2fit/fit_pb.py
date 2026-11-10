"""planter next-stay target tree + base recruit/ptype trees over the whole game; held-out by game; dump into m2_bot_ptrees.json"""
import os, sys, pickle, json
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from mfit.fit import tree_json
from m2fit.feats import D, NBR, DIRS
TAG = os.environ.get("EXTAG", "m2")
EX = pickle.load(open(os.path.join(HERE, f"ex_{TAG}.pkl"), "rb"))
PL = EX["PL"]; B = pd.DataFrame(EX["B"])
PN = list(PL[0]["cols"].keys())
BF = ["step", "left", "cash", "price", "own_tr", "en_tr", "n_col", "n_pl", "en_col", "en_pl", "n_wk", "rec_pl", "rec_col", "bidx", "base_occ", "cargo", "ecash"]
gk = GroupKFold(5)
maj = lambda v: pd.Series(v).value_counts().iloc[0] / len(v)

def pbuild(idx):
    X = np.concatenate([np.stack([PL[i]["cols"][k] for k in PN], 1) for i in idx]).astype(np.float32)
    y = np.concatenate([(PL[i]["cells"] == PL[i]["tgt"]).astype(np.int8) for i in idx]); return X, y

def ppick(m, r):
    X = np.stack([r["cols"][k] for k in PN], 1)
    key = m.predict_proba(X)[:, 1] - 1e-4 * r["cols"]["c_dist"] + 1e-7 * r["cols"]["c_nb"]
    return int(r["cells"][np.argmax(key)])

def planter(depth=8, dump=None):
    g = np.array([r["g"] for r in PL]); ok_t = []; ok_a = []
    for tr, te in gk.split(g, g, g):
        X, y = pbuild(tr); m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=0).fit(X, y)
        for i in te:
            r = PL[i]; tg = ppick(m, r); ok_t.append(tg == r["tgt"]); p = r["pos"]
            if tg == p: ok_a.append(float(r["y"] == "STAY"))
            else:
                ap = [DIRS[k] for k in range(4) if D[NBR[p][k], tg] < D[p, tg]]; ok_a.append((r["y"] in ap) / len(ap))
    ys = np.array([r["y"] for r in PL])
    print(f"PLANTER target d{depth}: next-stay top1 {np.mean(ok_t):.3f}  action(exp.) {np.mean(ok_a):.3f}  majority action {maj(ys):.3f} n={len(PL)}")
    if dump is not None:
        X, y = pbuild(range(len(PL))); m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=0).fit(X, y)
        dump["p_tgt"] = tree_json(m, PN); print(export_text(m, feature_names=PN, decimals=1, max_depth=4))

def base(depth=6, dump=None):
    can = B[(B.cash >= 30) & (B.n_wk < 10)]
    yr = (can.y != "N").astype(int).values; rec = B[B.y != "N"]; yt = np.asarray(rec.y.values, dtype=object)
    for nm, D_, y in (("recruit", can, yr), ("ptype", rec, yt)):
        p = np.empty(len(y), dtype=object)
        for tr, te in gk.split(D_, y, D_.g.values):
            p[te] = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=30, random_state=0).fit(D_[BF].values[tr], y[tr]).predict(D_[BF].values[te])
        print(f"BASE {nm} d{depth}: heldout {(p == y).mean():.3f}  majority {maj(y):.3f} n={len(y)}")
        if dump is not None:
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=30, random_state=0).fit(D_[BF].values, y)
            dump["b_" + nm] = tree_json(m, BF); print(export_text(m, feature_names=BF, decimals=1, max_depth=4))

if __name__ == "__main__":
    out = {} if "dump" in sys.argv else None
    base(int(os.environ.get("BD", 6)), out); planter(int(os.environ.get("PD", 8)), out)
    if out is not None:
        json.dump(out, open(os.path.join(HERE, "..", "m2_bot_ptrees.json"), "w"))
