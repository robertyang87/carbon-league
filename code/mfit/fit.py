"""fit shallow decision trees to M's opening decisions; held-out by game (GroupKFold 5)"""
import pickle, sys, json
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
X = pickle.load(open("mfit/ex.pkl", "rb")); B = pd.DataFrame(X["base"]); P = pd.DataFrame(X["pl"])
DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
P = P.sort_values(["g", "uid", "step"]).reset_index(drop=True)
P["prev"] = P.groupby(["g", "uid"]).y.shift(1).fillna("NONE")
P["prev_moved"] = (~P.prev.isin(["STAY", "NONE"])).astype(int)
for d in DIRS:
    P[d + ":d_same"] = (P.prev == d).astype(int); P[d + ":d_rev"] = (P.prev == OPP[d]).astype(int)
    P[d + ":b_d_same"] = P[d + ":d_same"]; P[d + ":b_d_rev"] = P[d + ":d_rev"]
gkf = GroupKFold(5)
BF = ["step", "cash", "price", "own_tr", "en_tr", "n_col", "n_pl", "en_col", "en_pl", "n_wk", "rec_pl", "rec_col", "bidx", "base_occ", "cargo"]
CTX = ["prev_moved", "step", "cash", "price", "own_tr", "en_tr", "n_col", "n_pl", "n_wk", "rec_pl", "rec_col", "afford"]
PF = ["nb_here", "free_here", "etree_here", "otree_here", "base_here", "d_etree", "d_epl", "d_ecol", "d_base", "best1", "best2", "best3",
      "bestE1", "bestE2", "nb_minus_best1", "n_opl_adj"]
DF = ["d_nb", "d_nb_rel", "d_free", "d_etree", "d_otree", "d_base", "d_epl1", "d_epl0", "d_ecol1", "d_dbase", "d_best1", "d_best2",
      "d_bestE1", "d_best2_rel", "d_detree", "d_opl", "d_same", "d_rev"]

def cv(Xm, y, g, depth, msl=20):
    pred = np.empty(len(y), dtype=object)
    for tr, te in gkf.split(Xm, y, g):
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(Xm[tr], y[tr]); pred[te] = m.predict(Xm[te])
    return (pred == y).mean()

def maj(y): return pd.Series(y).value_counts().iloc[0] / len(y)

def dir_long(P, pref):
    rows = []
    for k, d in enumerate(DIRS):
        Z = pd.DataFrame({f: P[d + ":" + pref + f].values for f in DF})
        for f in ["nb_here", "free_here", "etree_here", "d_etree", "best2", "best3", "bestE2", "step", "afford", "cash", "price"]:
            Z[f] = P[(pref + f) if (pref + f) in P else f].values
        Z["sid"] = np.arange(len(P)); Z["dir"] = k; Z["y"] = (P.y.values == d).astype(int); Z["g"] = P.g.values
        rows.append(Z)
    return pd.concat(rows, ignore_index=True)

def dir_cv(P, pref, depth, msl=20, ret=False):
    L = dir_long(P, pref); feats = [c for c in L.columns if c not in ("sid", "dir", "y", "g")]
    sc = np.zeros(len(L))
    for tr, te in gkf.split(L, L.y, L.g):
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(L[feats].values[tr], L.y.values[tr])
        sc[te] = m.predict_proba(L[feats].values[te])[:, 1]
    S = sc.reshape(4, len(P)).T + 1e-6 * np.random.RandomState(0).rand(len(P), 4)
    acc = (np.array(DIRS)[S.argmax(1)] == P.y.values).mean()
    if ret:
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=0).fit(L[feats].values, L.y.values)
        return acc, m, feats
    return acc

if __name__ == "__main__":
    yb = B.y.values; gb = B.g.values
    print(f"BASE  n={len(B)} majority {maj(yb):.3f}")
    for d in (3, 4, 5, 6):
        print("  depth", d, f"{cv(B[BF].values, yb, gb, d):.3f}")
    ys = (P.y.values == "STAY").astype(int); gp = P.g.values
    print(f"STAY  n={len(P)} majority {maj(ys):.3f}")
    for pref in ("", "b_"):
        F_ = CTX + [pref + f for f in PF]
        for d in (3, 4, 5, 6):
            print("  ", pref or "full", "depth", d, f"{cv(P[F_].values, ys, gp, d):.3f}")
    M = P[P.y != "STAY"].reset_index(drop=True)
    print(f"DIR   n={len(M)} majority {maj(M.y.values):.3f}")
    for pref in ("", "b_"):
        for d in (3, 4, 5, 6):
            print("  ", pref or "full", "depth", d, f"{dir_cv(M, pref, d):.3f}")

def final(db=5, ds=5, dd=6, pref=""):
    """fit on all games, print rules, save mfit/trees.pkl for m_rules.py"""
    mb = DecisionTreeClassifier(max_depth=db, min_samples_leaf=20, random_state=0).fit(B[BF].values, B.y.values)
    F_ = CTX + [pref + f for f in PF]
    ms = DecisionTreeClassifier(max_depth=ds, min_samples_leaf=20, random_state=0).fit(P[F_].values, (P.y.values == "STAY").astype(int))
    M = P[P.y != "STAY"].reset_index(drop=True)
    acc, md, dfeats = dir_cv(M, pref, dd, ret=True)
    for nm, m, f in (("BASE", mb, BF), ("STAY", ms, F_), ("DIR", md, dfeats)):
        print("=====", nm); print(export_text(m, feature_names=list(f), decimals=1, show_weights=False))
    pickle.dump(dict(base=(mb, BF), stay=(ms, F_), dir=(md, dfeats), pref=pref), open("mfit/trees.pkl", "wb"))

def tree_json(m, feats):
    t = m.tree_
    return dict(feats=list(feats), classes=[str(c) for c in m.classes_], left=t.children_left.tolist(), right=t.children_right.tolist(),
                feat=t.feature.tolist(), thr=t.threshold.tolist(), val=(t.value[:, 0, :] / t.value[:, 0, :].sum(1, keepdims=True)).tolist())

def base2(depth=4, show=True):
    """recruit? (rows where a recruit is possible) and P-vs-C (rows with a recruit)"""
    can = B[(B.cash >= 30) & (B.n_wk < 10)]
    yr = (can.y != "N").astype(int).values
    rec = B[B.y != "N"]; yt = rec.y.values
    out = {}
    for nm, D_, y in (("recruit", can, yr), ("ptype", rec, yt)):
        a = cv(D_[BF].values, y, D_.g.values, depth)
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=0).fit(D_[BF].values, y)
        if show:
            print(f"== {nm} n={len(y)} majority {maj(y):.3f} heldout d{depth} {a:.3f}"); print(export_text(m, feature_names=BF, decimals=1))
        out[nm] = tree_json(m, BF)
    return out

def dump(ds=5, dd=6, db=4, pref="", path="m_rules_trees.json"):
    out = base2(db)
    F_ = CTX + [pref + f for f in PF]
    ys = (P.y.values == "STAY").astype(int)
    print(f"== stay held-out d{ds} {cv(P[F_].values, ys, P.g.values, ds):.3f} (maj {maj(ys):.3f})")
    ms = DecisionTreeClassifier(max_depth=ds, min_samples_leaf=20, random_state=0).fit(P[F_].values, ys)
    print(export_text(ms, feature_names=F_, decimals=1))
    M = P[P.y != "STAY"].reset_index(drop=True)
    acc, md, dfeats = dir_cv(M, pref, dd, ret=True)
    print(f"== dir held-out d{dd} {acc:.3f} (maj {maj(M.y.values):.3f})"); print(export_text(md, feature_names=dfeats, decimals=1))
    out["stay"] = tree_json(ms, F_); out["dir"] = tree_json(md, dfeats)
    json.dump(out, open(path, "w"))
