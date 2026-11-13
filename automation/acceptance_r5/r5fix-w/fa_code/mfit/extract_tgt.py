"""candidate-target table: for each M planter-step (steps 0..99) whose next stay cell is known (the cell where it next stays:
plants / seizes / waits), every cell within RAD steps is a candidate; label = it is that next stay cell."""
import gzip, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from mfit.extract import state_feats, D, NBR, DIRS, MV, step
RAD = 7

def cand_feats(pos, ctx, arr, extra):
    """per-candidate features for planter at pos; returns (cells, matrix, names)"""
    nb, occ = arr["nb"], arr["occ"]
    cells = np.nonzero((D[pos] <= RAD) & (occ != 3))[0]
    d = D[pos, cells]
    opl = [o for o in arr["opl"] if o != pos]
    oplc = np.array([sum(1 for o in opl if D[o, c] < D[pos, c]) for c in cells])
    age = extra["age"]  # dict cell -> tree age (any player)
    cols = dict(c_nb=nb[cells], c_dist=d, c_free=(occ[cells] == 0).astype(int), c_etree=(occ[cells] == 2).astype(int),
                c_otree=(occ[cells] == 1).astype(int), c_age=np.array([age.get(int(c), -1) for c in cells]),
                c_depl=arr["depl"][cells], c_decol=arr["decol"][cells], c_dbase=arr["dbase"][cells], c_debase=arr["debase"][cells],
                c_oplcloser=oplc, c_here=(d == 0).astype(int),
                c_nbgain=nb[cells] - arr["best_at"][pos][np.minimum(d, RAD)], c_score=nb[cells] - 20 * d,
                step=np.full(len(cells), ctx["step"]), cash=np.full(len(cells), ctx["cash"]), price=np.full(len(cells), ctx["price"]),
                afford=np.full(len(cells), int(ctx["cash"] >= ctx["price"])), own_tr=np.full(len(cells), ctx["own_tr"]),
                en_tr=np.full(len(cells), ctx["en_tr"]))
    return cells, cols

def enrich(P, me, arr):
    eb = [int(b) for b in P[1 - me][1].values()]
    arr["debase"] = D[:, eb].min(1) if eb else np.full(225, 30)
    nb, occ = arr["nb"], arr["occ"]
    val = np.where(occ == 0, nb, 0.0)
    # best_at[pos][k] = best free nb4 exactly within distance <= k of pos
    arr["best_at"] = {}
    return arr

def best_at(arr, pos):
    nb, occ = arr["nb"], arr["occ"]; val = np.where(occ == 0, nb, 0.0)
    return np.array([val[D[pos] <= k].max() for k in range(RAD + 1)])

if __name__ == "__main__":
    root = os.path.join(HERE, "..", "..", "data", "m2_corpus")
    idx = json.load(open(os.path.join(root, "index.json")))
    rows = []
    rng = np.random.RandomState(0)
    for gi, e in enumerate(idx):
        mseats = [i for i, s in enumerate(e["order"]) if e["strategies"][s]["name"].startswith("RyanX-AIPowercore-M")]
        d = json.load(gzip.open(os.path.join(root, e["file"])))["steps"]
        for me in mseats:
            T = min(100, len(d) - 1)
            acts = [d[t + 1][me].get("action") or {} for t in range(T)]
            obs = [d[t][0]["observation"] for t in range(T)]
            # per unit: positions and stay flags
            nxt = {}  # (u, t) -> next stay cell at or after t
            units = {u for o in obs for u, w in o["players"][me][2].items() if w[2] == "PLANTER"}
            for u in units:
                cur = None
                for t in reversed(range(T)):
                    w = obs[t]["players"][me][2].get(u)
                    if w is None or w[2] != "PLANTER":
                        cur = None; continue
                    if acts[t].get(u) not in MV: cur = int(w[0])
                    nxt[(u, t)] = cur
            for t in range(T):
                P = obs[t]["players"]
                pls = [(u, int(w[0])) for u, w in P[me][2].items() if w[2] == "PLANTER" and nxt.get((u, t)) is not None]
                if not pls: continue
                ctx, arr = state_feats(P, me, obs[t]["full_carbon"], t); enrich(P, me, arr)
                age = {int(p): a for q in (0, 1) for p, a in P[q][3].values()}
                for u, pos in pls:
                    tg = nxt[(u, t)]
                    if D[pos, tg] > 7: continue
                    arr["best_at"][pos] = best_at(arr, pos)
                    cells, cols = cand_feats(pos, ctx, arr, dict(age=age))
                    a = acts[t].get(u)
                    rows.append(dict(g=gi, t=t, pos=pos, y=a if a in MV else "STAY", tgt=tg, cells=cells, cols=cols))
    pickle.dump(rows, open(os.path.join(HERE, "tgt.pkl"), "wb")); print(len(rows))
