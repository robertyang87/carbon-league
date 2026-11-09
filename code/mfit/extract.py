"""extract M-family (RyanX-AIPowercore-M*) opening decisions (steps 0..99) from data/m2_corpus into mfit/ex.pkl"""
import gzip, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import cnet2_feat as F
S = 15; N = 225
MV = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
D = F.DIST
def step(p, mv):
    r, c = divmod(p, S); dr, dc = MV[mv]; return ((r + dr) % S) * S + (c + dc) % S
NBR = np.array([[step(p, m) for m in DIRS] for p in range(N)])
def nb4(car): return np.asarray(car, float)[NBR].sum(1)

def state_feats(P, me, carbon, t):
    """shared context + per-cell arrays used by both the extractor and the playing wrapper"""
    op = 1 - me
    nb = nb4(carbon)
    occ = np.zeros(N, int)  # 0 free 1 own tree 2 enemy tree 3 base
    for q in (0, 1):
        for pos, _ in P[q][3].values(): occ[int(pos)] = 1 if q == me else 2
        for b in P[q][1].values(): occ[int(b)] = 3
    own_tr = len(P[me][3]); en_tr = len(P[op][3])
    live = sum(1 for q in (0, 1) for _, a in P[q][3].values() if a < 50)
    price = 20.0 if own_tr == 0 else 20.0 + 5.0 * 1.235 ** live
    W = P[me][2]; E = P[op][2]
    ctx = dict(step=t, cash=float(P[me][0]), price=price, own_tr=own_tr, en_tr=en_tr,
               n_col=sum(w[2] == "COLLECTOR" for w in W.values()), n_pl=sum(w[2] == "PLANTER" for w in W.values()),
               en_col=sum(w[2] == "COLLECTOR" for w in E.values()), en_pl=sum(w[2] == "PLANTER" for w in E.values()),
               n_wk=len(W), cargo=sum(float(w[1]) for w in W.values()))
    bases = [int(b) for b in P[me][1].values()]
    epl = [int(w[0]) for w in E.values() if w[2] == "PLANTER"]
    ecol = [int(w[0]) for w in E.values() if w[2] == "COLLECTOR"]
    opl = [int(w[0]) for w in W.values() if w[2] == "PLANTER"]
    big = 30
    dmin = lambda L: D[:, L].min(1) if L else np.full(N, big)
    arr = dict(nb=nb, occ=occ, dbase=dmin(bases), depl=dmin(epl), decol=dmin(ecol),
               detree=dmin([i for i in range(N) if occ[i] == 2]), dfree_good=None)
    # best nb4 of a free (or enemy-tree) cell within k steps of each cell
    free = occ == 0
    val = np.where(free, nb, 0.0); valE = np.where(occ == 2, nb, 0.0)
    for k in (1, 2, 3):
        arr[f"best{k}"] = np.array([val[D[c] <= k].max() for c in range(N)])
        arr[f"bestE{k}"] = np.array([valE[D[c] <= k].max() for c in range(N)])
    arr["epl"] = epl; arr["ecol"] = ecol; arr["opl"] = opl
    return ctx, arr

def planter_feats(pos, ctx, arr):
    nb, occ = arr["nb"], arr["occ"]
    f = dict(ctx)
    f.update(nb_here=nb[pos], occ_here=occ[pos], free_here=int(occ[pos] == 0), etree_here=int(occ[pos] == 2),
             otree_here=int(occ[pos] == 1), base_here=int(occ[pos] == 3),
             d_etree=arr["detree"][pos], d_epl=arr["depl"][pos], d_ecol=arr["decol"][pos], d_base=arr["dbase"][pos],
             best1=arr["best1"][pos], best2=arr["best2"][pos], best3=arr["best3"][pos],
             bestE1=arr["bestE1"][pos], bestE2=arr["bestE2"][pos],
             afford=int(ctx["cash"] >= ctx["price"]), nb_minus_best1=nb[pos] - arr["best1"][pos],
             n_opl_adj=sum(1 for o in arr["opl"] if o != pos and D[o, pos] <= 1))
    return f

def dir_feats(pos, mv, ctx, arr, pf):
    nb, occ = arr["nb"], arr["occ"]
    q = int(NBR[pos][DIRS.index(mv)])
    g = dict(d_nb=nb[q], d_nb_rel=nb[q] - nb[pos], d_free=int(occ[q] == 0), d_etree=int(occ[q] == 2), d_otree=int(occ[q] == 1),
             d_base=int(occ[q] == 3), d_epl1=int(arr["depl"][q] <= 1), d_epl0=int(arr["depl"][q] == 0),
             d_ecol1=int(arr["decol"][q] <= 1), d_dbase=arr["dbase"][q] - arr["dbase"][pos],
             d_best1=arr["best1"][q], d_best2=arr["best2"][q], d_bestE1=arr["bestE1"][q],
             d_best2_rel=arr["best2"][q] - arr["best3"][pos], d_detree=arr["detree"][q] - arr["detree"][pos],
             d_opl=int(any(o != pos and D[o, q] <= 1 for o in arr["opl"])))
    return g

if __name__ == "__main__":
    root = os.path.join(HERE, "..", "..", "data", "m2_corpus")
    idx = json.load(open(os.path.join(root, "index.json")))
    rows_b, rows_p = [], []
    for gi, e in enumerate(idx):
        sides = [s for s in e["order"]]
        mseats = [i for i, s in enumerate(sides) if e["strategies"][s]["name"].startswith("RyanX-AIPowercore-M")]
        if not mseats: continue
        try:
            d = json.load(gzip.open(os.path.join(root, e["file"])))
        except Exception as ex:
            print("skip", e["file"], ex); continue
        st = d["steps"]
        for me in mseats:
            name = e["strategies"][sides[me]]["name"]
            seen = {"PLANTER": set(), "COLLECTOR": set()}
            fb = F.Feat(me)
            for t in range(min(100, len(st) - 1)):
                o = st[t][0]["observation"]; P = o["players"]
                act = st[t + 1][me].get("action") or {}
                for u, w in P[me][2].items(): seen[w[2]].add(u)
                ctx, arr = state_feats(P, me, o["full_carbon"], t)
                ctx2 = dict(ctx, rec_pl=len(seen["PLANTER"]), rec_col=len(seen["COLLECTOR"]))
                fb.observe({"step": t, "player": me, "players": P, "carbon": st[t][me]["observation"]["carbon"]})
                _, arrb = state_feats(P, me, fb.est.copy(), t)
                for bi, (bid, bpos) in enumerate(sorted(P[me][1].items())):
                    a = act.get(bid)
                    lab = {"RECCOLLECTOR": "C", "RECPLANTER": "P"}.get(a, "N")
                    rows_b.append(dict(ctx2, g=gi, name=name, bidx=bi, nbase=len(P[me][1]), y=lab,
                                       base_occ=int(any(int(w[0]) == int(bpos) for w in P[me][2].values()))))
                for u, w in P[me][2].items():
                    if w[2] != "PLANTER": continue
                    pos = int(w[0]); pf = planter_feats(pos, ctx2, arr)
                    a = act.get(u); y = a if a in MV else "STAY"
                    r = dict(pf, g=gi, name=name, y=y, pos=pos, uid=u)
                    for k, v in planter_feats(pos, ctx2, arrb).items():
                        if k not in ctx2: r["b_" + k] = v
                    for mv in DIRS:
                        for k, v in dir_feats(pos, mv, ctx2, arr, pf).items(): r[mv + ":" + k] = v
                        for k, v in dir_feats(pos, mv, ctx2, arrb, pf).items(): r[mv + ":b_" + k] = v
                    rows_p.append(r)
    pickle.dump(dict(base=rows_b, pl=rows_p), open(os.path.join(HERE, "ex.pkl"), "wb"))
    print(len(rows_b), len(rows_p))
