"""extract whole-game M decisions: collector next-stop candidates, planter next-stay candidates, base recruits.
   python m2fit/extract_all.py [names...]  -> m2fit/ex_<tag>.pkl"""
import os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from m2fit.feats import board, col_cands, D, NBR, DIRS, MV
from m2fit.load import load
from mfit.extract import state_feats
from mfit.extract_tgt import cand_feats, enrich, best_at

def next_stops(g, kind):
    """(u,t) -> next stop cell (stay cell, or own base it arrives at for collectors)"""
    me = g["me"]; P = g["P"]; A = g["act"]; T = len(A)
    out = {}
    units = {u for t in range(T) for u, w in P[t][me][2].items() if w[2] == kind}
    for u in units:
        nxt = None
        for t in reversed(range(T)):
            w = P[t][me][2].get(u)
            if w is None:
                nxt = None; continue
            pos = int(w[0]); a = A[t].get(u)
            if a not in MV:
                nxt = pos
            elif kind == "COLLECTOR":
                w1 = P[t + 1][me][2].get(u)
                if w1 is not None and int(w1[0]) in {int(x) for x in P[t + 1][me][1].values()}:
                    nxt = int(w1[0])
            if nxt is not None:
                out[(u, t)] = nxt
    return out

def one(g, stride_c=2):
    me = g["me"]; P = g["P"]; A = g["act"]; T = len(A)
    cn = next_stops(g, "COLLECTOR"); pn = next_stops(g, "PLANTER")
    C, PL, B = [], [], []
    seen = {"PLANTER": set(), "COLLECTOR": set()}
    for t in range(T):
        for u, w in P[t][me][2].items():
            seen[w[2]].add(u)
        car = g["car"][t]
        b = board(P[t], me, car)
        # collectors
        if t % stride_c == 0:
            for u in sorted(b["ocol"]):
                if (u, t) not in cn:
                    continue
                cells, X = col_cands(b, u, t)
                tg = cn[(u, t)]
                hit = np.nonzero(cells == tg)[0]
                if not len(hit):
                    continue
                a = A[t].get(u); a = a if a in MV else "STAY"
                C.append(dict(g=g["g"], t=t, u=u, pos=b["ocol"][u][0], y=a, tgt=tg, cells=cells.astype(np.int16), X=X))
        # planters
        pls = [(u, int(w[0])) for u, w in P[t][me][2].items() if w[2] == "PLANTER" and pn.get((u, t)) is not None]
        ctx, arr = state_feats(P[t], me, car, t)
        ctx2 = dict(ctx, rec_pl=len(seen["PLANTER"]), rec_col=len(seen["COLLECTOR"]))
        if pls:
            enrich(P[t], me, arr)
            age = {int(p): a for q in (0, 1) for p, a in P[t][q][3].values()}
            for u, pos in pls:
                tg = pn[(u, t)]
                if D[pos, tg] > 7:
                    continue
                arr["best_at"][pos] = best_at(arr, pos)
                cells, cols = cand_feats(pos, ctx, arr, dict(age=age))
                a = A[t].get(u)
                PL.append(dict(g=g["g"], t=t, pos=pos, y=a if a in MV else "STAY", tgt=tg, cells=cells, cols=cols))
        for bi, (bid, bpos) in enumerate(sorted(P[t][me][1].items())):
            a = A[t].get(bid)
            B.append(dict(ctx2, g=g["g"], bidx=bi, nbase=len(P[t][me][1]), y={"RECCOLLECTOR": "C", "RECPLANTER": "P"}.get(a, "N"),
                          base_occ=int(any(int(w[0]) == int(bpos) for w in P[t][me][2].values())), left=300 - t,
                          ecash=float(P[t][1 - me][0])))
    return C, PL, B

if __name__ == "__main__":
    from multiprocessing import Pool
    names = tuple(sys.argv[1:]) or ("RyanX-AIPowercore-M2",)
    G = load(names)
    with Pool(10) as pool:
        R = pool.map(one, G)
    C = [r for x in R for r in x[0]]; PL = [r for x in R for r in x[1]]; B = [r for x in R for r in x[2]]
    tag = "m2" if names == ("RyanX-AIPowercore-M2",) else "all"
    pickle.dump(dict(C=C, PL=PL, B=B), open(os.path.join(HERE, f"ex_{tag}.pkl"), "wb"))
    print(len(C), len(PL), len(B), sum(len(r["cells"]) for r in C))
