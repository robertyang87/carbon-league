"""option-level (stay + 4 moves) rows for collectors and planters, given the fitted target trees; -> m2fit/opt_<tag>.pkl"""
import os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from m2fit.feats import board, DIRS
from m2fit.load import load
from m2fit.policy import load_trees, col_target, col_opts, pl_target, pl_opts, OPTS
from mfit.extract import state_feats
from mfit.extract_tgt import enrich
T = load_trees(os.path.join(HERE, ".."), ("m2_bot_ctrees.json", "m2_bot_ptrees.json"))

def lab(a):
    return OPTS.index(a) if a in DIRS else 0

def one(g):
    me = g["me"]; P = g["P"]; A = g["act"]; Tn = len(A)
    C, PL = [], []
    for t in range(Tn):
        b = board(P[t], me, g["car"][t])
        if t % 2 == 0:
            taken = set()
            for u in sorted(b["ocol"], key=lambda u: -b["ocol"][u][1]):
                tg, home, rf, sp, hp = col_target(T, b, u, t, taken); taken.add(tg)
                C.append(dict(g=g["g"], t=t, X=col_opts(b, u, tg, home, sp, hp, t), y=lab(A[t].get(u))))
        pls = sorted(u for u, w in P[t][me][2].items() if w[2] == "PLANTER")
        if pls:
            ctx, arr = state_feats(P[t], me, g["car"][t], t); enrich(P[t], me, arr)
            age = {int(p): a for q in (0, 1) for p, a in P[t][q][3].values()}
            pt = set()
            for u in pls:
                pos = int(P[t][me][2][u][0])
                tg, tp = pl_target(T, ctx, arr, age, pos, pt); pt.add(tg)
                PL.append(dict(g=g["g"], t=t, X=pl_opts(arr, ctx, pos, tg, tp, b), y=lab(A[t].get(u))))
    return C, PL

if __name__ == "__main__":
    from multiprocessing import Pool
    names = tuple(sys.argv[1:]) or ("RyanX-AIPowercore-M2",)
    with Pool(10) as pool:
        R = pool.map(one, load(names))
    tag = "m2" if names == ("RyanX-AIPowercore-M2",) else "all"
    C = [r for x in R for r in x[0]]; PL = [r for x in R for r in x[1]]
    pickle.dump(dict(C=C, PL=PL), open(os.path.join(HERE, f"opt_{tag}.pkl"), "wb")); print(len(C), len(PL))
