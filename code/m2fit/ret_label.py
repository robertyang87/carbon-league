"""'returning' label per collector row: from now until its next deposit the collector's distance to the nearest own
base never increases (it may stop to collect on the way)."""
import os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from m2fit.load import load
from m2fit.feats import D

def labels(g):
    me = g["me"]; P = g["P"]; T = len(P)
    out = {}
    units = {u for t in range(T) for u, w in P[t][me][2].items() if w[2] == "COLLECTOR"}
    for u in units:
        traj = []
        for t in range(T):
            w = P[t][me][2].get(u)
            bs = [int(x) for x in P[t][me][1].values()]
            traj.append(None if w is None else (int(w[0]), float(w[1]), min(D[int(w[0]), b] for b in bs) if bs else 30))
        nxt_ok = None   # for t going backwards: returning flag
        for t in reversed(range(T - 1)):
            if traj[t] is None:
                nxt_ok = None; continue
            p, c, db = traj[t]
            n = traj[t + 1]
            if n is None:
                nxt_ok = None; continue
            if n[2] == 0 and db > 0 and c > 0:      # deposits on this move
                nxt_ok = True
            elif nxt_ok is not None:
                nxt_ok = nxt_ok and n[2] <= db
            out[(u, t)] = nxt_ok
    return out

if __name__ == "__main__":
    G = {g["g"]: g for g in load()}
    R = pickle.load(open(os.path.join(HERE, "ex_m2.pkl"), "rb"))["C"]
    L = {}
    for gi, g in G.items():
        for k, v in labels(g).items(): L[(gi,) + k] = v
    y = np.array([bool(L.get((r["g"], r["u"], r["t"]))) for r in R])
    pickle.dump(y, open(os.path.join(HERE, "ret_m2.pkl"), "wb")); print(y.mean())
