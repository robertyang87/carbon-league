"""whole-game stats of M2 seats in the corpus (likeness reference)"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from m2fit.load import load

def seat_stats(g):
    me = g["me"]; P = g["P"]; T = len(P)
    bases = lambda t, q: {int(b) for b in P[t][q][1].values()}
    dep = 0.0; ndep = 0; depc = []; seen = {"PLANTER": set(), "COLLECTOR": set()}
    plants = seizes = 0
    for t in range(T - 1):
        W0, W1 = P[t][me][2], P[t + 1][me][2]
        for u, w in W1.items():
            seen[w[2]].add(u)
        b = bases(t + 1, me)
        for u, w in W0.items():
            if w[2] == "COLLECTOR" and u in W1 and int(W1[u][0]) in b and int(W1[u][0]) != int(w[0]) and w[1] > 0:
                dep += w[1]; ndep += 1; depc.append((t, w[1]))
        tr0 = {k: v for k, v in P[t][me][3].items()}; tr1 = P[t + 1][me][3]
        for k in tr1:
            if k not in tr0:
                if k.startswith(f"player-{me}"): plants += 1
                else: seizes += 1
        # a seized tree keeps its id but switches owner list
    trees = lambda t: len(P[t][me][3])
    cargo_end = sum(w[1] for w in P[T - 1][me][2].values())
    return dict(score=g["score"], oscore=g["oscore"], win=g["score"] > g["oscore"], dep=dep, ndep=ndep,
                dep_mean=dep / max(1, ndep), tr50=trees(50), tr100=trees(100), tr150=trees(150), tr200=trees(200), tr250=trees(250),
                recP=len(seen["PLANTER"]), recC=len(seen["COLLECTOR"]), plants=plants, seizes=seizes, cargo_end=cargo_end,
                ncol100=sum(w[2] == "COLLECTOR" for w in P[100][me][2].values()), ncol200=sum(w[2] == "COLLECTOR" for w in P[200][me][2].values()),
                npl100=sum(w[2] == "PLANTER" for w in P[100][me][2].values()), npl200=sum(w[2] == "PLANTER" for w in P[200][me][2].values()),
                dep100=sum(c for t, c in depc if t < 100), dep200=sum(c for t, c in depc if 100 <= t < 200), dep300=sum(c for t, c in depc if t >= 200),
                tinc=sum(g["tinc"]))

if __name__ == "__main__":
    import pandas as pd
    for nm in ("RyanX-AIPowercore-M2", "RyanX-AIPowercore-M3", "RyanX-AIPowercore-M4"):
        G = load((nm,))
        S = pd.DataFrame([seat_stats(g) for g in G])
        print(nm, len(G)); print(S.mean().round(1).to_string())
