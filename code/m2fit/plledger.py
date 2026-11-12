"""planter + tree ledger: python m2fit/plledger.py OPP SEED N [cfg]  (m2bot vs OPP in fastsim), or 'replay' (M2 corpus)"""
import os, sys, json, importlib.util
import numpy as np
from collections import Counter
sys.path.insert(0, "."); from m2fit.feats import D, NBR, DIRS
MV = set(DIRS)
def nx(p, a): return int(NBR[p][DIRS.index(a)]) if a in MV else p

def ledger(Ps, A, me, trees_obs):
    """Ps[t]: players at t; A[t] = (acts p0, acts p1) chosen at t; trees_obs[t]: obs trees dict (income)"""
    out = {}
    for side, q in (("me", me), ("op", 1 - me)):
        cause = Counter(); born = {}; yields = Counter(); life = []
        tlife = []; tend = Counter(); tinc = 0.0; tsteps = 0
        tb = {}
        for t in range(len(Ps) - 1):
            P, P1 = Ps[t], Ps[t + 1]; a = A[t]
            W, W1 = P[q][2], P1[q][2]
            T0 = {k: int(v[0]) for k, v in P[q][3].items()}; T1 = {k: int(v[0]) for k, v in P1[q][3].items()}
            for k in T1:
                if k not in T0:
                    tb[k] = t + 1
                    pl = [u for u, w in W.items() if w[2] == "PLANTER" and int(w[0]) == T1[k]]
                    for u in pl: yields[u] += 1
            for k in T0:
                tinc += float(trees_obs[t].get(k, [0, 0])[1] or 0); tsteps += 1
                if k not in T1:
                    age = P[q][3][k][1]
                    tend["withered" if age >= 49 else ("seized" if k in {kk for kk in P1[1 - q][3]} else "other")] += 1
                    if k in tb: tlife.append(t + 1 - tb[k])
            for u, w in W.items():
                if w[2] != "PLANTER": continue
                born.setdefault(u, t)
                if u in W1: continue
                c = nx(int(w[0]), a[q].get(u))
                mv = "move" if a[q].get(u) in MV else "stay"
                E = Ps[t][1 - q][2]
                ent = [E[v][2][0] for v in E if nx(int(E[v][0]), a[1 - q].get(v)) == c]
                own = [W[v][2][0] for v in W if v != u and nx(int(W[v][0]), a[q].get(v)) == c]
                k = ("enemyP" if "P" in ent and "C" not in ent else "enemyC" if "C" in ent else "ownC" if "C" in own else "ownP" if own else "?") + "/" + mv
                cause[k + ("/0yield" if yields[u] == 0 else "/yielded")] += 1
                life.append(t + 1 - born[u])
        out[side] = dict(deaths=cause, plife=np.mean(life) if life else 0, yields=sum(yields.values()),
                         planters=len(born), tree_life=np.mean(tlife) if tlife else 0, tree_end=tend, inc_per_treestep=tinc / max(1, tsteps))
    return out

if __name__ == "__main__":
    R = []
    if sys.argv[1] == "replay":
        import gzip
        from m2fit.load import load
        idx = json.load(open("../data/m2_corpus/index.json"))
        for g in [g for g in load(None) if "claude-opus55" in g["opp"]][:20]:
            st = json.load(gzip.open("../data/m2_corpus/" + idx[g["g"]]["file"]))["steps"]
            me = g["me"]; A = [((g["act"][t] if me == 0 else g["oact"][t]), (g["oact"][t] if me == 0 else g["act"][t])) for t in range(299)]
            R.append(ledger(g["P"], A, me, [s[0]["observation"]["trees"] for s in st]))
    else:
        from fastsim import FastCarbon
        if len(sys.argv) > 4: os.environ["M2BOT_CFG"] = sys.argv[4]
        for s in range(int(sys.argv[2]), int(sys.argv[2]) + int(sys.argv[3])):
            me = s % 2
            def ld(p, nm):
                sp = importlib.util.spec_from_file_location(nm + str(s), p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
            ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld(sys.argv[1], "b")
            sim = FastCarbon({}).reset(seed=s); Ps = []; A = []; TO = []
            for t in range(300):
                o = [sim.obs(p) for p in range(2)]; Ps.append(o[0]["players"]); TO.append(o[0]["trees"])
                if t == 299: break
                a = (ag[0](o[0], {}), ag[1](o[1], {})); A.append(a); sim.step(a[0], a[1])
            R.append(ledger(Ps, A, me, TO))
    n = len(R)
    for side in ("me", "op"):
        dc = Counter(); te = Counter()
        for r in R: dc.update(r[side]["deaths"]); te.update(r[side]["tree_end"])
        print(side, "planters/game", round(np.mean([r[side]["planters"] for r in R]), 1), "planter life", round(np.mean([r[side]["plife"] for r in R]), 1),
              "trees planted+seized/game", round(np.mean([r[side]["yields"] for r in R]), 1), "tree life", round(np.mean([r[side]["tree_life"] for r in R]), 1),
              "income/tree-step", round(np.mean([r[side]["inc_per_treestep"] for r in R]), 2))
        print("   deaths/game", {k: round(v / n, 1) for k, v in dc.most_common()})
        print("   tree ends/game", {k: round(v / n, 1) for k, v in te.most_common()})
