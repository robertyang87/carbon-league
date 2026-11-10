"""play m2bot vs an opponent with fastsim and print collector-economy diagnostics; same metrics on M2 replays
   python m2fit/diag.py replay | python m2fit/diag.py game OPP.py SEED N [cfgjson]"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from m2fit.feats import CHEB, D

def metrics(states, me):
    """states: list of (P, car) per step t (P = obs players list); returns dict"""
    absorb = [0.0, 0.0]; side = []; nmov = [0, 0]; lastpos = {}; ncol = []; colsteps = stays = 0; coll_amt = 0.0; dep = []; carstay = []
    for t in range(len(states) - 1):
        P, car = states[t]; P1, _ = states[t + 1]
        W, W1 = P[me][2], P1[me][2]; b1 = {int(x) for x in P1[me][1].values()}
        cs = [u for u, w in W.items() if w[2] == "COLLECTOR"]; ncol.append(len(cs))
        et = [int(q) for q, _ in P1[1 - me][3].values()]
        for u in cs:
            if u in W1 and et and (CHEB[int(W1[u][0]), et] == 1).any(): absorb[0] += 0.1 * W1[u][1] / 0.9
        for u in cs:
            if u not in W1:
                absorb[1] += W[u][1]; continue
            colsteps += 1
            p0, p1 = int(W[u][0]), int(W1[u][0])
            if p0 != p1:
                pp = lastpos.get(u)
                nmov[0] += 1; nmov[1] += pp is not None and pp == p1
                lastpos[u] = p0
            elif p0 == p1:
                lastpos[u] = None
            if p0 == p1 and p0 not in b1:
                eb = [int(x) for x in P[1 - me][1].values()]; ob = [int(x) for x in P[me][1].values()]
                if eb and ob: side.append(min(D[p0, q] for q in ob) - min(D[p0, q] for q in eb))
                stays += 1; coll_amt += max(0, W1[u][1] - W[u][1]); carstay.append(car[p0])
            if p1 in b1 and p1 != p0 and W[u][1] > 0:
                dep.append(W[u][1])
    return dict(ncol=np.mean(ncol), stayfrac=stays / max(1, colsteps), car_at_stay=np.mean(carstay), collected=coll_amt,
                absorbed=absorb[0], deathcargo=absorb[1], side=np.mean(side) if side else 0, reverse=nmov[1] / max(1, nmov[0]), enemyhalf=np.mean(np.array(side) > 0) if side else 0, strand=sum(w[1] for w in states[-1][0][me][2].values()), ndep=len(dep), depmean=np.mean(dep) if dep else 0, dep=sum(dep))

if __name__ == "__main__":
    if sys.argv[1] == "replay":
        from m2fit.load import load
        R = [metrics(list(zip(g["P"], g["car"])), g["me"]) for g in load()]
        print({k: round(float(np.mean([r[k] for r in R])), 3) for k in R[0]})
    else:
        import importlib.util
        from fastsim import FastCarbon
        opp, s0, n = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        if len(sys.argv) > 5: os.environ["M2BOT_CFG"] = sys.argv[5]
        full = json.loads(os.environ.get("M2BOT_CFG", "{}")).get("FULL")
        R = []
        for s in range(s0, s0 + n):
            def ld(p, nm):
                sp = importlib.util.spec_from_file_location(nm + str(s), p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
            me = s % 2
            ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld(opp, "b")
            sim = FastCarbon({}).reset(seed=s)
            states = []; tinc = 0.0
            for t in range(300):
                o = [sim.obs(p) for p in range(2)]
                states.append((o[0]["players"], np.array(sim.carbon, float)))
                tinc += sum(float(o[0]["trees"].get(k, [0, 0])[1] or 0) for k in o[0]["players"][me][3])
                if full: o[me]["full_carbon"] = list(sim.carbon)
                if t == 299: break
                sim.step(ag[0](o[0], {}), ag[1](o[1], {}))
            from m2fit.stats import seat_stats
            ss = seat_stats(dict(P=[x[0] for x in states], me=me, score=sim.cash[me], oscore=sim.cash[1 - me], tinc=[tinc]))
            m = metrics(states, me); m.update({"L_" + k: v for k, v in ss.items() if k in ("tr50", "tr100", "tr150", "tr200", "tr250", "recP", "recC", "plants", "seizes", "cargo_end", "tinc")}); om = metrics(states, 1 - me); m.update({"o_" + k: v for k, v in om.items() if k in ("reverse", "side", "enemyhalf", "ncol", "stayfrac", "car_at_stay", "collected", "absorbed", "deathcargo", "ndep", "depmean", "dep")}); m["margin"] = sim.cash[me] - sim.cash[1 - me]; R.append(m)
            print({k: round(float(v), 2) for k, v in m.items()}, flush=True)
        print("MEAN", {k: round(float(np.mean([r[k] for r in R])), 2) for k in R[0]})
