"""per-tree income/lifetime stats; replays (M2 and its opponent) or a sim game (m2bot vs OPP)"""
import os, sys, json, gzip, importlib.util
import numpy as np
sys.path.insert(0, ".")
def tree_stats(obs_list, me):
    life = {}; inc = {}; owner_end = {}
    for o in obs_list:
        for q in (0, 1):
            for tid in o["players"][q][3]:
                key = (tid, q)
                life[key] = life.get(key, 0) + 1
                inc[key] = inc.get(key, 0.0) + float(o["trees"].get(tid, [0, 0])[1] or 0)
    out = {}
    for side, q in (("me", me), ("op", 1 - me)):
        ks = [k for k in life if k[1] == q]
        out[side] = dict(n=len(ks), life=np.mean([life[k] for k in ks]) if ks else 0, inc=np.mean([inc[k] for k in ks]) if ks else 0,
                         inc_step=sum(inc[k] for k in ks) / max(1, sum(life[k] for k in ks)), tot=sum(inc[k] for k in ks))
    return out
if sys.argv[1] == "replay":
    from m2fit.load import load
    idx = json.load(open("../data/m2_corpus/index.json"))
    R = []
    for g in load()[:30]:
        st = json.load(gzip.open("../data/m2_corpus/" + idx[g["g"]]["file"]))["steps"]
        R.append(tree_stats([s[0]["observation"] for s in st], g["me"]))
else:
    from fastsim import FastCarbon
    os.environ["M2BOT_CFG"] = sys.argv[4] if len(sys.argv) > 4 else "{}"
    R = []
    for s in range(int(sys.argv[2]), int(sys.argv[2]) + int(sys.argv[3])):
        me = s % 2
        def ld(p, nm):
            sp = importlib.util.spec_from_file_location(nm + str(s), p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.agent
        ag = [None, None]; ag[me] = ld("exp/m2bot.py", "a"); ag[1 - me] = ld(sys.argv[1], "b")
        sim = FastCarbon({}).reset(seed=s); OL = []
        for t in range(299):
            o = [sim.obs(p) for p in range(2)]; OL.append(o[0])
            sim.step(ag[0](o[0], {}), ag[1](o[1], {}))
        R.append(tree_stats(OL, me))
for side in ("me", "op"):
    print(side, {k: round(float(np.mean([r[side][k] for r in R])), 2) for k in R[0][side]})
