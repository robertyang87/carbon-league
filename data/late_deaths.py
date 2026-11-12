"""late_deaths OURS.py OPP.py SEED SIDE -- every loss of a loaded own car after step 270: where, how far from each base,
which enemy collectors were within 2 cells (cargo), and whether the go-home rule held it in the 3 steps before."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../code"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../code/env_src/carbon_challenge-master")
import eval_v53, cnet2_feat as F
from fastsim import FastCarbon
ours, opp, seed, side = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
random.seed(eval_v53.rng_seed(seed, side)); sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
C = os.path.dirname(os.path.abspath(__file__)) + "/../code/"
fa = eval_v53.load_agent_fn(C + ours); fb = eval_v53.load_agent_fn(C + opp)
for step in range(1, 301):
    oa, ob = sim.obs(side), sim.obs(1 - side)
    a = fa(oa, eval_v53.CFG); b = fb(ob, eval_v53.CFG)
    P = oa["players"]; mine = dict(P[side][2]); bases = [int(x) for x in P[side][1].values()]
    en = [e for e in P[1 - side][2].values() if e[2] == "COLLECTOR"]
    sim.step(*((a, b) if side == 0 else (b, a)))
    O = sim.obs(side)["players"]; after = O[side][2]
    for uid, (pos, cargo, kind) in mine.items():
        if step > 270 and uid not in after and cargo > 0:
            act = set(map(tuple, (getattr(fa, "last_stats", None) or {}).get("eh_act", [])))
            held = [s for s in range(step - 4, step) if (s, uid) in act]
            print(f"t{step} car {uid[-4:]} cargo {cargo:.0f} pos {divmod(int(pos),15)} move {a.get(uid)} dist-to-bases {[int(F.DIST[pos, x]) for x in bases]} "
                  f"left {300-step} | enemy cols <=2: {[(divmod(int(e[0]),15), round(e[1]), int(F.DIST[pos, e[0]])) for e in en if F.DIST[pos, e[0]] <= 2]} | rule held at {held}")
            nxt = [(u2[-4:], round(w2[1]), divmod(int(w2[0]), 15)) for u2, w2 in mine.items() if u2 != uid and F.DIST[pos, w2[0]] <= 2]
            occ = [("own" if q == side else "ENEMY", u2[-4:], divmod(int(w2[0]), 15), round(w2[1])) for q in (0, 1) for u2, w2 in O[q][2].items() if F.DIST[pos, w2[0]] <= 1]
            print(f"     own units <=2 before: {nxt} | own moves {[(u2[-4:], a.get(u2)) for u2, w2 in mine.items() if F.DIST[pos, w2[0]] <= 2]} | units <=1 of old cell after: {occ}")
