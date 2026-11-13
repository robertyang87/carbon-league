"""cnet2_search_replay -- deterministic replay of one seed for the clock-free mfx config (or v12) vs an opponent,
printing per-step cash / unit counts, every death of our units (cell, cargo, who else was there) and search overrides.
python3 cnet2_search_replay.py SEED SIDE OPP.py [v12|mfx] [MAXSTEP] [CONF_OVERRIDES]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eval_v53, cnet2_search as S
from fastsim import FastCarbon
MFX = {'MODE': 'mcts', 'CLOCK': 'cpu', 'BUDGET': 99.0, 'HARD': 99.0, 'START_DL': None, 'ABORT_DL': None, 'ROOT_S': 12,
       'MINC': 20.0, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'RW': 'cnet2_s64_q8.npz', 'MIX': True}
seed, side, opp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
who = sys.argv[4] if len(sys.argv) > 4 else "mfx"
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 300
if len(sys.argv) > 6:
    MFX.update(eval(sys.argv[6]))
W = os.path.join(HERE, "..", "data", "weights", "cnet2_b128_q8.npz")
ag = S.make_search_agent(dict(MFX, LOG=True), weights=W) if who == "mfx" else eval_v53.load_agent_fn("exp/submission_cnet2_v12.py")
ob = eval_v53.load_agent_fn(opp)
sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
nlog = 0
for t in range(300):
    pre = sim.get_state()
    o = [sim.obs(0), sim.obs(1)]
    a = ag(o[side], eval_v53.CFG); b = ob(o[1 - side], eval_v53.CFG)
    ov = []
    if who == "mfx":
        for l in ag.log[nlog:]:
            if l[3] is not None and l[3] > MFX["MARGIN"]:
                ov.append(l[1][[i for i in range(len(l[1]))]][0] if False else l)
        nlog = len(ag.log)
    sim.step(*((a, b) if side == 0 else (b, a)))
    post = sim.get_state()
    dead = []
    for wid, w in pre["workers"].items():
        if w["player"] == side and wid not in post["workers"]:
            # who ended on / moved to that cell
            mv = a.get(wid, "NONE"); dest = sim._move_nbr[mv][w["pos"]] if mv in sim._move_nbr else w["pos"]
            others = [(k, v["player"], round(v["carbon"]), v["type"]) for k, v in post["workers"].items() if v["pos"] == dest]
            dead.append((wid, w["type"], round(w["carbon"]), mv, dest, others))
    mine = [w for w in post["workers"].values() if w["player"] == side]
    theirs = [w for w in post["workers"].values() if w["player"] != side]
    if t < mx and (t < 60 or dead or ov):
        line = {"step": t, "cash": [round(post["cash"][side]), round(post["cash"][1 - side])],
                "units": [sum(w["type"] == "COLLECTOR" for w in mine), sum(w["type"] == "PLANTER" for w in mine),
                          sum(w["type"] == "COLLECTOR" for w in theirs), sum(w["type"] == "PLANTER" for w in theirs)],
                "trees": [sum(tt["player"] == side for tt in post["trees"].values()), sum(tt["player"] != side for tt in post["trees"].values())],
                "base_acts": {k: v for k, v in a.items() if "recrt" in k}}
        if dead:
            line["dead"] = dead
        if ov:
            line["override"] = [(l[1], l[3]) for l in ov]
        print(json.dumps(line), flush=True)
print(json.dumps({"final": [round(sim.cash[side]), round(sim.cash[1 - side])], "margin": round(sim.cash[side] - sim.cash[1 - side]),
                  "stats": {k: v for k, v in (ag.stats.items() if who == "mfx" else []) if k in ("searched", "changed", "sims", "errors")}}))
