"""dagger_probe -- how far do OUR planters drift from the teacher on OUR OWN game states?

  python3 dagger_probe.py OURS.py LABELER.npz LABELER_SLOT OPP.py SEED0 N [WORKERS]

Plays OURS vs OPP (one game per seed, seats alternate). At every step, for each of our planters, asks the labeler
net (teacher-forced on our actual joint action, so earlier units are what we really played) for its greedy move
and compares it with the move we played. Prints the disagreement rate (all / NONE vs move / by phase) and the
context of disagreeing steps: distance to the nearest enemy worker and own trees within radius 2.
Writes one json line per game to ../data/dagger_probe.jsonl. Nothing is trained here.
"""
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("BC_W", "64")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))


def play(task):
    import numpy as np
    import eval_v53
    import cnet2_feat as F
    import cnet2_np as NP
    from cnet2_agent import load_weights
    from fastsim import FastCarbon
    ours, lab_w, lab_slot, opp, seed, side = task
    random.seed(eval_v53.rng_seed(seed, side))
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    fa = eval_v53.load_agent_fn(os.path.join(HERE, ours)); fb = eval_v53.load_agent_fn(os.path.join(HERE, opp))
    lab = NP.Policy(load_weights(lab_w))
    feat = F.Feat(side)
    n = dis = n_none = dis_none = 0; ph = {0: [0, 0], 1: [0, 0], 2: [0, 0]}; ctx = []
    for step in range(1, 301):
        o_me = sim.obs(side)
        x = feat.observe(o_me)
        a_me = fa(o_me, eval_v53.CFG)
        a_op = fb(sim.obs(1 - side), eval_v53.CFG)
        y = feat.labels(a_me)
        _, lg, _ = lab.act(x, feat.aux, teacher=lab_slot, forced=y)
        players = o_me["players"]
        enemy = [int(w[0]) for w in players[1 - side][2].values()]
        mytrees = [int(v[0]) for v in players[side][3].values()] if isinstance(players[side][3], dict) else []
        for s in range(10):
            if not x["tok_mask"][s] or int(x["tok_kind"][s]) != 1 or y[s] < 0:
                continue
            t = int(lg[s].argmax()); d = t != int(y[s])
            n += 1; dis += d
            if int(y[s]) == 0 or t == 0:
                n_none += 1; dis_none += d
            p = 0 if step <= 60 else (1 if step <= 150 else 2)
            ph[p][0] += 1; ph[p][1] += d
            if d:
                uid = feat.own_uid[s]
                pos = int(players[side][2][uid][0]) if uid in players[side][2] else None
                if pos is not None:
                    de = min([F.DIST[pos, e] for e in enemy], default=99)
                    tr = sum(1 for c in mytrees if F.DIST[pos, c] <= 2)
                    ctx.append((int(de), tr, int(y[s]), t))
        sim.step(*((a_me, a_op) if side == 0 else (a_op, a_me)))
    st = sim.get_state()
    return {"seed": seed, "side": side, "margin": float(st["cash"][side] - st["cash"][1 - side]), "n": n, "dis": dis,
            "n_none": n_none, "dis_none": dis_none, "phase": ph, "ctx": ctx}


def main():
    ours, lab_w, lab_slot, opp, s0, n = sys.argv[1:7]
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    tasks = [(ours, lab_w, int(lab_slot), opp, s, s % 2) for s in range(int(s0), int(s0) + int(n))]
    out = open(os.path.join(HERE, "..", "data", "dagger_probe.jsonl"), "a")
    R = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(play, tasks):
            R.append(r); out.write(json.dumps({k: v for k, v in r.items() if k != "ctx"}) + "\n"); out.flush()
    N = sum(r["n"] for r in R); D = sum(r["dis"] for r in R)
    NN = sum(r["n_none"] for r in R); DN = sum(r["dis_none"] for r in R)
    print(f"games {len(R)} planter decisions {N} | disagreement {100*D/max(1,N):.1f}% | on NONE-involving decisions {100*DN/max(1,NN):.1f}% (n {NN})")
    for p, nm in ((0, "steps 1-60"), (1, "61-150"), (2, "151-300")):
        a = sum(r["phase"][str(p)][0] if str(p) in r["phase"] else r["phase"][p][0] for r in R)
        b = sum(r["phase"][str(p)][1] if str(p) in r["phase"] else r["phase"][p][1] for r in R)
        print(f"  {nm}: {100*b/max(1,a):.1f}% of {a}")
    C = [c for r in R for c in r["ctx"]]
    if C:
        near = sum(1 for c in C if c[0] <= 2); lone = sum(1 for c in C if c[1] == 0)
        stay_lab = sum(1 for c in C if c[3] == 0); stay_us = sum(1 for c in C if c[2] == 0)
        print(f"  disagreements: enemy within 2 cells {100*near/len(C):.0f}% | no own tree within 2 {100*lone/len(C):.0f}% | "
              f"labeler says stay {100*stay_lab/len(C):.0f}% vs we stayed {100*stay_us/len(C):.0f}%")


if __name__ == "__main__":
    main()
