"""one 100-step game dg5 (side 0) vs a wrapper exp (side 1); print the wrapper side's composition every 10 steps"""
import sys, os, json
sys.path.insert(0, ".")
import eval_v53 as E
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 996001
sim = E.FastCarbon(E.CFG).reset(seed=seed)
A = [E.load_agent_fn("exp/cnet2_dg5e5s57.py"), E.load_agent_fn(sys.argv[1])]
for step in range(1, 101):
    acts = [A[p](sim.obs(p), E.CFG) for p in range(2)]
    sim.step(acts[0], acts[1])
    if step % 10 == 0:
        s = sim.get_state()
        row = []
        for p in range(2):
            W = [w for w in s["workers"].values() if w["player"] == p]
            row.append(f"p{p} cash {sim.cash[p]:6.0f} col {sum(w['type']=='COLLECTOR' for w in W)} pl {sum(w['type']=='PLANTER' for w in W)} tr {sum(t['player']==p for t in s['trees'].values())}")
        print(step, " | ".join(row))
