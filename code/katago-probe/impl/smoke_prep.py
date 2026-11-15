#!/usr/bin/env python3
"""SMOKE ONLY -- synthetic data, strictly separate from the registered
rootset32. Generates a FastCarbon game (seed 777) driven by two stock
SEARCH=False agents, captures observations at steps [200,232) x both
seats as driver tasks labelled SYNTH-SEED777, and unit-checks that the
variant leaf's z agrees between the module's batched path (encode_b)
and NP.Policy's own encode (CachePolicy.enc_for).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "modules"))
import numpy as np  # noqa: E402
import cnet2_feat as F  # noqa: E402
import cnet2_search as base  # noqa: E402
import cnet2_search_vleaf as vleaf  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

A = os.path.join(HERE, "smoke_assets")
W = os.path.join(A, "cnet2_m3s2r1L.npz")
RW = os.path.join(A, "cnet2_s64.npz")


def load(p):
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


wd = load(W)
rwd = load(RW)

sim = FastCarbon()
sim.reset(seed=777)
agents = [base.make_search_agent(conf={"SEARCH": False}, weights=wd,
                                 rweights=rwd) for _ in range(2)]
tasks = []
for step in range(232):
    obs = [sim.obs(0), sim.obs(1)]
    if step >= 200:
        for seat in (0, 1):
            tasks.append({"game_key": "SYNTH-SEED777", "seat": seat,
                          "step": step, "obs": obs[seat]})
    a0 = agents[0](obs[0])
    a1 = agents[1](obs[1])
    sim.step(a0, a1)

out = os.path.join(HERE, "smoke", "tasks.jsonl")
with open(out, "w") as f:
    for t in tasks:
        f.write(json.dumps(t) + "\n")
print(f"synthetic tasks: {len(tasks)} -> {out}")

# ---- unit check: z via encode_b (variant leaf path) vs enc_for (NP path)
obs0 = tasks[0]["obs"]
fx = F.Feat(0)
x = fx.observe(obs0)
pol = vleaf.CachePolicy(wd)
_, ctx1 = pol.enc_for(x, 1)
ctx2 = vleaf.encode_b(pol, vleaf.stack_x([x]), np.array([1]))[1][0]
z1 = float(ctx1 @ pol.w["value.weightT"][:, 0] + pol.w["value.bias"][0])
z2 = float(ctx2 @ pol.w["value.weightT"][:, 0] + pol.w["value.bias"][0])
print(f"z enc_for={z1:.8f}  z encode_b={z2:.8f}  |diff|={abs(z1 - z2):.3e}")
print(f"M_eq={2000.0 * z2:.2f}  u_hat={(np.tanh(z2) + 1) / 2:.4f}")
assert abs(z1 - z2) < 1e-4, "encode paths disagree"
print("UNIT_CHECK_OK")
