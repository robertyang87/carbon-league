#!/usr/bin/env python3
"""Probe: read the search pack's in-process telemetry (agent.last_stats)
after one full game, to pin the budget_gate counter field name and scale.

Replicates certify_pack.child_main's call pattern (pack exec + opp load +
FastCarbon game loop), but prints the full _AG.stats dict at game end
instead of only *_errors keys. One game, seed 986500, pack side 0.
Timing is NOT measured here; only the counter fields and values.
"""
import json, sys, time

sys.path.insert(0, "/root/carbon/code")
import eval_v53
from fastsim import FastCarbon

cfg = eval_v53.CFG
seed = 986500

g = {"__name__": "__main__", "__builtins__": __builtins__}
with open("/root/carbon/code/packs/echo4_search_int8.py", "rb") as f:
    pack_raw = f.read()
t0 = time.perf_counter()
exec(compile(pack_raw, "<submission>", "exec"), g)
print("pack exec_ms:", round((time.perf_counter() - t0) * 1000, 1), flush=True)

# opponent: Master2 pack (opp_master2r.py) — load like certify's
# _load_agent_from_bytes: exec file, take make_agent() if present.
og = {"__name__": "opp", "__builtins__": __builtins__,
      "__file__": "/root/carbon/code/exp/opp_master2r.py"}
with open("/root/carbon/code/exp/opp_master2r.py", "rb") as f:
    opp_raw = f.read()
exec(compile(opp_raw, "<opp>", "exec"), og)
opp_fn = og["make_agent"]() if "make_agent" in og else og["agent"]

sim = FastCarbon(cfg).reset(seed=seed)
side = seed % 2
steps = 0
t1 = time.perf_counter()
for i in range(299):
    o = [sim.obs(0), sim.obs(1)]
    a = g["agent"](o[side], cfg)
    b = opp_fn(o[1 - side], cfg)
    sim.step(*((a, b) if side == 0 else (b, a)))
    steps += 1
print("steps:", steps, "wall_s:", round(time.perf_counter() - t1, 1), flush=True)
st = getattr(g["agent"], "last_stats", None)
print("last_stats type:", type(st).__name__, flush=True)
print(json.dumps(st, default=str, ensure_ascii=False, indent=1), flush=True)
