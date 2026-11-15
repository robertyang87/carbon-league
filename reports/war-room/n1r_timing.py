#!/usr/bin/env python3
"""n1r ladder-timing probe: ONE game vs echo_e0, replicated from
eval_v53.play_game's loop (same seeding, same FastCarbon, same CFG),
recording wall time of every n1r agent() call. In-process timing only."""
import json, os, random, statistics as st, sys, time

CODE = "/root/carbon/code"
sys.path.insert(0, CODE)
os.chdir(CODE)

SEED = 700001
A_SIDE = SEED % 2  # bigeval convention: seat alternates by seed parity

t_start = time.perf_counter()
import eval_v53  # noqa: E402  (provides CFG, FastCarbon, load_agent_fn, rng_seed, NSTEPS)
CFG = eval_v53.CFG

res = {"seed": SEED, "a_side": A_SIDE, "nsteps_env": eval_v53.NSTEPS}

# --- init part 1: module import (load_agent_fn execs the pack module) ---
t0 = time.perf_counter()
fa = eval_v53.load_agent_fn(os.path.join(CODE, "..", "data", "packs", "n1r_int8.py"))
res["import_ms"] = round((time.perf_counter() - t0) * 1000, 1)
fb = eval_v53.load_agent_fn(os.path.join(CODE, "exp", "echo_e0.py"))

# --- game, exact play_game replication (minus ledgers, which touch neither sim nor agents) ---
random.seed(eval_v53.rng_seed(SEED, A_SIDE))
sim = eval_v53.FastCarbon(CFG).reset(seed=SEED)
agents = [None, None]
agents[A_SIDE] = fa
agents[1 - A_SIDE] = fb

step_ms = []
err = None
t_game0 = time.perf_counter()
try:
    for step in range(1, eval_v53.NSTEPS + 1):
        acts = []
        for p in range(2):
            o = sim.obs(p)
            ts = time.perf_counter()
            a = agents[p](o, CFG)
            dt = (time.perf_counter() - ts) * 1000
            acts.append(a)
            if p == A_SIDE:
                step_ms.append(dt)
        sim.step(acts[0], acts[1])
except Exception as e:
    err = repr(e)[:300]
res["game_wall_s"] = round(time.perf_counter() - t_game0, 2)
res["error"] = err

if step_ms:
    srt = sorted(step_ms)
    n = len(srt)
    p95 = srt[min(n - 1, int(0.95 * n))]
    res.update({
        "n_steps": n,
        "first_call_ms": round(step_ms[0], 1),          # includes lazy dequantize + agent construction + 1st decision
        "init_total_ms": round(res["import_ms"] + step_ms[0], 1),
        "p50_ms": round(st.median(step_ms), 1),
        "p95_ms": round(p95, 1),
        "max_ms": round(srt[-1], 1),
        "slow_steps_gt1000": sum(1 for x in step_ms if x > 1000),
        "extra_ms": round(sum(max(0.0, x - 1000.0) for x in step_ms), 1),
        "sum_steps_s": round(sum(step_ms) / 1000, 2),
    })
try:
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmHWM"):
                res["peak_rss_kb"] = int(ln.split()[1])
except Exception:
    pass
print("N1RTIME_JSON " + json.dumps(res))
