#!/usr/bin/env python3
"""bench_distilled -- per-step latency of the distilled policies,
measured EXCLUSIVELY: waits for ~/workspace/carbon-league/.eval_lock,
holds it during the measurement, releases after.  One game per
variant vs base_v53 (seed 3000), timing every agent call with
perf_counter; base_v53's own calls are timed in the same games as
the reference line.  Site budget: 1000 ms/step average, 3 s max.
"""

import importlib.util
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import record_trace  # noqa: E402

LOCK = os.path.join(HERE, ".eval_lock")
_counter = [0]


def load_agent_fresh(path):
    _counter[0] += 1
    name = f"bench_{_counter[0]}_{os.getpid()}_" + os.path.splitext(
        os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_agent"):
        return mod.make_agent()
    return mod.agent


def acquire_lock(timeout=1200):
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            return True
        except FileExistsError:
            if time.time() - t0 > timeout:
                return False
            time.sleep(5)


def timed_game(vfile, seed=3000, side=0):
    random.seed(7919 * seed + 104729 * side + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=seed)
    policy = load_agent_fresh(os.path.join(HERE, vfile))
    opp = load_agent_fresh(os.path.join(HERE, "base_v53.py"))
    agents = [None, None]
    agents[side] = policy
    agents[1 - side] = opp
    times = {0: [], 1: []}
    for _k in range(300):
        acts = []
        for p in range(2):
            o = sim.obs(p)
            t0 = time.perf_counter()
            acts.append(agents[p](o, record_trace.CFG))
            times[p].append((time.perf_counter() - t0) * 1000.0)
        sim.step(acts[0], acts[1])
    return times[side], times[1 - side]


def stats(xs):
    a = sorted(xs)
    n = len(a)
    return {"mean_ms": round(sum(a) / n, 2),
            "p50_ms": round(a[n // 2], 2),
            "p95_ms": round(a[int(n * 0.95)], 2),
            "max_ms": round(a[-1], 2)}


def main():
    if not acquire_lock():
        print("could not acquire .eval_lock in 20 min; aborting")
        return 1
    try:
        out = {}
        for variant, vfile in (("pure", "distilled_v1.py"),
                               ("hybrid", "distilled_v1_hybrid.py")):
            mine, base = timed_game(vfile)
            out[variant] = {"distilled": stats(mine),
                            "base_v53_same_game": stats(base)}
            print(variant, json.dumps(out[variant]))
        with open(os.path.join(HERE, "distill_bench.json"), "w") as f:
            json.dump(out, f, indent=1)
        print("wrote distill_bench.json")
        return 0
    finally:
        os.rmdir(LOCK)


if __name__ == "__main__":
    sys.exit(main())
