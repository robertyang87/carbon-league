#!/usr/bin/env python3
"""A/B runner: PACK vs baseline echo4_int8 (SEARCH=False), FastCarbon
self-play, pack seat alternating per game, fixed seeds. Records every
action of both seats (canonical JSON), final cash, and the pack's
per-game comparator telemetry snapshot. One pack per process invocation.

Usage: python3 ab_runner.py PACK.py GAMES SEED0 OUT.json [SEAT0]
SEAT0 (default 0): pack seat of the first game; alternates per game.
"""
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/workspace/search-group/proto/vendor"))
from fastsim import FastCarbon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(os.path.dirname(HERE), "ref_echo4_int8.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def canon(a):
    try:
        return json.dumps(a, sort_keys=True)
    except Exception:
        return repr(a)


def main():
    pack_path, games, seed0, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    seat0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    pack = load("ab_pack", pack_path)
    base = load("ab_base", BASELINE)
    result = {"pack": pack_path, "pack_conf": {k: pack.CONF.get(k) for k in ("SEARCH", "ROLL", "MODE", "BUDGET", "CMP_H")},
              "baseline": BASELINE, "games": []}
    for g in range(games):
        seed = seed0 + g
        pseat = (seat0 + g) % 2
        sim = FastCarbon()
        sim.reset(seed=seed)
        agents = [None, None]
        agents[pseat] = pack.agent
        agents[1 - pseat] = base.agent
        acts = []
        t0 = time.time()
        stalled = 0
        while sim.step_count < 300:
            if sim.step_count % 50 == 0:
                print(f"  game {g} step {sim.step_count} t={time.time() - t0:.0f}s", flush=True)
            prev = sim.step_count
            raw_acts = []
            step_acts = []
            for seat in (0, 1):
                a = agents[seat](sim.obs(seat), None)
                raw_acts.append(a)
                step_acts.append(canon(a))
            sim.step(raw_acts[0], raw_acts[1])
            acts.append(step_acts)
            if sim.step_count == prev:
                stalled += 1
                if stalled > 3:
                    break
        snap = pack._AG.cmp_stats.snapshot()
        result["games"].append({
            "seed": seed, "pack_seat": pseat, "steps": sim.step_count,
            "completed": sim.step_count >= 300, "cash": [float(c) for c in sim.cash],
            "actions": acts, "cmp": snap,
            "wall_s": round(time.time() - t0, 2)})
        print(f"game {g} seed={seed} pseat={pseat} steps={sim.step_count} "
              f"cmp_calls={snap['calls']} completed={snap['completed_comparisons']} "
              f"changed={snap['changed_count']} leaf={snap['leaf_calls']} "
              f"depth={snap.get('depth_hist')} wall={result['games'][-1]['wall_s']}s", flush=True)
        json.dump(result, open(out_path, "w"))  # incremental: survive a kill mid-run
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
