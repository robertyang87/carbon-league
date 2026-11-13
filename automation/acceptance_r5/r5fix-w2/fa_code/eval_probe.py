"""eval_probe: zero-pollution parity check + avoidance-tax measurement
for probe_v53.py (structure roadmap case C, step 1).

Phases (argv[1]):
  parity  base_v53 vs shadow_alpha_v2 and probe_v53 vs shadow_alpha_v2,
          seed 1000, sides 0/1. Probe and base runs must produce IDENTICAL
          per-step action dicts (both players) and identical final cash.
  measure probe_v53 (file A) vs shadow_alpha_v2 / policy_v0_5 / example,
          seed 1000, sides 0/1 (2 paired games each). Per game the probe
          dumps its instrumentation stats to probe_stats/<tag>_s<side>.json
          (path passed via PROBE_STATS_PATH inside the worker). Results
          append to eval_probe_results.jsonl with the same ledger
          reconstruction as eval_v53.

RNG protocol identical to eval_v53: random.seed(7919*seed + 104729*side
+ 31) before reset -> load A -> load B -> loop.
Concurrency: max_workers=2 (fleet-wide compute discipline, 2026-11-03).
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import eval_v53  # noqa: E402  (CFG, load_agent_fn, rng_seed, ledger helpers)
from fastsim import FastCarbon  # noqa: E402

CFG = eval_v53.CFG
BASE = "base_v53.py"
PROBE = "probe_v53.py"
STATS_DIR = os.path.join(HERE, "probe_stats")

MEASURE_JOBS = []
for _tag, _opp in (("pbshadow", "shadow_alpha_v2.py"),
                   ("pbv05", "policy_v0_5.py"),
                   ("pbex", "example.py")):
    for _side in (0, 1):
        MEASURE_JOBS.append((_tag, PROBE, _opp, 1000, _side))

PARITY_JOBS = []
for _a in (BASE, PROBE):
    for _side in (0, 1):
        PARITY_JOBS.append(("parity", _a, "shadow_alpha_v2.py", 1000, _side))


def play_game(task):
    tag, a_file, b_file, seed, a_side = task
    random.seed(eval_v53.rng_seed(seed, a_side))
    stats_path = None
    if a_file == PROBE and tag != "parity":
        os.makedirs(STATS_DIR, exist_ok=True)
        stats_path = os.path.join(STATS_DIR, f"{tag}_seed{seed}_side{a_side}.json")
        os.environ["PROBE_STATS_PATH"] = stats_path
    else:
        os.environ.pop("PROBE_STATS_PATH", None)
    sim = FastCarbon(CFG).reset(seed=seed)
    fa = eval_v53.load_agent_fn(os.path.join(HERE, a_file))
    fb = eval_v53.load_agent_fn(os.path.join(HERE, b_file))
    agents = [None, None]
    agents[a_side] = fa
    agents[1 - a_side] = fb
    led = [eval_v53.new_ledger(), eval_v53.new_ledger()]
    times = [0.0, 0.0]
    acts_log = []
    for step in range(1, 301):
        pre = sim.get_state()
        acts = []
        for p in range(2):
            o = sim.obs(p)
            ts = time.time()
            acts.append(agents[p](o, CFG))
            times[p] += time.time() - ts
        acts_log.append([{k: v for k, v in (a or {}).items() if v is not None}
                         for a in acts])
        sim.step(acts[0], acts[1])
        post = sim.get_state()
        for p in range(2):
            L = led[p]
            price = eval_v53._plant_price(pre["trees"], p)
            rs = ps = ss = 0.0
            for tid, t in post["trees"].items():
                if t["player"] != p:
                    continue
                L["tree_income"] += float(
                    post["trees_dict"].get(tid, [None, 0])[1])
                if tid not in pre["trees"]:
                    L["plant_count"] += 1
                    ps += price
                elif pre["trees"][tid]["player"] != p:
                    L["seize_count"] += 1
                    ss += float(CFG["seizeCost"])
            for wid, w in post["workers"].items():
                if w["player"] == p and wid not in pre["workers"]:
                    L["recruit_count"] += 1
                    rs += float(CFG["recCollectorCost"])
            for wid, w in pre["workers"].items():
                if w["player"] == p and wid not in post["workers"]:
                    L["deaths"] += 1
                    L["death_cargo"] += float(w["carbon"])
            L["recruit_spend"] += rs
            L["plant_spend"] += ps
            L["seize_spend"] += ss
            L["deposits"] += (post["cash"][p] - pre["cash"][p]) + rs + ps + ss \
                - float(sum(post["trees_dict"].get(tid, [None, 0])[1]
                            for tid, t in post["trees"].items()
                            if t["player"] == p))
    st = sim.get_state()
    for p in range(2):
        led[p]["stranded"] = round(sum(
            w["carbon"] for w in st["workers"].values() if w["player"] == p), 1)
        for k in ("deposits", "tree_income", "recruit_spend", "plant_spend",
                  "seize_spend", "death_cargo"):
            led[p][k] = round(led[p][k], 1)
    return {
        "tag": tag, "seed": seed, "policy_side": a_side,
        "a_file": a_file, "b_file": b_file,
        "my_cash": round(st["cash"][a_side], 2),
        "opp_cash": round(st["cash"][1 - a_side], 2),
        "margin": round(st["cash"][a_side] - st["cash"][1 - a_side], 2),
        "a_ms_per_step": round(1000.0 * times[a_side] / 300, 2),
        "b_ms_per_step": round(1000.0 * times[1 - a_side] / 300, 2),
        "ledger_pol": led[a_side], "ledger_opp": led[1 - a_side],
        "stats_path": stats_path,
        "acts": acts_log,
    }


def run_jobs(jobs):
    results = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(play_game, t): t for t in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results[(r["tag"], r["a_file"], r["seed"], r["policy_side"])] = r
            print(r["tag"], r["a_file"], r["seed"], r["policy_side"],
                  r["my_cash"], r["opp_cash"], r["margin"],
                  f"a_ms={r['a_ms_per_step']}", flush=True)
    return results


def phase_parity():
    res = run_jobs(PARITY_JOBS)
    ok_all = True
    for side in (0, 1):
        rb = res[("parity", BASE, 1000, side)]
        rp = res[("parity", PROBE, 1000, side)]
        same_acts = rb["acts"] == rp["acts"]
        same_cash = (rb["my_cash"] == rp["my_cash"]
                     and rb["opp_cash"] == rp["opp_cash"])
        ndiff = sum(1 for x, y in zip(rb["acts"], rp["acts"]) if x != y)
        print(f"side{side}: acts_identical={same_acts} "
              f"steps_differing={ndiff} cash_identical={same_cash} "
              f"base {rb['my_cash']}:{rb['opp_cash']} "
              f"probe {rp['my_cash']}:{rp['opp_cash']}", flush=True)
        ok_all = ok_all and same_acts and same_cash
    print("PARITY:", "PASS" if ok_all else "FAIL", flush=True)
    return 0 if ok_all else 2


def phase_measure():
    out = open(os.path.join(HERE, "eval_probe_results.jsonl"), "a")
    res = run_jobs(MEASURE_JOBS)
    for key in sorted(res):
        r = dict(res[key])
        r.pop("acts", None)
        out.write(json.dumps(r) + "\n")
        out.flush()
        sp = r.get("stats_path")
        print("stats:", sp, "exists:", bool(sp and os.path.exists(sp)),
              flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "parity"
    sys.exit(phase_parity() if phase == "parity" else phase_measure())
