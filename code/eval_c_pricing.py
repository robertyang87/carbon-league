"""eval_c_pricing: acceptance evaluation for structure case C step 2
(safety-layer pricing forks), following avoid_tax_report.md section 7
and the 2026-11-03 calibration (shadow tier downweighted).

Design (paired, same seed, both sides; harness identical to eval_v53):
  cparity  fork_c_off     vs shadow_alpha_v2  seed 1000 x2
           -> must reproduce the base/probe games 14255.07:2.43 and
              15796.46:1.30 exactly (zero-pollution gate for the edits)
  cbase    base_v53       vs shadow_alpha_v2  seeds 1001-1003 x2
           (seed 1000 controls already in eval_probe_results.jsonl)
  call     fork_c_pricing vs shadow_alpha_v2  seeds 1000-1003 x2  (main)
  csoft    fork_c_soft    vs shadow_alpha_v2  seeds 1000-1001 x2  (abl.)
  clethal  fork_c_lethal  vs shadow_alpha_v2  seeds 1000-1001 x2  (abl.)
  callv05  fork_c_pricing vs policy_v0_5      seeds 1000-1001 x2
  cmirror  fork_c_pricing vs base_v53         seeds 1000-1001 x2

Fork games also snapshot the Bot's pricing tables (pb1/pb1m/pbt/pb2/
pb2m/pb_lvl) into the result line for the accuracy (准) audit.
Appends one JSON line per game to eval_c_pricing_results.jsonl.
Concurrency is controlled by the caller (max_workers argument).
"""

import importlib.util
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import eval_v53  # noqa: E402  (CFG, rng_seed, ledger helpers)
from fastsim import FastCarbon  # noqa: E402

CFG = eval_v53.CFG
SHADOW = "shadow_alpha_v2.py"
BASE = "base_v53.py"

JOBS = []
for _side in (0, 1):
    JOBS.append(("cparity", "fork_c_off.py", SHADOW, 1000, _side))
for _seed in (1001, 1002, 1003):
    for _side in (0, 1):
        JOBS.append(("cbase", BASE, SHADOW, _seed, _side))
for _seed in (1000, 1001, 1002, 1003):
    for _side in (0, 1):
        JOBS.append(("call", "fork_c_pricing.py", SHADOW, _seed, _side))
for _seed in (1000, 1001):
    for _side in (0, 1):
        JOBS.append(("csoft", "fork_c_soft.py", SHADOW, _seed, _side))
        JOBS.append(("clethal", "fork_c_lethal.py", SHADOW, _seed, _side))
        JOBS.append(("callv05", "fork_c_pricing.py", "policy_v0_5.py",
                     _seed, _side))
        JOBS.append(("cmirror", "fork_c_pricing.py", BASE, _seed, _side))

_counter = [0]


def load_agent_mod(path):
    """eval_v53 loader rule, but also returns the module (for _BOT)."""
    _counter[0] += 1
    name = f"evalc_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if os.path.basename(path).startswith("policy_") and hasattr(m, "make_agent"):
        return m.make_agent(), None
    return m.agent, m


def _snap(bot):
    if bot is None or not hasattr(bot, "pb1"):
        return None
    def conv(tab):
        return {",".join(str(x) for x in (k if isinstance(k, tuple)
                                         else (k,))): v
                for k, v in tab.items()}
    return {
        "pb1": conv(bot.pb1), "pb1m": conv(bot.pb1m),
        "pbt": conv(bot.pbt), "pb2": conv(bot.pb2),
        "pb2m": conv(bot.pb2m), "pb_lvl": list(bot.pb_lvl),
    }


def play_game(task):
    tag, a_file, b_file, seed, a_side = task
    random.seed(eval_v53.rng_seed(seed, a_side))
    sim = FastCarbon(CFG).reset(seed=seed)
    fa, ma = load_agent_mod(os.path.join(HERE, a_file))
    fb, mb = load_agent_mod(os.path.join(HERE, b_file))
    agents = [None, None]
    agents[a_side] = fa
    agents[1 - a_side] = fb
    led = [eval_v53.new_ledger(), eval_v53.new_ledger()]
    times = [0.0, 0.0]
    worst_a = 0.0
    comp = {}
    t0 = time.time()
    for step in range(1, 301):
        pre = sim.get_state()
        acts = []
        for p in range(2):
            o = sim.obs(p)
            ts = time.time()
            acts.append(agents[p](o, CFG))
            dt = time.time() - ts
            times[p] += dt
            if p == a_side:
                worst_a = max(worst_a, dt)
        sim.step(acts[0], acts[1])
        post = sim.get_state()
        if step in (100, 200, 300):
            comp[step] = [eval_v53._comp(post, 0), eval_v53._comp(post, 1)]
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
    wall = time.time() - t0
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
        "win": st["cash"][a_side] > st["cash"][1 - a_side],
        "a_ms_per_step": round(1000.0 * times[a_side] / 300, 2),
        "b_ms_per_step": round(1000.0 * times[1 - a_side] / 300, 2),
        "a_worst_step_ms": round(1000.0 * worst_a, 0),
        "wall_s": round(wall, 1),
        "ledger_pol": led[a_side], "ledger_opp": led[1 - a_side],
        "comp": {str(k): v for k, v in comp.items()},
        "pb": _snap(getattr(ma, "_BOT", None) if ma else None),
    }


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j[0] == only]
    out_path = os.path.join(HERE, "eval_c_pricing_results.jsonl")
    out = open(out_path, "a")
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(play_game, t): t for t in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  f"a_ms={res['a_ms_per_step']}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
