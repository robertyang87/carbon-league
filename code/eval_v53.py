"""eval_v53: paired-seed strength verification of the two public v53
sources (reference_claude_opus_v53.py, reference_gloriousevolution_v53.py)
against juzi, our v0.5, each other, and example.

Methodology = eval_v0_8 (same FastCarbon, same seed = same map, both
sides, same per-player ledger reconstructed from state diffs; see
eval_v0_8.py docstring for ledger definitions). Differences:
  * BOTH players are arbitrary files; loader rule (shared verbatim with
    record_v53_traces.py): files whose stem starts with "policy_" load
    via make_agent(), everything else via the module-level agent().
  * Both players timed per step.
  * RNG PROTOCOL: reference v53 draws from the global `random` module
    (hunt-plan roulette, line ~1354). To keep games reproducible for
    the trace re-run check, play_game seeds
        random.seed(7919*seed + 104729*policy_side + 31)
    as its first statement, mirroring record_trace.record_game's call
    order (reset -> load A -> load B -> loop). record_v53_traces.py
    applies the identical seeding before delegating.
  * Composition snapshots (collectors/planters/trees per player) at
    steps 100/200/300 for the ledger comparison.

Jobs (20 games):
  v53juzi  v53  vs reference_juzi   seeds 1000-1003 x2 sides
  v53v05   v53  vs policy_v0_5      seeds 1000-1001 x2
  gev53    GE   vs v53              seeds 1000-1001 x2
  v53ex    v53  vs example          seed  2000      x2
  geex     GE   vs example          seed  2000      x2
Appends one JSON line per game to eval_v53_results.jsonl.
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

from fastsim import FastCarbon  # noqa: E402

_spec = json.load(open(os.path.join(
    HERE, "env_src", "carbon_challenge-master",
    "zerosum_env", "envs", "carbon", "carbon.json")))
CFG = {}
for _k, _v in _spec["configuration"].items():
    CFG[_k] = _v.get("default") if isinstance(_v, dict) else _v

V53 = "reference_claude_opus_v53.py"
GE = "reference_gloriousevolution_v53.py"

JOBS = []
for _seed in (1000, 1001, 1002, 1003):
    for _side in (0, 1):
        JOBS.append(("v53juzi", V53, "reference_juzi.py", _seed, _side))
for _seed in (1000, 1001):
    for _side in (0, 1):
        JOBS.append(("v53v05", V53, "policy_v0_5.py", _seed, _side))
        JOBS.append(("gev53", GE, V53, _seed, _side))
for _side in (0, 1):
    JOBS.append(("v53ex", V53, "example.py", 2000, _side))
    JOBS.append(("geex", GE, "example.py", 2000, _side))

_counter = [0]


def load_agent_fn(path):
    """Shared loader rule: our policy_*.py via make_agent() (fresh
    instance), everything else via module-level agent()."""
    _counter[0] += 1
    name = f"evalv53_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if os.path.basename(path).startswith("policy_") and hasattr(m, "make_agent"):
        return m.make_agent()
    return m.agent


def rng_seed(seed, policy_side):
    return 7919 * seed + 104729 * policy_side + 31


def _plant_price(pre_trees, player):
    mine = sum(1 for t in pre_trees.values() if t["player"] == player)
    if mine <= CFG["playerTreeProtectiveNumber"]:
        return float(CFG["plantCost"])
    alive = sum(1 for t in pre_trees.values() if t["age"] < CFG["treeLifespan"])
    return (CFG["plantCost"] + CFG["plantCostInflationRatio"]
            * (CFG["plantCostInflationBase"] ** alive))


def new_ledger():
    return dict(deposits=0.0, tree_income=0.0,
                recruit_spend=0.0, recruit_count=0,
                plant_spend=0.0, plant_count=0,
                seize_spend=0.0, seize_count=0,
                deaths=0, death_cargo=0.0, stranded=0.0)


def _comp(st, p):
    c = sum(1 for w in st["workers"].values()
            if w["player"] == p and w["type"] == "COLLECTOR")
    pl = sum(1 for w in st["workers"].values()
             if w["player"] == p and w["type"] == "PLANTER")
    tr = sum(1 for t in st["trees"].values() if t["player"] == p)
    return [c, pl, tr]


def play_game(task):
    tag, a_file, b_file, seed, a_side = task
    random.seed(rng_seed(seed, a_side))
    sim = FastCarbon(CFG).reset(seed=seed)
    fa = load_agent_fn(os.path.join(HERE, a_file))
    fb = load_agent_fn(os.path.join(HERE, b_file))
    agents = [None, None]
    agents[a_side] = fa
    agents[1 - a_side] = fb
    led = [new_ledger(), new_ledger()]
    times = [0.0, 0.0]
    worst_a = 0.0
    comp = {}
    t0 = time.time()
    bk_steps = 0
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
        # bankruptcy: policy side has no workers and cannot afford a collector (after the opening)
        if step >= 30 and post["cash"][a_side] < float(CFG["recCollectorCost"]) and \
                not any(w["player"] == a_side for w in post["workers"].values()):
            bk_steps += 1
        if step in (100, 200, 300):
            comp[step] = [_comp(post, 0), _comp(post, 1)]
        for p in range(2):
            L = led[p]
            price = _plant_price(pre["trees"], p)
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
        "wall_s": round(wall, 1), "bankrupt_steps": bk_steps,
        "ledger_pol": led[a_side], "ledger_opp": led[1 - a_side],
        "comp": {str(k): v for k, v in comp.items()},
    }


def main():
    out_path = os.path.join(HERE, "eval_v53_results.jsonl")
    out = open(out_path, "a")
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(play_game, t): t for t in JOBS}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  f"a_ms={res['a_ms_per_step']} b_ms={res['b_ms_per_step']}",
                  flush=True)
    out.close()


if __name__ == "__main__":
    main()
