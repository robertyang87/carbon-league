"""eval_s2: paired-seed eval for the S2 posture line.

Same FastCarbon, same RNG protocol and same per-player ledger as
eval_v53 (numbers comparable with eval_v53_results.jsonl / BASELINE.md),
plus what S2 needs:
  * composition + cash snapshots every 25 steps (switcher behaviour
    self-checks; endgame-phase analysis)
  * phase-split ledgers: deposits in steps <=150 vs >150, plant counts
    in windows [0-50] [51-100] [101-150] [151-300]
  * posture trace capture: for fork_s2_* players, the finished Bot's
    post_trace (checkpoints at obs-step 60/100 + switch events) is
    read back from the loaded module's _BOT and stored in the result.

Phases (argv[1]); every game appends one JSON line to
eval_s2_results.jsonl. Run under .eval_lock (caller wraps), with
EVAL_WORKERS<=2.
  selfcheck switcher_{harvest,sprint,treewar} vs base_v53,
            seeds 1000-1001 x2 sides (12 games; doubles as the
            base-vs-switcher "no posture layer" validation rows)
  posture   fork_s2_posture vs each switcher (12) + vs base_v53 (4)
  control   fork_s2_off vs base_v53 (4) -- classifier-off control
  abl_tree  fork_s2_no_tree vs switcher_treewar (4)
  abl_harv  fork_s2_no_harv vs switcher_harvest (4)
            + vs switcher_sprint (4)
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

import eval_v53  # noqa: E402  (CFG, FastCarbon setup, ledger helpers)

BASE = "base_v53.py"
POST = "fork_s2_posture.py"
SEEDS = (1000, 1001)
_counter = [0]


def load_module(path):
    _counter[0] += 1
    name = f"evals2_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _trace_of(mod):
    bot = getattr(mod, "_BOT", None)
    tr = getattr(bot, "post_trace", None)
    return tr if tr else None


def play_game_s2(task):
    tag, a_file, b_file, seed, a_side = task
    random.seed(eval_v53.rng_seed(seed, a_side))
    sim = eval_v53.FastCarbon(eval_v53.CFG).reset(seed=seed)
    mods = [None, None]
    ma = load_module(os.path.join(HERE, a_file))
    mb = load_module(os.path.join(HERE, b_file))
    mods[a_side] = ma
    mods[1 - a_side] = mb
    agents = [mods[0].agent, mods[1].agent]
    led = [eval_v53.new_ledger(), eval_v53.new_ledger()]
    phase = [dict(dep_le150=0.0, dep_gt150=0.0,
                  plant_w1=0, plant_w2=0, plant_w3=0, plant_w4=0)
             for _ in range(2)]
    times = [0.0, 0.0]
    worst_a = 0.0
    series = {}
    t0 = time.time()
    for step in range(1, 301):
        pre = sim.get_state()
        acts = []
        for p in range(2):
            o = sim.obs(p)
            ts = time.time()
            acts.append(agents[p](o, eval_v53.CFG))
            dt = time.time() - ts
            times[p] += dt
            if p == a_side:
                worst_a = max(worst_a, dt)
        sim.step(acts[0], acts[1])
        post = sim.get_state()
        if step % 25 == 0:
            series[step] = [eval_v53._comp(post, 0),
                            eval_v53._comp(post, 1),
                            round(post["cash"][0], 1),
                            round(post["cash"][1], 1)]
        for p in range(2):
            L = led[p]
            price = eval_v53._plant_price(pre["trees"], p)
            rs = ps = ss = 0.0
            nplant = 0
            for tid, t in post["trees"].items():
                if t["player"] != p:
                    continue
                L["tree_income"] += float(
                    post["trees_dict"].get(tid, [None, 0])[1])
                if tid not in pre["trees"]:
                    L["plant_count"] += 1
                    nplant += 1
                    ps += price
                elif pre["trees"][tid]["player"] != p:
                    L["seize_count"] += 1
                    ss += float(eval_v53.CFG["seizeCost"])
            for wid, w in post["workers"].items():
                if w["player"] == p and wid not in pre["workers"]:
                    L["recruit_count"] += 1
                    rs += float(eval_v53.CFG["recCollectorCost"])
            for wid, w in pre["workers"].items():
                if w["player"] == p and wid not in post["workers"]:
                    L["deaths"] += 1
                    L["death_cargo"] += float(w["carbon"])
            L["recruit_spend"] += rs
            L["plant_spend"] += ps
            L["seize_spend"] += ss
            dep = (post["cash"][p] - pre["cash"][p]) + rs + ps + ss \
                - float(sum(post["trees_dict"].get(tid, [None, 0])[1]
                            for tid, t in post["trees"].items()
                            if t["player"] == p))
            L["deposits"] += dep
            ph = phase[p]
            if step <= 150:
                ph["dep_le150"] += dep
            else:
                ph["dep_gt150"] += dep
            if nplant:
                if step <= 50:
                    ph["plant_w1"] += nplant
                elif step <= 100:
                    ph["plant_w2"] += nplant
                elif step <= 150:
                    ph["plant_w3"] += nplant
                else:
                    ph["plant_w4"] += nplant
    wall = time.time() - t0
    st = sim.get_state()
    for p in range(2):
        led[p]["stranded"] = round(sum(
            w["carbon"] for w in st["workers"].values()
            if w["player"] == p), 1)
        for k in ("deposits", "tree_income", "recruit_spend",
                  "plant_spend", "seize_spend", "death_cargo"):
            led[p][k] = round(led[p][k], 1)
        for k in ("dep_le150", "dep_gt150"):
            phase[p][k] = round(phase[p][k], 1)
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
        "phase_pol": phase[a_side], "phase_opp": phase[1 - a_side],
        "series": {str(k): v for k, v in series.items()},
        "trace_pol": _trace_of(mods[a_side]),
        "trace_opp": _trace_of(mods[1 - a_side]),
    }


def jobs_for(phase):
    sw = ["switcher_harvest.py", "switcher_sprint.py", "switcher_treewar.py"]
    if phase == "selfcheck":
        return [(f"chk_{os.path.splitext(f)[0][9:]}", f, BASE, s, side)
                for f in sw for s in SEEDS for side in (0, 1)]
    if phase == "posture":
        jobs = [(f"post_{os.path.splitext(f)[0][9:]}", POST, f, s, side)
                for f in sw for s in SEEDS for side in (0, 1)]
        jobs += [("post_base", POST, BASE, s, side)
                 for s in SEEDS for side in (0, 1)]
        return jobs
    if phase == "control":
        return [("ctrl_base", "fork_s2_off.py", BASE, s, side)
                for s in SEEDS for side in (0, 1)]
    if phase == "abl_tree":
        return [("abl_notree", "fork_s2_no_tree.py", "switcher_treewar.py",
                 s, side) for s in SEEDS for side in (0, 1)]
    if phase == "abl_harv":
        return ([("abl_noharv_h", "fork_s2_no_harv.py",
                  "switcher_harvest.py", s, side)
                 for s in SEEDS for side in (0, 1)]
                + [("abl_noharv_s", "fork_s2_no_harv.py",
                    "switcher_sprint.py", s, side)
                   for s in SEEDS for side in (0, 1)])
    if phase == "smoke":
        return [("smoke", POST, "switcher_treewar.py", 1000, 0)]
    raise SystemExit(f"unknown phase {phase}")


def main():
    phase = sys.argv[1]
    jobs = jobs_for(phase)
    out = open(os.path.join(HERE, "eval_s2_results.jsonl"), "a")
    with ProcessPoolExecutor(
            max_workers=int(os.environ.get("EVAL_WORKERS", "2"))) as ex:
        futs = {ex.submit(play_game_s2, t): t for t in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  f"a_ms={res['a_ms_per_step']}",
                  flush=True)
    out.close()


if __name__ == "__main__":
    main()
