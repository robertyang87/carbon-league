"""eval_gs: paired-seed eval for the GS (guard/steal) single-knob line.

Same FastCarbon, same RNG protocol and same per-player ledger as
eval_v53 / eval_s2 (numbers comparable with eval_s2_results.jsonl,
whose chk_harvest rows are the base-vs-harvest control for seeds
1000-1001), plus what GS needs:
  * seized-tree income attribution: income credited by trees that the
    player seized from the opponent (ownership flip opp -> player),
    accumulated separately as ledger key seized_income
  * the opponent's planter-count trajectory as seen by the policy
    player (pre-step state, steps 1..300): enp_hist, so the trigger
    rule can be reconstructed independently of the bot's own log,
    both in the armed form used by the variants and in the raw
    unarmed form (which fires in the opening of every game)
  * GS trigger readback from the finished Bot (gs_trigger_step,
    gs_max_run, gs_armed, gs_trace), mirroring eval_s2's post_trace

Phases (argv[1]); every game appends one JSON line to
eval_gs_results.jsonl. Run under .eval_lock (caller wraps), with
EVAL_WORKERS<=2.
  seize_h     fork_gs_seize vs switcher_harvest, seeds 1000-1001
  guard_h     fork_gs_guard vs switcher_harvest, seeds 1000-1001
  seize_base  fork_gs_seize vs base_v53,         seeds 1000-1001
  guard_base  fork_gs_guard vs base_v53,         seeds 1000-1001
  seize_h2    fork_gs_seize vs switcher_harvest, seeds 1002-1003
  guard_h2    fork_gs_guard vs switcher_harvest, seeds 1002-1003
  base_h2     base_v53 vs switcher_harvest,      seeds 1002-1003
              (paired control for the h2 expansions)
  base_h1     base_v53 vs switcher_harvest,      seeds 1000-1001
              (same-convention control: the eval_s2 chk_harvest rows
              were generated with the opposite policy-side RNG
              convention; if a variant slot deviates from the chk
              baseline while a second, different-knob variant deviates
              identically, this phase decides whether the deviation is
              a variant effect or a pairing artifact)
  both_h      fork_gs_both vs switcher_harvest,  seeds 1000-1001
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
HARV = "switcher_harvest.py"
SEIZE = "fork_gs_seize.py"
GUARD = "fork_gs_guard.py"
BOTH = "fork_gs_both.py"
SEEDS1 = (1000, 1001)
SEEDS2 = (1002, 1003)
_counter = [0]


def load_module(path):
    _counter[0] += 1
    name = f"evalgs_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _gs_of(mod):
    bot = getattr(mod, "_BOT", None)
    if bot is None or not hasattr(bot, "gs_trace"):
        return None
    return dict(trigger_step=bot.gs_trigger_step,
                max_run=bot.gs_max_run, armed=bot.gs_armed,
                trace=bot.gs_trace)


def play_game_gs(task):
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
    for L in led:
        L["seized_income"] = 0.0
    seized_by = [set(), set()]
    times = [0.0, 0.0]
    worst_a = 0.0
    enp_hist = []
    t0 = time.time()
    for step in range(1, 301):
        pre = sim.get_state()
        enp_hist.append(sum(
            1 for w in pre["workers"].values()
            if w["player"] == 1 - a_side and w["type"] == "PLANTER"))
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
        for p in range(2):
            L = led[p]
            price = eval_v53._plant_price(pre["trees"], p)
            rs = ps = ss = 0.0
            for tid, t in post["trees"].items():
                if t["player"] != p:
                    continue
                inc = float(post["trees_dict"].get(tid, [None, 0])[1])
                L["tree_income"] += inc
                if tid in seized_by[p]:
                    L["seized_income"] += inc
                if tid not in pre["trees"]:
                    L["plant_count"] += 1
                    ps += price
                elif pre["trees"][tid]["player"] != p:
                    L["seize_count"] += 1
                    ss += float(eval_v53.CFG["seizeCost"])
                    seized_by[p].add(tid)
                    seized_by[1 - p].discard(tid)
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
    wall = time.time() - t0
    st = sim.get_state()
    for p in range(2):
        led[p]["stranded"] = round(sum(
            w["carbon"] for w in st["workers"].values()
            if w["player"] == p), 1)
        for k in ("deposits", "tree_income", "seized_income",
                  "recruit_spend", "plant_spend", "seize_spend",
                  "death_cargo"):
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
        "enp_hist": enp_hist,
        "gs_pol": _gs_of(mods[a_side]),
        "gs_opp": _gs_of(mods[1 - a_side]),
    }


def jobs_for(phase):
    if phase == "seize_h":
        return [("gs_seize_h", SEIZE, HARV, s, side)
                for s in SEEDS1 for side in (0, 1)]
    if phase == "guard_h":
        return [("gs_guard_h", GUARD, HARV, s, side)
                for s in SEEDS1 for side in (0, 1)]
    if phase == "seize_base":
        return [("gs_seize_base", SEIZE, BASE, s, side)
                for s in SEEDS1 for side in (0, 1)]
    if phase == "guard_base":
        return [("gs_guard_base", GUARD, BASE, s, side)
                for s in SEEDS1 for side in (0, 1)]
    if phase == "seize_h2":
        return [("gs_seize_h2", SEIZE, HARV, s, side)
                for s in SEEDS2 for side in (0, 1)]
    if phase == "guard_h2":
        return [("gs_guard_h2", GUARD, HARV, s, side)
                for s in SEEDS2 for side in (0, 1)]
    if phase == "base_h2":
        return [("gs_base_h2", BASE, HARV, s, side)
                for s in SEEDS2 for side in (0, 1)]
    if phase == "base_h1":
        return [("gs_base_h1", BASE, HARV, s, side)
                for s in SEEDS1 for side in (0, 1)]
    if phase == "both_h":
        return [("gs_both_h", BOTH, HARV, s, side)
                for s in SEEDS1 for side in (0, 1)]
    if phase == "smoke":
        return [("gs_smoke", SEIZE, HARV, 1000, 0)]
    raise SystemExit(f"unknown phase {phase}")


def main():
    phase = sys.argv[1]
    jobs = jobs_for(phase)
    out = open(os.path.join(HERE, "eval_gs_results.jsonl"), "a")
    with ProcessPoolExecutor(
            max_workers=int(os.environ.get("EVAL_WORKERS", "2"))) as ex:
        futs = {ex.submit(play_game_gs, t): t for t in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  f"a_ms={res['a_ms_per_step']}",
                  f"trig={(res['gs_pol'] or {}).get('trigger_step')}",
                  flush=True)
    out.close()


if __name__ == "__main__":
    main()
