"""eval_s3p2: paired-seed ablation for the S3 v2 structural line
(fork_s3_phase2: P6 sprint phase recalibrated -- ratio modes +
hysteresis + banking + protect fade + widened recruit gate;
P5 hunt loop byte-identical to v1) vs base_v53.

Methodology = eval_v53 (same FastCarbon, same RNG protocol
random.seed(7919*seed + 104729*policy_side + 31), same ledger
reconstruction from state diffs). Extensions:
  * per-step deposit series -> deposits in the last 50 steps and the
    score margin at sprint-phase entry (after loop step 250);
  * per-worker deposit attribution (cargo decrease while standing on
    an own base cell, incl. disappearance on an own base cell) ->
    payback accounting for collectors recruited in the phase window
    (first appearance at loop steps 250..285, i.e. recruit actions at
    R in [15,50]);
  * post-game read of Bot internals (hs war ledger for both files,
    s3_stats where present).

Phases (argv[1]):
  smoke    both2 vs base, seed 999 x2
  sanity   off2-variant vs base + base vs base, seed 1000 x2
           (off2 must reproduce base-vs-base margins exactly;
           honly2 must reproduce v1 s3honly margins exactly)
  main     both2 seeds 1000-1003 x2 (8), honly2 seeds
           1000-1003 x2 (8) -- phase net effect decomposable on
           all 8 paired games
  abl      norec (v1 recruit bars) seeds 1000-1001 x2 (4)
Results append to eval_s3p2_results.jsonl.
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
import eval_v53  # noqa: E402

CFG = eval_v53.CFG
BASE = "base_v53.py"
BOTH = "fork_s3_phase2.py"
NOREC = "fork_s3p2_norec.py"
HONLY = "fork_s3p2_honly.py"
OFF = "fork_s3p2_off.py"

_counter = [0]


def load_module(path):
    _counter[0] += 1
    name = f"evals3_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if os.path.basename(path).startswith("policy_") and hasattr(m, "make_agent"):
        return m, m.make_agent()
    return m, m.agent


def bot_stats(module):
    bot = getattr(module, "_BOT", None)
    if bot is None:
        return {}
    out = {"hs": dict(getattr(bot, "hs", {}))}
    st = getattr(bot, "s3_stats", None)
    if st is not None:
        out["s3"] = {k: v for k, v in st.items()}
        out["s3"]["hunt_level_final"] = getattr(bot, "s3_hunt_level", None)
    st2 = getattr(bot, "s3p2_stats", None)
    if st2 is not None:
        out["s3p2"] = {k: v for k, v in st2.items()}
        out["s3p2"]["mode_final"] = getattr(bot, "s3p2_mode", None)
    return out


def play_game(task):
    tag, a_file, b_file, seed, a_side = task
    random.seed(eval_v53.rng_seed(seed, a_side))
    sim = FastCarbon(CFG).reset(seed=seed)
    ma, fa = load_module(os.path.join(HERE, a_file))
    mb, fb = load_module(os.path.join(HERE, b_file))
    agents = [None, None]
    modules = [None, None]
    agents[a_side] = fa
    agents[1 - a_side] = fb
    modules[a_side] = ma
    modules[1 - a_side] = mb
    led = [eval_v53.new_ledger(), eval_v53.new_ledger()]
    times = [0.0, 0.0]
    worst_a = 0.0
    comp = {}
    st0 = sim.get_state()
    base_cells = {p: {pos for _, pos in st0["bases"][p]} for p in range(2)}
    first_seen = {}          # (p, wid) -> (loop_step, type)
    dep_worker = {}          # (p, wid) -> attributed deposits
    dep_last50 = [0.0, 0.0]
    cash250 = [None, None]
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
        if step == 250:
            cash250 = [post["cash"][0], post["cash"][1]]
        if step in (100, 200, 300):
            comp[step] = [eval_v53._comp(post, 0), eval_v53._comp(post, 1)]
        # --- per-worker deposit attribution
        for p in range(2):
            prew = {w: v for w, v in pre["workers"].items()
                    if v["player"] == p}
            postw = {w: v for w, v in post["workers"].items()
                     if v["player"] == p}
            for wid, w in postw.items():
                if wid not in first_seen:
                    first_seen[(p, wid)] = (step, w["type"])
                if wid in prew and w["pos"] in base_cells[p] \
                        and w["carbon"] < prew[wid]["carbon"]:
                    dep_worker[(p, wid)] = dep_worker.get((p, wid), 0.0) \
                        + (prew[wid]["carbon"] - w["carbon"])
            for wid, w in prew.items():
                if wid not in postw and w["pos"] in base_cells[p]:
                    dep_worker[(p, wid)] = dep_worker.get((p, wid), 0.0) \
                        + float(w["carbon"])
        # --- ledger (verbatim from eval_v53)
        for p in range(2):
            L = led[p]
            price = eval_v53._plant_price(pre["trees"], p)
            rs = ps = ss = 0.0
            ti_step = 0.0
            for tid, t in post["trees"].items():
                if t["player"] != p:
                    continue
                ti = float(post["trees_dict"].get(tid, [None, 0])[1])
                L["tree_income"] += ti
                ti_step += ti
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
            dep_step = (post["cash"][p] - pre["cash"][p]) + rs + ps + ss \
                - ti_step
            L["deposits"] += dep_step
            if step >= 251:
                dep_last50[p] += dep_step
    wall = time.time() - t0
    st = sim.get_state()
    for p in range(2):
        led[p]["stranded"] = round(sum(
            w["carbon"] for w in st["workers"].values()
            if w["player"] == p), 1)
        for k in ("deposits", "tree_income", "recruit_spend", "plant_spend",
                  "seize_spend", "death_cargo"):
            led[p][k] = round(led[p][k], 1)

    def recruits(p):
        out = []
        for (pp, wid), (fs, typ) in first_seen.items():
            if pp == p and typ == "COLLECTOR" and 250 <= fs <= 285:
                out.append(round(dep_worker.get((p, wid), 0.0), 1))
        return {
            "n": len(out),
            "paid_back": sum(1 for d in out if d >= 30.0 - 1e-9),
            "mean_dep": round(sum(out) / len(out), 1) if out else None,
            "deps": sorted(out),
        }

    stats = [bot_stats(modules[0]), bot_stats(modules[1])]
    return {
        "tag": tag, "seed": seed, "policy_side": a_side,
        "a_file": a_file, "b_file": b_file,
        "my_cash": round(st["cash"][a_side], 2),
        "opp_cash": round(st["cash"][1 - a_side], 2),
        "margin": round(st["cash"][a_side] - st["cash"][1 - a_side], 2),
        "win": st["cash"][a_side] > st["cash"][1 - a_side],
        "margin_entry250": round(cash250[a_side] - cash250[1 - a_side], 2),
        "dep_last50_pol": round(dep_last50[a_side], 1),
        "dep_last50_opp": round(dep_last50[1 - a_side], 1),
        "a_ms_per_step": round(1000.0 * times[a_side] / 300, 2),
        "b_ms_per_step": round(1000.0 * times[1 - a_side] / 300, 2),
        "a_worst_step_ms": round(1000.0 * worst_a, 0),
        "wall_s": round(wall, 1),
        "ledger_pol": led[a_side], "ledger_opp": led[1 - a_side],
        "recruits_pol": recruits(a_side),
        "recruits_opp": recruits(1 - a_side),
        "bot_pol": stats[a_side], "bot_opp": stats[1 - a_side],
        "comp": {str(k): v for k, v in comp.items()},
    }


def jobs_for(phase):
    if phase == "smoke":
        return [("s3p2smoke", BOTH, BASE, 999, side) for side in (0, 1)]
    if phase == "sanity":
        return ([("s3p2off", OFF, BASE, 1000, side) for side in (0, 1)]
                + [("basebase", BASE, BASE, 1000, side) for side in (0, 1)])
    if phase == "main":
        return ([("s3p2both", BOTH, BASE, s, side)
                 for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
                + [("s3p2honly", HONLY, BASE, s, side)
                   for s in (1000, 1001, 1002, 1003) for side in (0, 1)])
    if phase == "abl":
        return [("s3p2norec", NOREC, BASE, s, side)
                for s in (1000, 1001) for side in (0, 1)]
    raise SystemExit(f"unknown phase {phase}")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "main"
    jobs = jobs_for(phase)
    out = open(os.path.join(HERE, "eval_s3p2_results.jsonl"), "a")
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
