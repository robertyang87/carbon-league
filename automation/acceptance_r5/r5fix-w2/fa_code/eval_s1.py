"""eval_s1: paired-seed evaluation of fork_s1_treeval (tree-campaign
valuation, structure line S1) and its ablation variants against
base_v53.

Same protocol as eval_v53 (FastCarbon, same seed = same map, both
sides, RNG seeding random.seed(7919*seed + 104729*policy_side + 31),
ledger reconstructed from state diffs). Extensions for the
稳/准/狠 scorecard:
  * per-tree ownership spells with realized income attributed from
    trees_dict[tid][1] (the sim's per-step tree income, absorption +
    cargo tax), seizure flips, recapture delays, plant payback
    (cumulative income-for-planter >= plant price);
  * the fork's PRED_LOG (plant/seize commitments with new- and
    old-model income predictions) matched to the realized spells, so
    valuation calibration (准) is measured on identical decisions.

Phases (env S1_PHASE):
  full : fork_s1_treeval vs base, seeds 1000-1003 x2 (8 games)
  off1 : one-off ablations (nosim/notax/nohaz), seeds 1000-1001 x2
  on1  : one-on ablations (simonly/taxonly/hazonly), seeds 1000-1001 x2
Appends to eval_s1_results.jsonl.
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

import eval_v53  # noqa: E402  (CFG, rng_seed, _plant_price, _comp, new_ledger)

CFG = eval_v53.CFG
BASE = "base_v53.py"
FORK = "fork_s1_treeval.py"

PHASE = os.environ.get("S1_PHASE", "full")
if PHASE == "full":
    JOBS = [("s1full", FORK, BASE, s, side)
            for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
elif PHASE == "off1":
    JOBS = [(tag, fn, BASE, s, side)
            for tag, fn in (("s1nosim", "fork_s1_nosim.py"),
                            ("s1notax", "fork_s1_notax.py"),
                            ("s1nohaz", "fork_s1_nohaz.py"))
            for s in (1000, 1001) for side in (0, 1)]
elif PHASE == "on1":
    JOBS = [(tag, fn, BASE, s, side)
            for tag, fn in (("s1simonly", "fork_s1_simonly.py"),
                            ("s1taxonly", "fork_s1_taxonly.py"),
                            ("s1hazonly", "fork_s1_hazonly.py"))
            for s in (1000, 1001) for side in (0, 1)]
else:
    raise SystemExit(f"unknown S1_PHASE {PHASE}")

_counter = [0]


def load_module(path):
    _counter[0] += 1
    name = f"evals1_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if os.path.basename(path).startswith("policy_") and hasattr(m, "make_agent"):
        return m.make_agent(), m
    return m.agent, m


def play_game(task):
    tag, a_file, b_file, seed, a_side = task
    random.seed(eval_v53.rng_seed(seed, a_side))
    sim = eval_v53.FastCarbon(CFG).reset(seed=seed)
    fa, mod_a = load_module(os.path.join(HERE, a_file))
    fb, mod_b = load_module(os.path.join(HERE, b_file))
    agents = [None, None]
    mods = [None, None]
    agents[a_side] = fa
    agents[1 - a_side] = fb
    mods[a_side] = mod_a
    mods[1 - a_side] = mod_b
    led = [eval_v53.new_ledger(), eval_v53.new_ledger()]
    times = [0.0, 0.0]
    worst_a = 0.0
    comp = {}
    # --- per-tree spell tracking (狠 + 准)
    active = {}
    spells = []
    cum_for = {}
    planted_info = {}
    paid_step = {}
    loss_step = {}
    recap = {0: [], 1: []}
    flips_to = {0: 0, 1: 0}
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
        # --- spells
        for tid, t in post["trees"].items():
            p = t["player"]
            inc = float(post["trees_dict"].get(tid, [None, 0])[1])
            sp = active.get(tid)
            if sp is None:
                planted = tid not in pre["trees"]
                sp = {"tid": tid, "owner": p, "start": step, "income": 0.0,
                      "planted": planted, "cell": t["pos"],
                      "price": (eval_v53._plant_price(pre["trees"], p)
                                if planted else None),
                      "seized_in": (not planted) and (tid in pre["trees"])}
                active[tid] = sp
                if planted:
                    planted_info[tid] = (p, sp["price"], step)
            elif sp["owner"] != p:
                sp["end"] = step - 1
                sp["how"] = "seized_away"
                spells.append(sp)
                loser = sp["owner"]
                loss_step[(tid, loser)] = step
                flips_to[p] += 1
                if (tid, p) in loss_step:
                    recap[p].append(step - loss_step.pop((tid, p)))
                sp = {"tid": tid, "owner": p, "start": step, "income": 0.0,
                      "planted": False, "cell": t["pos"], "price": None,
                      "seized_in": True}
                active[tid] = sp
            sp["income"] += inc
            key = (tid, p)
            cum_for[key] = cum_for.get(key, 0.0) + inc
            if tid in planted_info and tid not in paid_step:
                planter, price, _ps = planted_info[tid]
                if cum_for.get((tid, planter), 0.0) >= price:
                    paid_step[tid] = step
        for tid in list(active):
            if tid not in post["trees"]:
                sp = active.pop(tid)
                sp["end"] = step
                sp["how"] = "died"
                spells.append(sp)
    for sp in active.values():
        sp["end"] = 300
        sp["how"] = "censored"
        spells.append(sp)
    wall = time.time() - t0
    st = sim.get_state()
    for p in range(2):
        led[p]["stranded"] = round(sum(
            w["carbon"] for w in st["workers"].values() if w["player"] == p), 1)
        for k in ("deposits", "tree_income", "recruit_spend", "plant_spend",
                  "seize_spend", "death_cargo"):
            led[p][k] = round(led[p][k], 1)
    # --- ruth aggregates per player
    ruth = {}
    for p in range(2):
        sps = [s for s in spells if s["owner"] == p]
        conv = []
        nplant = 0
        for tid, (planter, _price, pstep) in planted_info.items():
            if planter != p:
                continue
            nplant += 1
            if tid in paid_step:
                conv.append(paid_step[tid] - pstep)
        ruth[p] = {
            "spells": len(sps),
            "seized_away": sum(1 for s in sps if s["how"] == "seized_away"),
            "died": sum(1 for s in sps if s["how"] == "died"),
            "censored": sum(1 for s in sps if s["how"] == "censored"),
            "seize_flips": flips_to[p],
            "recap_delays": recap[p],
            "planted_trees": nplant,
            "payback_n": len(conv),
            "payback_delays": conv,
        }
    # --- PRED_LOG matching (准)
    preds = []
    used = set()
    for p in range(2):
        log = getattr(mods[p], "PRED_LOG", None)
        if not log:
            continue
        for e in log:
            best = None
            for i, s in enumerate(spells):
                if i in used or s["owner"] != p:
                    continue
                if e["kind"] == "plant":
                    if not s["planted"] or s["cell"] != e["cell"]:
                        continue
                else:
                    if not s["seized_in"] or s["tid"] != e.get("tid"):
                        continue
                d = abs(s["start"] - (e["step"] + 1))
                if d <= 1 and (best is None or d < best[0]):
                    best = (d, i)
            rec = {"player": p, "kind": e["kind"], "step": e["step"],
                   "cell": e["cell"], "pin_new": e["pin_new"],
                   "pin_old": e["pin_old"], "price": e["price"],
                   "matched": best is not None}
            if best is not None:
                used.add(best[1])
                s = spells[best[1]]
                rec["realized"] = round(s["income"], 2)
                rec["how"] = s["how"]
                rec["spell_len"] = s["end"] - s["start"] + 1
            preds.append(rec)
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
        "ruth_pol": ruth[a_side], "ruth_opp": ruth[1 - a_side],
        "preds": preds,
    }


def main():
    out_path = os.path.join(HERE, "eval_s1_results.jsonl")
    out = open(out_path, "a")
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(play_game, t): t for t in JOBS}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  f"a_ms={res['a_ms_per_step']} b_ms={res['b_ms_per_step']}",
                  f"preds={len(res['preds'])}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
