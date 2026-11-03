"""eval_shadow -- fidelity test for shadow_alpha.py (the BC clone).

Forked from eval_v0_5's ledger instrumentation (same definitions, see
that file): per-step state diffs give exact tree_income, plant/seize/
recruit spends, deaths/death_cargo, residual deposits, stranded cargo.

Extensions for the Alpha-fidelity comparison vs alpha_analysis.md:
  * comp200: shadow-side (collectors, planters, trees) snapshot taken
    from the state at step 200 (AlphaCarbon-2 median: 7 / 2 / 7).
  * dep_last30: residual deposits accumulated over steps 271-300
    (AlphaCarbon-2 median ~ +997).
  * tree_net = tree_income - plant_spend (the accounting residual the
    analysis calls tree net contribution, median ~ +1440).

Games (sequential, one core -- a rule-line job shares this box):
  shadow vs policy_v0_5, seed 1000, both sides
  shadow vs example,     seed 2000, both sides

Run: python3 eval_shadow.py   (system python3, like the other evals)
Appends one JSON line per game to eval_shadow_results.jsonl.
"""

import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

from fastsim import FastCarbon  # noqa: E402
from eval_v0_5 import CFG, new_ledger, _plant_price  # noqa: E402

_counter = [0]


def load_agent_fn(path, kind):
    _counter[0] += 1
    name = f"shadowmod_{kind}_{_counter[0]}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if kind == "policy":
        return m.make_agent()
    return m.agent


def play_game(seed, shadow_side, opp_kind, opp_file):
    sim = FastCarbon(CFG).reset(seed=seed)
    shadow = load_agent_fn(os.path.join(HERE, "shadow_alpha.py"), "policy")
    opp = load_agent_fn(os.path.join(HERE, opp_file),
                        "policy" if opp_kind == "policy" else "opp")
    agents = [None, None]
    agents[shadow_side] = shadow
    agents[1 - shadow_side] = opp
    led = [new_ledger(), new_ledger()]
    dep_last30 = [0.0, 0.0]
    comp200 = None
    my_time = 0.0
    my_calls = 0
    for step in range(300):
        pre = sim.get_state()
        acts = []
        for p in range(2):
            o = sim.obs(p)
            if p == shadow_side:
                ts = time.time()
                acts.append(agents[p](o, CFG))
                my_time += time.time() - ts
                my_calls += 1
            else:
                acts.append(agents[p](o, CFG))
        sim.step(acts[0], acts[1])
        post = sim.get_state()
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
            dep_inc = (post["cash"][p] - pre["cash"][p]) + rs + ps + ss \
                - float(sum(post["trees_dict"].get(tid, [None, 0])[1]
                            for tid, t in post["trees"].items()
                            if t["player"] == p))
            L["deposits"] += dep_inc
            if step >= 270:
                dep_last30[p] += dep_inc
        if post["step"] == 200:
            ws = [w for w in post["workers"].values()
                  if w["player"] == shadow_side]
            comp200 = {
                "collectors": sum(1 for w in ws
                                  if w["type"] == "COLLECTOR"),
                "planters": sum(1 for w in ws if w["type"] == "PLANTER"),
                "trees": sum(1 for t in post["trees"].values()
                             if t["player"] == shadow_side),
            }
    st = sim.get_state()
    for p in range(2):
        led[p]["stranded"] = round(sum(
            w["carbon"] for w in st["workers"].values()
            if w["player"] == p), 1)
        for k in ("deposits", "tree_income", "recruit_spend", "plant_spend",
                  "seize_spend", "death_cargo"):
            led[p][k] = round(led[p][k], 1)
    sl = led[shadow_side]
    sl["tree_net"] = round(sl["tree_income"] - sl["plant_spend"], 1)
    return {
        "seed": seed, "shadow_side": shadow_side, "opp": opp_kind,
        "shadow_cash": round(st["cash"][shadow_side], 2),
        "opp_cash": round(st["cash"][1 - shadow_side], 2),
        "margin": round(st["cash"][shadow_side]
                        - st["cash"][1 - shadow_side], 2),
        "win": st["cash"][shadow_side] > st["cash"][1 - shadow_side],
        "shadow_ms_per_step": round(1000.0 * my_time / max(my_calls, 1), 2),
        "comp200": comp200,
        "dep_last30_shadow": round(dep_last30[shadow_side], 1),
        "ledger_shadow": sl,
        "ledger_opp": led[1 - shadow_side],
    }


def main():
    games = [
        (1000, 0, "policy_v0_5", "policy_v0_5.py"),
        (1000, 1, "policy_v0_5", "policy_v0_5.py"),
        (2000, 0, "example", "example.py"),
        (2000, 1, "example", "example.py"),
    ]
    out = open(os.path.join(HERE, "eval_shadow_results.jsonl"), "a")
    for seed, side, okind, ofile in games:
        t0 = time.time()
        res = play_game(seed, side, okind, ofile)
        res["wall_s"] = round(time.time() - t0, 1)
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(json.dumps(res), flush=True)
    out.close()


if __name__ == "__main__":
    main()
