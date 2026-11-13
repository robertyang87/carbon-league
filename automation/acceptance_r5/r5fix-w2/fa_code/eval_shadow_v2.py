"""eval_shadow_v2 -- fidelity test for shadow_alpha_v2.py, forked from
eval_shadow.py (same ledger instrumentation/definitions), extended
with DEATH ATTRIBUTION so the v2 decorrelation fix can be measured
against its target (v1's own-worker stacking suicides):

For every step, each worker's destination cell is reconstructed from
the pre-step state plus the action dict actually submitted (movement
is deterministic; recruits spawn on their base cell and stay). A death
is then classified by the composition of its destination group:
  own_collision -- the group held only the dying side's own workers
  mixed         -- the group also held an enemy worker
  other         -- singleton group (e.g. empty carrier deleted on an
                   enemy base cell)
Also records: per-death cargo list (median reported), own-destination
overlap exposure (own workers sharing a destination with >= 1 other
own worker), and the agent's decorrelation counters when present
(fn.stats from make_agent()).

Usage:
  python3 eval_shadow_v2.py SHADOW_FILE OUT_JSONL [--juzi]
Games: shadow vs policy_v0_5 seed 1000 both sides; shadow vs example
seed 2000 both sides; with --juzi also shadow vs reference_juzi
seed 3000 side 0 (juzi is slow -- reference only).
"""

import importlib.util
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

from fastsim import FastCarbon  # noqa: E402
from eval_v0_5 import CFG, new_ledger, _plant_price  # noqa: E402

_counter = [0]
SIZE = 15
_DELTA = {"UP": (0, 1), "RIGHT": (1, 0), "DOWN": (0, -1), "LEFT": (-1, 0)}


def load_agent_fn(path, kind):
    _counter[0] += 1
    name = f"shadowv2mod_{kind}_{_counter[0]}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if kind == "policy":
        return m.make_agent()
    return m.agent


def _dest(pos, act):
    if act not in _DELTA:
        return pos
    x, y = pos % SIZE, SIZE - 1 - pos // SIZE
    dx, dy = _DELTA[act]
    nx, ny = (x + dx) % SIZE, (y + dy) % SIZE
    return (SIZE - 1 - ny) * SIZE + nx


def play_game(seed, shadow_side, opp_kind, opp_file, shadow_file):
    sim = FastCarbon(CFG).reset(seed=seed)
    shadow = load_agent_fn(os.path.join(HERE, shadow_file), "policy")
    opp = load_agent_fn(os.path.join(HERE, opp_file),
                        "policy" if opp_kind == "policy_v0_5" else "opp")
    agents = [None, None]
    agents[shadow_side] = shadow
    agents[1 - shadow_side] = opp
    led = [new_ledger(), new_ledger()]
    for L in led:
        L.update(deaths_own_collision=0, deaths_mixed=0, deaths_other=0,
                 death_cargo_own_collision=0.0, own_dest_overlap=0)
    death_cargo_list = []
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
        # ---- death attribution from destinations ----
        dest = {}
        for wid, w in pre["workers"].items():
            act = (acts[w["player"]] or {}).get(wid)
            dest[wid] = _dest(w["pos"], act)
        for wid, w in post["workers"].items():
            if wid not in pre["workers"]:
                dest[wid] = w["pos"]  # recruit: spawned here, stays
        groups = {}
        for wid, d in dest.items():
            if wid in pre["workers"] or wid in post["workers"]:
                groups.setdefault(d, []).append(wid)
        for p in range(2):
            own_dests = [dest[wid] for wid, w in pre["workers"].items()
                         if w["player"] == p]
            from collections import Counter as _C
            cnt = _C(own_dests)
            led[p]["own_dest_overlap"] += sum(
                1 for d in own_dests if cnt[d] > 1)
            for wid, w in pre["workers"].items():
                if w["player"] != p or wid in post["workers"]:
                    continue
                grp = groups.get(dest[wid], [])
                owners = {pre["workers"][g]["player"] if g in pre["workers"]
                          else post["workers"][g]["player"] for g in grp}
                if len(grp) <= 1:
                    led[p]["deaths_other"] += 1
                elif owners == {p}:
                    led[p]["deaths_own_collision"] += 1
                    led[p]["death_cargo_own_collision"] += float(w["carbon"])
                else:
                    led[p]["deaths_mixed"] += 1
                if p == shadow_side:
                    death_cargo_list.append(float(w["carbon"]))
        # ---- ledger (verbatim from eval_shadow.py) ----
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
                  "seize_spend", "death_cargo",
                  "death_cargo_own_collision"):
            led[p][k] = round(led[p][k], 1)
    sl = led[shadow_side]
    sl["tree_net"] = round(sl["tree_income"] - sl["plant_spend"], 1)
    res = {
        "seed": seed, "shadow_side": shadow_side, "opp": opp_kind,
        "shadow_file": shadow_file,
        "shadow_cash": round(st["cash"][shadow_side], 2),
        "opp_cash": round(st["cash"][1 - shadow_side], 2),
        "margin": round(st["cash"][shadow_side]
                        - st["cash"][1 - shadow_side], 2),
        "win": st["cash"][shadow_side] > st["cash"][1 - shadow_side],
        "shadow_ms_per_step": round(1000.0 * my_time / max(my_calls, 1), 2),
        "comp200": comp200,
        "dep_last30_shadow": round(dep_last30[shadow_side], 1),
        "death_cargo_median_shadow": (
            round(statistics.median(death_cargo_list), 1)
            if death_cargo_list else None),
        "ledger_shadow": sl,
        "ledger_opp": led[1 - shadow_side],
    }
    stats = getattr(shadow, "stats", None)
    if stats is not None:
        res["decorr_stats"] = dict(stats)
    return res


def main():
    shadow_file = sys.argv[1]
    out_path = sys.argv[2]
    games = [
        (1000, 0, "policy_v0_5", "policy_v0_5.py"),
        (1000, 1, "policy_v0_5", "policy_v0_5.py"),
        (2000, 0, "example", "example.py"),
        (2000, 1, "example", "example.py"),
    ]
    if "--juzi" in sys.argv:
        games.append((3000, 0, "reference_juzi", "reference_juzi.py"))
    out = open(out_path, "a")
    for seed, side, okind, ofile in games:
        t0 = time.time()
        res = play_game(seed, side, okind, ofile, shadow_file)
        res["wall_s"] = round(time.time() - t0, 1)
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(json.dumps(res), flush=True)
    out.close()


if __name__ == "__main__":
    main()
