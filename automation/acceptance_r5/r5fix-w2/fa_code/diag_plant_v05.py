"""diag_plant_v05: per-tree planting profile from the 8 recorded v0.4 vs
juzi games (seeds 1000-1003 x both sides).

Method: deterministic replay of exactly the recorded games on
FastCarbon (same loop as eval_v0_4), tracking every tree id:
  - plant event: planter, step, cell; context from the PRE-step full
    state (own/enemy base distance, cell carbon, 4-neighbour carbon
    sum/mean, enemy planter distance, own-collector density, sharing);
  - per-step income attributed to the tree's CURRENT owner via
    state["trees_dict"][tid][1] (the environment's exact absorption,
    same source the eval ledger uses);
  - seize flips (same tid, owner change) and death (age-out only --
    tids never otherwise disappear).

Validation (hard gates, printed): replay final cash vs the trace
meta final_cash (must be 0.00) and per-player summed tree income /
plant / seize counts vs eval_v0_4_results.jsonl ledgers.

Output: diag_plant_v05.json (per-tree records + aggregates) and a
printed side-by-side profile (us = policy_v0_4, opp = juzi).
"""

import gzip
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, median

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

SIZE = 15
N = SIZE * SIZE


def _nbr4(pos):
    x, y = pos % SIZE, pos // SIZE
    return [((y + dy) % SIZE) * SIZE + (x + dx) % SIZE
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))]


def _dist(a, b):
    ax, ay = a % SIZE, a // SIZE
    bx, by = b % SIZE, b // SIZE
    dx = min(abs(ax - bx), SIZE - abs(ax - bx))
    dy = min(abs(ay - by), SIZE - abs(ay - by))
    return dx + dy


_counter = [0]


def load_agent_fn(path, kind):
    _counter[0] += 1
    name = f"diagplant_{kind}_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if kind == "policy":
        return m.make_agent()
    return m.agent


def play_tracked(task):
    seed, policy_side = task
    sim = FastCarbon(CFG).reset(seed=seed)
    policy = load_agent_fn(os.path.join(HERE, "policy_v0_4.py"), "policy")
    opp = load_agent_fn(os.path.join(HERE, "reference_juzi.py"),
                        "reference_juzi")
    agents = [None, None]
    agents[policy_side] = policy
    agents[1 - policy_side] = opp

    trees = {}          # tid -> record
    income_tot = [0.0, 0.0]
    plant_tot = [0, 0]
    seize_tot = [0, 0]
    for step in range(1, 301):
        pre = sim.get_state()
        acts = []
        for p in range(2):
            o = sim.obs(p)
            acts.append(agents[p](o, CFG))
        sim.step(acts[0], acts[1])
        post = sim.get_state()
        pre_t = pre["trees"]
        post_t = post["trees"]
        bases = {p: [pos for _bid, pos in post["bases"][p]] for p in range(2)}
        # income attribution (owner = post-step owner, eval semantics)
        for tid, t in post_t.items():
            inc = float(post["trees_dict"].get(tid, [None, 0])[1])
            if inc:
                income_tot[t["player"]] += inc
                rec = trees.get(tid)
                if rec is not None:
                    rec["income_by_owner"][t["player"]] += inc
        # plant events
        for tid, t in post_t.items():
            if tid in pre_t:
                continue
            planter = t["player"]
            pos = t["pos"]
            plant_tot[planter] += 1
            carb = pre["carbon"]
            nb4 = _nbr4(pos)
            nb_vals = [carb[c] for c in nb4]
            enemy = 1 - planter
            eplant_d = [min((_dist(pos, w["pos"])
                             for w in pre["workers"].values()
                             if w["player"] == enemy
                             and w["type"] == "PLANTER"), default=99)]
            own_coll_near = sum(
                1 for w in pre["workers"].values()
                if w["player"] == planter and w["type"] == "COLLECTOR"
                and _dist(pos, w["pos"]) <= 2)
            share_trees = sum(
                1 for t2 in pre_t.values()
                if _dist(pos, t2["pos"]) <= 2)
            trees[tid] = {
                "tid": tid, "planter": planter, "birth": step, "pos": pos,
                "death": None, "age_at_death": None,
                "ctx": {
                    "cell_carbon": round(carb[pos], 1),
                    "nb4_sum": round(sum(nb_vals), 1),
                    "nb4_mean": round(mean(nb_vals), 1),
                    "d_own_base": min(_dist(pos, b) for b in bases[planter]),
                    "d_enemy_base": min(_dist(pos, b) for b in bases[enemy]),
                    "d_enemy_planter": eplant_d[0],
                    "own_coll_le2": own_coll_near,
                    "trees_le2": share_trees,
                },
                "flips": [],
                "income_by_owner": {0: 0.0, 1: 0.0},
            }
        # flips
        for tid, t in post_t.items():
            if tid in pre_t and pre_t[tid]["player"] != t["player"]:
                seize_tot[t["player"]] += 1
                if tid in trees:
                    trees[tid]["flips"].append(
                        [step, pre_t[tid]["player"], t["player"]])
        # deaths
        for tid, t in pre_t.items():
            if tid not in post_t and tid in trees:
                trees[tid]["death"] = step
                trees[tid]["age_at_death"] = t["age"]
    final = sim.get_state()
    return {
        "seed": seed, "policy_side": policy_side,
        "final_cash": final["cash"],
        "income_tot": [round(x, 1) for x in income_tot],
        "plant_tot": plant_tot, "seize_tot": seize_tot,
        "trees": list(trees.values()),
    }


def bucket(v, edges):
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def dist_stats(vals):
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(mean(vals), 1),
            "median": round(median(vals), 1)}


def profile(records, planter):
    """Aggregate stats for trees planted by `planter` (absolute player
    indices vary per game; records carry per-game policy_side)."""
    mine = [r for r in records if r["planter_rel"] == planter]
    out = {"n_planted": len(mine)}
    if not mine:
        return out
    steps = [0] * 7
    for r in mine:
        steps[min(r["birth"] // 50, 6)] += 1
    out["plant_step_buckets_50"] = steps
    out["d_own_base"] = dist_stats([r["ctx"]["d_own_base"] for r in mine])
    out["d_enemy_base"] = dist_stats([r["ctx"]["d_enemy_base"] for r in mine])
    out["cell_carbon"] = dist_stats([r["ctx"]["cell_carbon"] for r in mine])
    out["nb4_sum"] = dist_stats([r["ctx"]["nb4_sum"] for r in mine])
    life = [r["lifetime"] for r in mine]
    out["lifetime"] = dist_stats(life)
    out["lifetime_buckets"] = {
        "<10": sum(1 for r in mine if r["lifetime"] < 10),
        "10-24": sum(1 for r in mine if 10 <= r["lifetime"] < 25),
        "25-49": sum(1 for r in mine if 25 <= r["lifetime"] < 50),
        "full50": sum(1 for r in mine
                      if r["lifetime"] >= 50 and not r["censored"]),
        "censored_alive_at_300": sum(1 for r in mine if r["censored"]),
    }
    inc_self = [r["income_self"] for r in mine]
    out["income_to_planter"] = dist_stats(inc_self)
    out["income_buckets"] = {
        "0": sum(1 for v in inc_self if v <= 0.01),
        "0-25": sum(1 for v in inc_self if 0.01 < v < 25),
        "25-75": sum(1 for v in inc_self if 25 <= v < 75),
        "75-150": sum(1 for v in inc_self if 75 <= v < 150),
        "150+": sum(1 for v in inc_self if v >= 150),
    }
    seized = [r for r in mine if r["seized_ever"]]
    out["seized_ever_rate"] = round(len(seized) / len(mine), 3)
    out["mean_flips"] = round(mean(len(r["flips"]) for r in mine), 2)
    tts = [r["time_to_first_seize"] for r in seized
           if r["time_to_first_seize"] is not None]
    out["time_to_first_seize"] = dist_stats(tts)
    tot = sum(r["income_total"] for r in mine)
    out["income_total_of_my_trees"] = round(tot, 1)
    out["income_kept_share"] = round(
        sum(inc_self) / tot, 3) if tot > 0 else None
    out["income_lost_to_enemy"] = round(
        sum(r["income_other"] for r in mine), 1)
    # cross-tabs: income-to-planter and seized rate by own-base distance
    for label, sel in (("d<=3", lambda d: d <= 3),
                       ("d4-6", lambda d: 4 <= d <= 6),
                       ("d>=7", lambda d: d >= 7)):
        sub = [r for r in mine if sel(r["ctx"]["d_own_base"])]
        if sub:
            out[f"xtab_{label}"] = {
                "n": len(sub),
                "income_mean": round(mean(r["income_self"] for r in sub), 1),
                "seized_rate": round(
                    sum(1 for r in sub if r["seized_ever"]) / len(sub), 2),
                "lifetime_mean": round(mean(r["lifetime"] for r in sub), 1),
            }
    sums = sorted(r["ctx"]["nb4_sum"] for r in mine)
    if len(sums) >= 6:
        t1, t2 = sums[len(sums) // 3], sums[2 * len(sums) // 3]
        for label, sel in (("nb4_low", lambda v: v < t1),
                           ("nb4_mid", lambda v: t1 <= v < t2),
                           ("nb4_high", lambda v: v >= t2)):
            sub = [r for r in mine if sel(r["ctx"]["nb4_sum"])]
            if sub:
                out[f"xtab_{label}"] = {
                    "n": len(sub),
                    "income_mean": round(
                        mean(r["income_self"] for r in sub), 1),
                    "seized_rate": round(
                        sum(1 for r in sub if r["seized_ever"]) / len(sub), 2),
                    "lifetime_mean": round(
                        mean(r["lifetime"] for r in sub), 1),
                }
    return out


def main():
    tasks = [(seed, side) for seed in (1000, 1001, 1002, 1003)
             for side in (0, 1)]
    games = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(play_tracked, t): t for t in tasks}
        for fut in as_completed(futs):
            g = fut.result()
            games.append(g)
            print(f"seed{g['seed']} side{g['policy_side']}: cash "
                  f"{g['final_cash'][0]:.2f}:{g['final_cash'][1]:.2f} "
                  f"plants {g['plant_tot']} income "
                  f"{g['income_tot']}", flush=True)
    games.sort(key=lambda g: (g["seed"], g["policy_side"]))

    # ---- validation vs traces + eval ledgers --------------------------
    exp_eval = {}
    with open(os.path.join(HERE, "eval_v0_4_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["tag"] == "eval_v04":
                exp_eval[(r["seed"], r["policy_side"])] = r
    checks = []
    all_records = []
    for g in games:
        key = (g["seed"], g["policy_side"])
        tr = json.load(gzip.open(os.path.join(
            HERE, "traces",
            f"policy_v0_4_vs_reference_juzi_seed{g['seed']}"
            f"_side{g['policy_side']}.json.gz"), "rt"))
        tc = tr["meta"]["final_cash"]
        r = exp_eval[key]
        ps = g["policy_side"]
        led_inc = [0.0, 0.0]
        led_inc[ps] = r["ledger_pol"]["tree_income"]
        led_inc[1 - ps] = r["ledger_opp"]["tree_income"]
        led_plant = [0, 0]
        led_plant[ps] = int(r["ledger_pol"]["plant_count"])
        led_plant[1 - ps] = int(r["ledger_opp"]["plant_count"])
        led_seize = [0, 0]
        led_seize[ps] = int(r["ledger_pol"]["seize_count"])
        led_seize[1 - ps] = int(r["ledger_opp"]["seize_count"])
        chk = {
            "seed": g["seed"], "side": ps,
            "cash_diff": [round(g["final_cash"][p] - tc[p], 4)
                          for p in range(2)],
            "income_diff": [round(g["income_tot"][p] - led_inc[p], 2)
                            for p in range(2)],
            "plant_diff": [g["plant_tot"][p] - led_plant[p]
                           for p in range(2)],
            "seize_diff": [g["seize_tot"][p] - led_seize[p]
                           for p in range(2)],
        }
        chk["ok"] = (all(abs(d) <= 0.02 for d in chk["cash_diff"])
                     and all(abs(d) <= 3.0 for d in chk["income_diff"])
                     and chk["plant_diff"] == [0, 0]
                     and chk["seize_diff"] == [0, 0])
        checks.append(chk)
        print("CHECK", json.dumps(chk), flush=True)
        for rec in g["trees"]:
            rec = dict(rec)
            rec["seed"] = g["seed"]
            rec["policy_side"] = ps
            # relative planter: "us" if the planter is the policy player
            rec["planter_rel"] = "us" if rec["planter"] == ps else "juzi"
            rec["lifetime"] = (rec["death"] or 300) - rec["birth"]
            rec["censored"] = rec["death"] is None
            io = rec["income_by_owner"]     # in-memory: int keys 0/1
            rec["income_total"] = round(io[0] + io[1], 1)
            rec["income_self"] = round(float(io[rec["planter"]]), 1)
            rec["income_other"] = round(rec["income_total"]
                                        - rec["income_self"], 1)
            flips_to_foe = [f for f in rec["flips"]
                            if f[2] != rec["planter"]]
            rec["seized_ever"] = bool(flips_to_foe) or (
                rec["flips"] and rec["flips"][-1][2] != rec["planter"])
            rec["time_to_first_seize"] = (
                flips_to_foe[0][0] - rec["birth"]) if flips_to_foe else None
            all_records.append(rec)

    prof_us = profile(all_records, "us")
    prof_jz = profile(all_records, "juzi")
    out = {"checks": checks, "profile_us": prof_us, "profile_juzi": prof_jz,
           "records": all_records}
    with open(os.path.join(HERE, "diag_plant_v05.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n===== PROFILE us (policy_v0_4) =====")
    print(json.dumps(prof_us, ensure_ascii=False, indent=1))
    print("\n===== PROFILE juzi =====")
    print(json.dumps(prof_jz, ensure_ascii=False, indent=1))
    bad = [c for c in checks if not c["ok"]]
    print(f"\nchecks ok: {len(checks) - len(bad)}/{len(checks)}")


if __name__ == "__main__":
    main()
