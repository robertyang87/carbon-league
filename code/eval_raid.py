"""eval_raid: paired-seed evaluation for the raid (offensive) line.

Same methodology and ledger instrumentation as eval_v0_1.py (same seed
= same map on FastCarbon, policy plays both sides), plus raid-specific
instrumentation reconstructed from FastCarbon state diffs:

  seized_income[p]  -- tree absorption credited to trees AFTER they
                       flipped to player p by a seize (absorption is
                       attributed to the post-step owner, the same
                       convention as ledger tree_income; counting
                       starts on the flip step itself).
  seized_prime[p]   -- number of trees p seized at age <= 25 (most of
                       the income integral still ahead).
  seized_tail[p]    -- number seized at age > 25.
  raid_stats        -- the policy module's own accounting, read from
                       module.LAST_POLICY.raid_stats after the game
                       (hunt sorties / successes / captured carbon /
                       hunter deaths / escapes / switch step).
                       Policy-side only; None for the opponent.

Run:
  python3 eval_raid.py POLICY_FILE TAG OPP [seeds...]
    OPP in {reference_juzi, example}; default seeds 1000 1001 1002 1003.
  Appends one JSON line per game to eval_raid_results.jsonl.
"""

import importlib.util
import json
import os
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

_counter = [0]


def load_module(path, kind):
    _counter[0] += 1
    name = f"raidmod_{kind}_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


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


def play_game(task):
    seed, policy_side, opp_kind, policy_file, tag = task
    sim = FastCarbon(CFG).reset(seed=seed)
    pol_mod = load_module(os.path.join(HERE, policy_file), "policy")
    opp_mod = load_module(os.path.join(HERE, f"{opp_kind}.py"), opp_kind)
    agents = [None, None]
    agents[policy_side] = pol_mod.make_agent()
    agents[1 - policy_side] = opp_mod.agent
    led = [new_ledger(), new_ledger()]
    # raid seized-tree tracking: tid -> player it flipped to by seize
    seized_by = {}
    seized_income = [0.0, 0.0]
    seized_prime = [0, 0]
    seized_tail = [0, 0]
    my_time = 0.0
    my_calls = 0
    t0 = time.time()
    for _ in range(300):
        pre = sim.get_state()
        acts = []
        for p in range(2):
            o = sim.obs(p)
            if p == policy_side:
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
                absorb = float(post["trees_dict"].get(tid, [None, 0])[1])
                if t["player"] == p:
                    L["tree_income"] += absorb
                if tid not in pre["trees"]:
                    if t["player"] == p:
                        L["plant_count"] += 1
                        ps += price
                    seized_by.pop(tid, None)
                elif pre["trees"][tid]["player"] != t["player"]:
                    new_owner = t["player"]
                    seized_by[tid] = new_owner
                    if new_owner == p:
                        L["seize_count"] += 1
                        ss += float(CFG["seizeCost"])
                        if t["age"] <= 25:
                            seized_prime[p] += 1
                        else:
                            seized_tail[p] += 1
                # absorption credited to a seized tree's current owner
                if seized_by.get(tid) == t["player"]:
                    seized_income[t["player"]] += absorb
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
    pol = led[policy_side]
    oppl = led[1 - policy_side]
    lp = getattr(pol_mod, "LAST_POLICY", None)
    raid_stats = getattr(lp, "raid_stats", None) if lp is not None else None
    return {
        "tag": tag, "seed": seed, "opp": opp_kind, "policy_side": policy_side,
        "my_cash": round(st["cash"][policy_side], 2),
        "opp_cash": round(st["cash"][1 - policy_side], 2),
        "margin": round(st["cash"][policy_side] - st["cash"][1 - policy_side], 2),
        "win": st["cash"][policy_side] > st["cash"][1 - policy_side],
        "my_ms_per_step": round(1000.0 * my_time / max(my_calls, 1), 2),
        "wall_s": round(wall, 1),
        "ledger_pol": pol, "ledger_opp": oppl,
        "seized_pol": {"income": round(seized_income[policy_side], 1),
                       "prime": seized_prime[policy_side],
                       "tail": seized_tail[policy_side]},
        "seized_opp": {"income": round(seized_income[1 - policy_side], 1),
                       "prime": seized_prime[1 - policy_side],
                       "tail": seized_tail[1 - policy_side]},
        "raid_stats": raid_stats,
    }


def main():
    policy_file = sys.argv[1]
    tag = sys.argv[2]
    opp_kind = sys.argv[3]
    seeds = [int(s) for s in sys.argv[4:]] or [1000, 1001, 1002, 1003]
    tasks = []
    for seed in seeds:
        for side in (0, 1):
            tasks.append((seed, side, opp_kind, policy_file, tag))
    out_path = os.path.join(HERE, "eval_raid_results.jsonl")
    out = open(out_path, "a")
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(play_game, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  flush=True)
    out.close()


if __name__ == "__main__":
    main()
