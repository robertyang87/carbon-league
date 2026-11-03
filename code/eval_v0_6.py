"""eval_v0_6: paired-seed evaluation with per-player ledger instrumentation
plus the policy's v0.4 tree-warfare probe counters (result key "tw").

Forked from eval_v0_1 (which see for the ledger definitions).

Extends eval_v1 (same paired methodology: same seed = same map on
FastCarbon, policy plays both sides in two games) with a per-step ledger
for BOTH players, reconstructed from FastCarbon state diffs:

  tree_income   -- exact: sum over the player's trees of
                   state["trees_dict"][tid][1], which Board.next() sets to
                   that tree's actual absorption (cells + enemy collectors)
                   during the step just played.
  plant_spend / plant_count
                -- a tree id appearing in the post-step state owned by p:
                   count += 1, spend += market price computed from the
                   PRE-step state (20 flat while p has <= 0 trees, else
                   20 + 5 * 1.235 ** (alive trees on the whole board)),
                   mirroring helpers.py Board.next() phase 0.
  seize_spend / seize_count
                -- a tree id present in both states whose owner flipped to
                   p: count += 1, spend += 20 (seizeCost).
  recruit_spend / recruit_count
                -- a worker id appearing in the post-step state owned by p:
                   count += 1, spend += 30 (recCollectorCost /
                   recPlanterCost, increase rates are 0 in this config).
  deaths / death_cargo
                -- worker ids of p present pre-step, absent post-step:
                   count += 1, death_cargo += their pre-step cargo.
  deposits      -- RESIDUAL: cash_delta + recruit_spend + plant_spend
                   + seize_spend - tree_income. This is every cash inflow
                   that is not tree absorption: base deposits (the bulk),
                   plus base-mouth collision spill to a base/tree owner and
                   enemy collectors depositing on my bases. Cash is rounded
                   to 2 dp per step by the environment, so the residual
                   carries <= ~3 units of rounding noise per game.
  stranded      -- cargo still on the player's workers after step 300.

Run:
  python3 eval_v0_6.py POLICY_FILE TAG OPP [seeds...]
    OPP in {reference_juzi, example}; default seeds 1000 1001 1002 1003.
  Appends one JSON line per game to eval_v0_5_results.jsonl.
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


def load_agent_fn(path, kind):
    _counter[0] += 1
    name = f"eval2mod_{kind}_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if kind == "policy":
        return m.make_agent()
    return m.agent


def _plant_price(pre_trees, player):
    """Market plant price from the pre-step tree state (Board.next phase 0)."""
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
    policy = load_agent_fn(os.path.join(HERE, policy_file), "policy")
    opp = load_agent_fn(os.path.join(HERE, f"{opp_kind}.py"), opp_kind)
    agents = [None, None]
    agents[policy_side] = policy
    agents[1 - policy_side] = opp
    led = [new_ledger(), new_ledger()]
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
        # per-step accounting (spends recomputed here so the residual uses
        # THIS step's spends, not cumulative ones)
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
    pol = led[policy_side]
    oppl = led[1 - policy_side]
    # v0.4 probe counters (policy_v0_4 exposes fn.policy; other
    # policies simply yield an empty dict)
    tw = {}
    _pol = getattr(policy, "policy", None)
    if _pol is not None:
        tw = dict(_pol.tw)
        for _k in ("flip_val_sum",):
            tw[_k] = round(tw[_k], 1)
        tw["disp_active_end"] = len(_pol.tw_targets)
        tw["targets_left"] = len(_pol.reseize_targets)
        tw["tw_targets_detail"] = {
            str(k): v for k, v in _pol.tw_targets.items()}
    return {
        "tag": tag, "seed": seed, "opp": opp_kind, "policy_side": policy_side,
        "my_cash": round(st["cash"][policy_side], 2),
        "opp_cash": round(st["cash"][1 - policy_side], 2),
        "margin": round(st["cash"][policy_side] - st["cash"][1 - policy_side], 2),
        "win": st["cash"][policy_side] > st["cash"][1 - policy_side],
        "my_ms_per_step": round(1000.0 * my_time / max(my_calls, 1), 2),
        "wall_s": round(wall, 1),
        "ledger_pol": pol, "ledger_opp": oppl,
        "tw": tw,
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
    out_path = os.path.join(HERE, "eval_v0_6_results.jsonl")
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
