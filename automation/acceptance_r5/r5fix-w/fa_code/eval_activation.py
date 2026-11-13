"""eval_activation: paired-seed eval for the lineage-regression
activation scan (D1-D4, D6 candidates vs base_v53).

Same protocol as eval_v53.play_game (FastCarbon, random.seed(
7919*seed + 104729*side + 31), same ledger reconstruction), plus
mechanism-specific window metrics:

  dep_r40 / dep_r28 : deposits banked in steps >=261 (R<=40) / >=273
                      (R<=28) -- late-game delivery account
  rec_r40           : recruits in steps >=261 (late_post footprint)
  posts_r28         : mean # of own collectors standing on enemy-base
                      diagonal cells during steps >=273 (camp post
                      on-duty proxy, observed from sim state)
  posts_all         : same, averaged over all 300 steps

Phases (argv[1]):
  smoke    1 game per candidate, seed 1000 side 0 (sanity only)
  scan     4 games per candidate, seeds 1000-1001 x2 sides
  confirm  4 more games for tags in argv[2:], seeds 1002-1003 x2 sides
  stack    4 games for fork_dstack.py, seeds 1000-1001 x2 sides
Results append to eval_activation_results.jsonl.
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_v53  # noqa: E402

CFG = eval_v53.CFG
BASE = "base_v53.py"
SIZE = 15

CANDS = {
    "d6mark": "fork_d6mark.py",
    "d6rovers": "fork_d6rovers.py",
    "d6pextra": "fork_d6pextra.py",
    "d6campocc": "fork_d6campocc.py",
    "d2nopost": "fork_d2nopost.py",
    "d3keep25": "fork_d3keep25.py",
    "d4tie3": "fork_d4tie3.py",
    "d1floor": "fork_d1floor.py",
}


def _diag_cells(bases):
    cells = set()
    for _bid, pos in bases:
        r, c = divmod(pos, SIZE)
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            cells.add(((r + dr) % SIZE) * SIZE + (c + dc) % SIZE)
    return cells


def play_game(task):
    tag, a_file, b_file, seed, a_side = task
    random.seed(eval_v53.rng_seed(seed, a_side))
    sim = eval_v53.FastCarbon(CFG).reset(seed=seed)
    fa = eval_v53.load_agent_fn(os.path.join(HERE, a_file))
    fb = eval_v53.load_agent_fn(os.path.join(HERE, b_file))
    agents = [None, None]
    agents[a_side] = fa
    agents[1 - a_side] = fb
    led = [eval_v53.new_ledger(), eval_v53.new_ledger()]
    ex = [dict(dep_r40=0.0, dep_r28=0.0, rec_r40=0,
               posts_r28_sum=0, posts_all_sum=0) for _ in range(2)]
    st0 = sim.get_state()
    diag = {p: _diag_cells(st0["bases"][1 - p]) for p in range(2)}
    times = [0.0, 0.0]
    worst_a = 0.0
    comp = {}
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
            ti_step = 0.0
            rec_step = 0
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
                    rec_step += 1
                    rs += float(CFG["recCollectorCost"])
            for wid, w in pre["workers"].items():
                if w["player"] == p and wid not in post["workers"]:
                    L["deaths"] += 1
                    L["death_cargo"] += float(w["carbon"])
            L["recruit_spend"] += rs
            L["plant_spend"] += ps
            L["seize_spend"] += ss
            dep_step = (post["cash"][p] - pre["cash"][p]) + rs + ps + ss - ti_step
            L["deposits"] += dep_step
            E = ex[p]
            if step >= 261:
                E["dep_r40"] += dep_step
                E["rec_r40"] += rec_step
            if step >= 273:
                E["dep_r28"] += dep_step
            posts = sum(1 for w in post["workers"].values()
                        if w["player"] == p and w["type"] == "COLLECTOR"
                        and w["pos"] in diag[p])
            E["posts_all_sum"] += posts
            if step >= 273:
                E["posts_r28_sum"] += posts
    wall = time.time() - t0
    st = sim.get_state()
    for p in range(2):
        led[p]["stranded"] = round(sum(
            w["carbon"] for w in st["workers"].values() if w["player"] == p), 1)
        for k in ("deposits", "tree_income", "recruit_spend", "plant_spend",
                  "seize_spend", "death_cargo"):
            led[p][k] = round(led[p][k], 1)
        ex[p]["dep_r40"] = round(ex[p]["dep_r40"], 1)
        ex[p]["dep_r28"] = round(ex[p]["dep_r28"], 1)
        ex[p]["posts_all"] = round(ex[p].pop("posts_all_sum") / 300.0, 2)
        ex[p]["posts_r28"] = round(ex[p].pop("posts_r28_sum") / 28.0, 2)
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
        "extra_pol": ex[a_side], "extra_opp": ex[1 - a_side],
        "comp": {str(k): v for k, v in comp.items()},
    }


def jobs_for(phase, extra):
    if phase == "smoke":
        return [(t, f, BASE, 1000, 0) for t, f in CANDS.items()]
    if phase == "scan":
        return [(t, f, BASE, s, side) for t, f in CANDS.items()
                for s in (1000, 1001) for side in (0, 1)]
    if phase == "confirm":
        return [(t, CANDS[t], BASE, s, side) for t in extra
                for s in (1002, 1003) for side in (0, 1)]
    if phase == "stack":
        return [("dstack", "fork_dstack.py", BASE, s, side)
                for s in (1000, 1001) for side in (0, 1)]
    raise SystemExit(f"unknown phase {phase}")


def main():
    phase = sys.argv[1]
    jobs = jobs_for(phase, sys.argv[2:])
    res_path = os.path.join(HERE, "eval_activation_results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["tag"], r["seed"], r["policy_side"]))
    jobs = [j for j in jobs if (j[0], j[3], j[4]) not in done]
    print(f"{len(jobs)} new games to run", flush=True)
    out = open(res_path, "a")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("EVAL_WORKERS", "2"))) as ex_:
        futs = {ex_.submit(play_game, t): t for t in jobs}
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
