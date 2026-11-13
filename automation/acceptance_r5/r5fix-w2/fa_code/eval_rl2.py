"""eval_rl2 -- same-scale final evaluation of rl_policy_v2 (rl-v0.2)
plus shield ablations on the probe set.

Part A (final, eval_v0_5.play_game ledgers):
  rl_policy_v2 vs base_v53,       seeds 1000-1003, both sides (8)
  rl_policy_v2 vs shadow_alpha_v2, seeds 1000-1001, both sides (4)
  rl_policy_v2 vs policy_v0_5,    seed 1000, both sides (2)
  rl_policy_v2 vs example,        seed 2000, both sides (2)
    (Composition per user directives 2026-11-03 09:29/09:46: v53 is
    the house benchmark base and juzi is retired as a sparring
    opponent -- no new juzi games; historical juzi numbers are cited
    in the report only.)
Part B (acceptance, custom runner over the deployed file):
  rl_policy_v2 vs example, seeds 2000/2001, both sides (4):
  self-collision deaths (landing-group attribution) + stranded.
Part C (shield ablations, probe set, play_game ledgers):
  {rl_policy_v2, rl_policy_v2_noopen, rl_policy_v2_noend}
    vs base_v53 seed 4100 both sides (2) + vs v0.5 seeds 4100/4101
    both sides (4) = 6 games per variant.

Results -> eval_rl2_results.jsonl (parts A/C) and
eval_rl2_acceptance.jsonl (part B). Run: python3 eval_rl2.py [part]
"""

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src",
                                "carbon_challenge-master"))

from eval_v0_5 import play_game  # noqa: E402


def _seeded_play(task):
    """Project-standard RNG protocol (eval_v53.py): base_v53 (and
    example) draw from the global `random` module, so every game is
    seeded, in its worker, immediately before play, with
    random.seed(7919*seed + 104729*policy_side + 31) where
    policy_side is rl_policy_v2's side. All margins in
    eval_rl2_results.jsonl are same-convention and only comparable
    within it (cross-convention pseudo-differences up to ~1839 were
    measured by the GS line, guard_steal_report.md)."""
    seed, policy_side = task[0], task[1]
    random.seed(7919 * seed + 104729 * policy_side + 31)
    return play_game(task)


def part_a_tasks():
    tasks = []
    for seed in (1000, 1001, 1002, 1003):
        for side in (0, 1):
            tasks.append((seed, side, "base_v53",
                          "rl_policy_v2.py", "rl2_v53"))
    for seed in (1000, 1001):
        for side in (0, 1):
            tasks.append((seed, side, "shadow_alpha_v2",
                          "rl_policy_v2.py", "rl2_shadow2"))
    for side in (0, 1):
        tasks.append((1000, side, "policy_v0_5", "rl_policy_v2.py",
                      "rl2_v05"))
        tasks.append((2000, side, "example", "rl_policy_v2.py",
                      "rl2_example"))
    return tasks


def part_c_tasks():
    tasks = []
    for pf in ("rl_policy_v2.py", "rl_policy_v2_noopen.py",
               "rl_policy_v2_noend.py"):
        tag = pf.replace("rl_policy_", "rl2abl_").replace(".py", "")
        for side in (0, 1):
            tasks.append((4100, side, "base_v53", pf, tag))
        for seed in (4100, 4101):
            for side in (0, 1):
                tasks.append((seed, side, "policy_v0_5", pf, tag))
    return tasks


def _acceptance_game(task):
    seed, side = task
    random.seed(7919 * seed + 104729 * side + 31)  # eval_v53 protocol
    import rl_common
    from bc_features import _idx, _xs, _ys
    from fastsim import FastCarbon
    cfg = rl_common.CFG
    sim = FastCarbon(cfg).reset(seed=seed)
    pol = rl_common.load_agent_fn(
        os.path.join(HERE, "rl_policy_v2.py"), "policy")
    opp = rl_common.load_agent_fn(
        os.path.join(HERE, "example.py"), "opp")
    agents = [None, None]
    agents[side] = pol
    agents[1 - side] = opp
    DELTA = {None: (0, 0), "UP": (0, 1), "RIGHT": (1, 0),
             "DOWN": (0, -1), "LEFT": (-1, 0)}
    suicides = 0
    for _ in range(300):
        pre = {wid: (w.player, w.pos) for wid, w in
               sim.workers.items()}
        acts = [agents[p](sim.obs(p), cfg) for p in range(2)]
        groups = {}
        for wid, (pl, pos) in pre.items():
            a = (acts[pl] or {}).get(wid)
            dx, dy = DELTA.get(a, (0, 0))
            land = int(_idx(int(_xs[pos]) + dx, int(_ys[pos]) + dy))
            groups.setdefault(land, []).append((wid, pl))
        sim.step(acts[0], acts[1])
        alive = set(sim.workers)
        for land, members in groups.items():
            sides = {pl for _w, pl in members}
            if sides == {side} and len(members) > 1:
                suicides += sum(1 for w, _pl in members
                                if w not in alive)
    st = sim.get_state()
    stranded = float(sum(w["carbon"] for w in st["workers"].values()
                         if w["player"] == side))
    return {"tag": "rl2_accept_example", "seed": seed, "side": side,
            "margin": round(st["cash"][side] - st["cash"][1 - side], 2),
            "my_cash": round(st["cash"][side], 2),
            "suicides": suicides, "stranded": round(stranded, 1)}


def run_play_tasks(tasks, out_name):
    out = open(os.path.join(HERE, out_name), "a")
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_seeded_play, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  flush=True)
    out.close()


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("all", "a"):
        run_play_tasks(part_a_tasks(), "eval_rl2_results.jsonl")
    if part in ("all", "b"):
        out = open(os.path.join(HERE, "eval_rl2_acceptance.jsonl"),
                   "a")
        tasks = [(s, side) for s in (2000, 2001) for side in (0, 1)]
        with ProcessPoolExecutor(max_workers=2) as ex:
            for res in ex.map(_acceptance_game, tasks):
                out.write(json.dumps(res) + "\n")
                out.flush()
                print(res, flush=True)
        out.close()
    if part in ("all", "c"):
        run_play_tasks(part_c_tasks(), "eval_rl2_results.jsonl")


if __name__ == "__main__":
    main()
