"""Measure opening economics against two fixed scripted opponents on four held-out maps."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reports/sapling_bootstrap"))
import transfer_3s2 as audit
import numpy as np

SEEDS = tuple(range(620096810, 620096814))
OPPONENTS = ("policy_raid_v0_1", "policy_v0_8")
ARMS = {"v2": "top5_h2h_v2_agent.py", "v3": "cash_v3_agent.py"}


def economy(sim, observer, seat, tree_income):
    events = [e for e in observer.events if e["event"] == "deposit" and e["player"] == seat]
    steps = sorted({e["step"] for e in events})
    workers = [w for w in sim.workers.values() if w.player == seat]
    return {
        "cash": float(sim.cash[seat]), "workers": len(workers),
        "trees": len(sim.player_trees[seat]), "trees_planted": sim.tree_counters[seat],
        "tree_income": tree_income[seat], "onboard": sum(w.carbon for w in workers),
        "positive_own_delivery_events": len(events), "positive_own_delivery_steps": steps,
        "positive_own_delivery_carbon": float(observer.d[seat]["own_positive_delivery_carbon"]),
        "base_collision_credit": float(observer.d[seat]["base_collision_credit"]),
    }


def play(candidate, opponent, seed, side):
    random.seed(7919 * seed + 104729 * side + 31)
    np.random.seed(seed)
    sim = audit.FastCarbon(audit.CFG).reset(seed=seed)
    agents = [None, None]
    agents[side], agents[1-side] = candidate.agent, opponent.make_agent()
    observer = audit.Observer()
    extinct = [False, False]
    tree_income = [0., 0.]
    for t in range(1, 101):
        actions = [agent(sim.obs(player), sim.config) for player, agent in enumerate(agents)]
        for player, commands in enumerate(actions):
            owned = set(sim.player_workers[player]) | {base for base, _ in sim.bases[player]}
            if not isinstance(commands, dict) or set(commands) - owned:
                raise ValueError("agent returned malformed or foreign command")
        observer.step(sim, actions)
        post = sim.get_state()
        for tree_id, tree in post["trees"].items():
            tree_income[tree["player"]] += post["trees_dict"][tree_id][1]
        extinct = [audit._extinct(post, p) for p in (0, 1)]
        if any(extinct):
            break
    result = [economy(sim, observer, p, tree_income) for p in (0, 1)]
    return {"map": seed, "side": side, "opponent": opponent.__name__, "end_step": sim.step_count,
            "extinct_by_seat": extinct, "student": result[side], "opponent_economy": result[1-side],
            "final_state_sha256": audit.hashed(post),
            "cash_margin_at_t100": result[side]["cash"] - result[1-side]["cash"] if t == 100 else None}


def summarize(rows):
    reached = [r for r in rows if r["end_step"] == 100]
    return {
        "games": len(rows), "reached_t100": len(reached),
        "own_extinctions": sum(r["extinct_by_seat"][r["side"]] for r in rows),
        "opponent_extinctions": sum(r["extinct_by_seat"][1-r["side"]] for r in rows),
        "mean_recorded_delivery_carbon": statistics.mean(r["student"]["positive_own_delivery_carbon"] for r in rows),
        "games_with_delivery": sum(r["student"]["positive_own_delivery_events"] > 0 for r in rows),
        "games_with_delivery_on_two_or_more_turns": sum(len(r["student"]["positive_own_delivery_steps"]) >= 2 for r in rows),
        "mean_t100_student_cash": statistics.mean(r["student"]["cash"] for r in reached) if reached else None,
        "mean_t100_cash_margin": statistics.mean(r["cash_margin_at_t100"] for r in reached) if reached else None,
        "mean_recorded_tree_income": statistics.mean(r["student"]["tree_income"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a fresh result path")
    pack = ROOT / "data/weights/sapling_1200k" / ARMS[args.arm]
    paths = [ROOT / "code" / (name + ".py") for name in OPPONENTS]
    sources = [Path(__file__), Path(audit.__file__), Path(audit.shared.__file__), pack,
               ROOT / "code/fastsim.py", ROOT / "code/eval_v53.py", *paths]
    hashes = {str(p.relative_to(ROOT)): audit.digest(p) for p in sources}
    candidate = audit.shared.module(pack, "student")
    games = []
    for path in paths:
        opponent = audit.shared.module(path, path.stem)
        for seed in SEEDS:
            for side in (0, 1):
                games.append(play(candidate, opponent, seed, side))
    report = dict(schema="sapling-cash-scripted-economy/1", arm=args.arm, pack_sha256=audit.digest(pack),
                  sources_sha256=hashes, python_hash_seed=os.getenv("PYTHONHASHSEED"),
                  scope="Fixed four maps, two seats and two deterministic scripted opponents per map. "
                        "State at 100 actual joint transitions or at earlier extinction. Cash at t100 "
                        "is left unavailable if extinction happened before t100. Separate from the external examiner.",
                  games=games, summary=summarize(games),
                  by_opponent={name:summarize([r for r in games if r["opponent"]==name]) for name in OPPONENTS})
    if any(audit.digest(p) != hashes[str(p.relative_to(ROOT))] for p in sources):
        raise RuntimeError("source changed during evaluation")
    audit.shared.atomic_json(args.out, report)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
