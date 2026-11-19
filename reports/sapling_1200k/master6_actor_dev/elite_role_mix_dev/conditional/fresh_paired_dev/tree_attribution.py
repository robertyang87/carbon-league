"""Reconcile the two players' tree revenue and factor absorption into ownership and yield."""
from collections import Counter
from pathlib import Path
import math
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import responses as replay


def measure(row):
    seat = row["seat"]
    raw_path = HERE / "games" / row["arm"] / f"raw_{row['seed']}_{seat}.json.gz"
    assert replay.sha(raw_path) == row["raw_sha256"]
    tape = replay.load(raw_path)
    sim = replay.FastCarbon(tape["game_configuration"]).reset(seed=row["seed"])
    sim.cash = replay.CashEvents(sim.cash)
    accounts = [Counter(), Counter()]
    for commands in tape["actions"]:
        previous_ids = set(sim.trees)
        sim.cash.events.clear()
        sim.step(*commands)
        for tree_id, tree in sim.trees.items():
            if tree_id in previous_ids:
                accounts[tree.player]["eligible_tree_turns"] += 1
        for kind, player, amount, *detail in sim.cash.events:
            account = accounts[player]
            if kind in ("plant", "seize"):
                account[kind + "_payments"] += 1
                account[kind + "_cost"] -= amount
            elif kind == "tree_absorption":
                account["absorption"] += amount
                account["freight_tax"] += detail[-1]
            elif kind == "tree_collision_cash":
                account["collision_cash"] += amount
    assert list(sim.cash) == row["score_by_seat"]
    for account in accounts:
        account["ground"] = account["absorption"] - account["freight_tax"]
        account["per_eligible_turn"] = account["absorption"] / account["eligible_tree_turns"]
    return {"own": accounts[seat], "opponent": accounts[1-seat]}


def compare(parent, candidate):
    terms = Counter()
    for name, sign in (("own", 1), ("opponent", -1)):
        old, new = parent[name], candidate[name]
        turns = "eligible_tree_turns"
        rate = "per_eligible_turn"
        terms["ownership_turns"] += sign * (new[turns]-old[turns]) * (new[rate]+old[rate])/2
        terms["yield_per_turn"] += sign * (new[rate]-old[rate]) * (new[turns]+old[turns])/2
        for key in ("absorption", "ground", "freight_tax", "collision_cash", "plant_cost", "seize_cost"):
            terms[key] += sign * (new[key] - old[key])
    assert math.isclose(terms["ownership_turns"] + terms["yield_per_turn"], terms["absorption"], abs_tol=1e-7)
    assert math.isclose(terms["ground"] + terms["freight_tax"], terms["absorption"], abs_tol=1e-7)
    return terms


if __name__ == "__main__":
    accounts = {(row["arm"], row["seat"]): measure(row) for row in replay.load(HERE / "scores.json")}
    result = {}
    for seat in (0, 1):
        parent, candidate = accounts["parent", seat], accounts["candidate", seat]
        result[str(seat)] = dict(parent=parent, candidate=candidate, net_change=compare(parent, candidate))
    replay.write(HERE / "tree_attribution.json", result)
