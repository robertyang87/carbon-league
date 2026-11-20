"""Reconcile registered development matches and look back from their largest cash swings."""
import argparse
import importlib.util
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
BASE = STUDY / "cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original"
spec = importlib.util.spec_from_file_location("original_cash_accounting", BASE / "economy.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)
LOOKBACKS = (3, 5, 10)


def trace_pivots(path, turn_books):
    """Reference each executed joint order once within its game; both policies have independent paths."""
    result = {}
    for label, (start, end) in ledger.swing_windows(path).items():
        prior = {n: list(range(max(0, start - n), start)) for n in LOOKBACKS}
        turns = list(range(max(0, start - max(LOOKBACKS)), end))
        for book in turn_books.values():
            book.update(turns)
        result[label] = dict(start_state=start, end_state=end, delta=round(path[end]-path[start], 2),
                             comparison_turns=list(range(start, end)), antecedent_turns=prior)
    return result


def turn_lookup(game):
    rows = game.pop("transitions")
    used = game.pop("_used_turns")
    return {t: rows[t] for t in sorted(used)}


def analyze(hosted, plan):
    settlement = hosted / "settlement.json"
    doc = json.loads(settlement.read_bytes())
    arms = plan["arms"]
    seeds = plan["seeds"]
    keys = {(seed, seat, arm) for seed in seeds for seat in (0, 1) for arm in arms}
    ledger.require(len(keys) == len(doc["games"]) == 12 and doc["records_reconciled"], "Incomplete settlement")
    cards = {(g["seed"], g["seat"], g["arm"]): g for g in doc["games"]}
    ledger.require(set(cards) == keys, "Incorrect seed/seat/arm roster")
    ledger.require(len(doc["archives"]) == 4, "Expected four paired host receipts")
    cfg = ledger.config()
    games = {}
    for block in doc["archives"]:
        path = hosted / (str(block["id"]) + ".zip")
        ledger.require(ledger.sha(path.read_bytes()) == block["sha256"], "Archive bytes differ")
        seed, seat = block["seed"], block["seat"]
        with zipfile.ZipFile(path) as archive:
            for arm in arms:
                key = (seed, seat, arm)
                card = cards[key]
                ledger.require(key not in games and card["terrain"] == plan["physical_boards"][str(seed)],
                               "Duplicate game or different registered terrain")
                game = ledger.replay(card, archive, cfg)
                if not card["official_transitions"] and card["status"] != "complete":
                    # No action reached the engine: preserve the failure without imputing any income.
                    game["observed_flow"] = False
                else:
                    game["observed_flow"] = True
                used = set()
                game["turn_references"] = trace_pivots(game["cash_margin_by_state"], {arm: used})
                game["_used_turns"] = used
                games[key] = game
    ledger.require(set(games) == keys, "Missing archive")
    comparisons = {}
    for trial, control in ((arms[1], arms[0]), (arms[2], arms[1]), (arms[2], arms[0])):
        pair_name = trial + " minus " + control
        pairs = []
        for seed in seeds:
            for seat in (0, 1):
                a, b = (games[seed, seat, arm] for arm in (trial, control))
                pair = ledger.comparison(a, b)
                relative_path = [x-y for x, y in zip(a["cash_margin_by_state"], b["cash_margin_by_state"])]
                pair["turn_references"] = trace_pivots(relative_path, {a["arm"]: a["_used_turns"], b["arm"]: b["_used_turns"]})
                pairs.append(pair)
        maps = {str(seed): ledger.summarize([p for p in pairs if p["seed"] == seed])["all"]
                for seed in seeds}
        comparisons[pair_name] = dict(map_seats=pairs, by_map=maps, **ledger.summarize(pairs))
    for game in games.values():
        game["selected_turns"] = turn_lookup(game)
        if not game["observed_flow"]:
            for field in ("signed_account_by_seat", "physical_cash_by_seat", "cash_margin_by_state",
                          "phases", "end_assets_by_physical_seat", "last30_tree_and_cargo", "opponent_lead"):
                game[field] = None
    return dict(run_id=doc.get("workflow_run", doc.get("run_id")), commit=doc["commit"],
                settlement_sha256=ledger.sha(settlement.read_bytes()),
                plan_sha256=doc.get("protocol_sha256", doc.get("plan_sha256")), analyzer_sha256=ledger.sha(Path(__file__).read_bytes()),
                replay_source_sha256=ledger.sha((BASE/"economy.py").read_bytes()),
                environment_config_sha256=ledger.CFG_HASH,
                semantics=dict(margin="Learner official score minus original opponent; punitive scoring kept separate from cash.",
                               physical_order="All two-entry arrays use physical seats 0/1.",
                               windows="Largest realized advance and decline over 1 or 10 joint transitions, earliest tie in cents. Antecedents are the prior 3/5/10 transitions, clipped at opening. Relative paths stop at the last common state.",
                               actions="For each referenced game and turn, selected_turns contains both actual order dictionaries, before-state workers [position,cargo,is_collector,physical_seat,distance_to_own_base], cash and tree counts, named cash events, ownership transitions and realized collisions. Late-settlement tree purchases are followed to expiry, seizure, or horizon.",
                               comparison="Differences are realized accounts on independently responding trajectories. These comparisons cannot assign a causal effect to a particular antecedent order."),
                games=[games[key] for key in sorted(games)], comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Replay the already-settled historical 12-game fixture.")
    args = parser.parse_args()
    if args.smoke:
        hosted = BASE / "hosted"
        registration = json.loads((BASE / "protocol.json").read_bytes())
        old = json.loads((hosted / "settlement.json").read_bytes())
        arms = ["greedy_parent", "sampled_parent", "sampled_child"]
        plan = dict(arms=arms, seeds=sorted({g["seed"] for g in old["games"]}),
                    physical_boards={str(g["seed"]): g["terrain"] for g in old["games"]})
        ledger.require(set(registration["seeds"]) == set(plan["seeds"]), "Historical fixture identity differs")
    else:
        hosted = HERE / "hosted"
        frozen = HERE / "protocol.json"
        plan = json.loads(frozen.read_bytes())
        settled = json.loads((hosted / "settlement.json").read_bytes())
        ledger.require(settled["protocol_sha256"] == ledger.sha(frozen.read_bytes()), "Settlement for a different protocol")
        ledger.require(plan["arms"] == ["parent", "bc", "joint"] and plan["seeds"] == [621758500, 621758501],
                       "Unexpected development registration")
    result = analyze(hosted, plan)
    if args.smoke:
        old_means = old["descriptive_contrasts"]
        for key, data in result["comparisons"].items():
            ledger.check_equal(data["all"]["mean_over_all_planned"], old_means[key]["mean_over_fixed_maps"], key)
    else:
        for key, data in result["comparisons"].items():
            registered = settled["results"]["paired"][key.replace(" minus ", "_minus_")]
            actual = {(p["seed"], p["seat"]): p.get("official_margin_delta") for p in data["map_seats"]}
            for ticket in registered["tickets"]:
                value = actual[ticket["seed"], ticket["seat"]]
                reference = ticket["paired_margin_change"]
                if value is None or reference is None:
                    ledger.require(value is reference, "Missing-score coverage differs")
                else:
                    ledger.check_equal(value, reference, "Same seat official contrast")
        ledger.write(hosted / "economy.json.gz", result, compressed=True)
    for key, row in result["comparisons"].items():
        print(key, row["all"].get("mean_over_all_planned"),
              {s: v.get("mean_over_all_planned") for s, v in row["by_map"].items()})


if __name__ == "__main__":
    main()
