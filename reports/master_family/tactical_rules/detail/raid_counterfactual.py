"""One-step alternative to a planter retreat against a real, recorded enemy response.

Example: python raid_counterfactual.py --replays /tmp/carbon_top5_games/replays --out /tmp/raid.json
The selected games, eligible population, original public-player parity, actions of
all units and tree IDs are reported. No search, training or longer-horizon inference.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
CODE = ARCHIVE.parents[2] / "code"
sys.path.insert(0, str(CODE))
from fastsim import FastCarbon
from planting import neighbors


def sha(data):
    return hashlib.sha256(data).hexdigest()


def select_games(scores, names):
    """Select by fixed hash before checking scores, board situations, or outcomes."""
    ordered = sorted(scores, key=lambda k: sha(("planter-raid-check/" + k).encode()))
    games = []
    for key in ordered:
        row = scores[key]
        participants = [row["name_" + tag] for tag in row["order"]]
        candidates = [i for i, name in enumerate(participants) if name in names]
        if len(candidates) == 1:
            seat = candidates[0]
            games.append((key, seat, participants[seat]))
        if len(games) == 50:
            break
    if len(games) != 50:
        raise ValueError("Expected 50 disjoint fixtures")
    return games


def simulate(cfg, state, actions):
    # The simulator alone reads the full board for exact one-step adjudication.
    sim = FastCarbon(cfg).load_observation(state)
    sim.step(*actions)
    return sim.full_observation()["players"]


def outcome(players, seat, worker, tree):
    return {"planter_alive": worker in players[seat][2],
            "seized_same_tree": tree in players[seat][3],
            "cash_by_seat": [player[0] for player in players]}


def compare_game(replay, key, seat, name):
    cfg = replay["configuration"]
    steps = replay["steps"]
    adjacent = neighbors(cfg["size"])
    mismatches = []
    for t, (frame, following) in enumerate(zip(steps, steps[1:])):
        state = frame[0]["observation"]
        ours, theirs = state["players"][seat], state["players"][1 - seat]
        if ours[0] < cfg["seizeCost"] or cfg["episodeSteps"] - 1 - t < 10:
            continue
        actions = [unit["action"] or {} for unit in following]
        tree_at = {p: (tree, age) for tree, (p, age) in theirs[3].items()}
        enemy_collectors = {uid: w[0] for uid, w in theirs[2].items() if w[2] == "COLLECTOR"}
        for uid, (pos, _, kind) in sorted(ours[2].items()):
            action = actions[seat].get(uid)
            if kind != "PLANTER" or pos not in tree_at or action not in adjacent[pos]:
                continue
            tree_id, age = tree_at[pos]
            if cfg["treeLifespan"] - age < 10:
                continue
            threat_cells = {pos, *adjacent[pos].values()}
            reachable = [e for e, p in enemy_collectors.items() if p in threat_cells]
            if not reachable:
                continue
            original = simulate(cfg, state, actions)
            expected = following[0]["observation"]["players"]
            if original != expected:
                mismatches.append({"t": t, "worker": uid})
                continue
            alternate = [dict(acts) for acts in actions]
            alternate[seat].pop(uid)  # No submitted move means stay at current tree.
            counterfactual = simulate(cfg, state, alternate)
            return {"key": key, "seat": seat, "model": name, "t": t, "worker": uid,
                    "tree_id": tree_id, "tree_age": age, "cell": pos, "cash_before": ours[0],
                    "enemy_collectors_in_reach": reachable, "submitted_actions": actions,
                    "recorded_move": action, "original_players_equal_recorded": original == expected,
                    "recorded": outcome(original, seat, uid, tree_id),
                    "force_stay": outcome(counterfactual, seat, uid, tree_id)}, mismatches
    return None, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_scores = (ARCHIVE / "top5_score_rows.json").read_bytes()
    scores = json.loads(raw_scores)
    inventory = json.loads((ARCHIVE / "top5_inventory.json").read_bytes())
    if sha(raw_scores) != inventory["score_rows_sha256"]:
        raise ValueError("Score ledger digest differs from the frozen corpus")
    names = {r["name"] for r in inventory["ranks"]}
    cases, coverage = [], {}
    for key, seat, name in select_games(scores, names):
        row = scores[key]
        raw = (args.replays / (key + ".json.gz")).read_bytes()
        replay = json.loads(gzip.decompress(raw))
        ident = inventory["replays"][key]
        if sha(raw) != ident["sha256"]:
            canonical = json.dumps(replay, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if sha(canonical.encode()) != ident["canonical_sha256"]:
                raise ValueError(f"Replay digest mismatch for {key}")
        if replay["rewards"] != [row["scores"][s] for s in row["order"]]:
            raise ValueError(f"Score or seat mismatch for {key}")
        case, skipped = compare_game(replay, key, seat, name)
        coverage[key] = {"seat": seat, "model": name, "found": case is not None,
                         "roots_with_baseline_mismatch_skipped": skipped}
        if case is not None:
            cases.append(case)
    summary = {"games_examined": len(coverage), "games_with_paired_root": len(cases),
               "mismatched_baselines_skipped": sum(len(g["roots_with_baseline_mismatch_skipped"]) for g in coverage.values()),
               "recorded_move_planter_deaths": sum(not c["recorded"]["planter_alive"] for c in cases),
               "force_stay_planter_deaths": sum(not c["force_stay"]["planter_alive"] for c in cases),
               "force_stay_steals": sum(c["force_stay"]["seized_same_tree"] for c in cases),
               "move_survives_stay_dies": sum(c["recorded"]["planter_alive"] and not c["force_stay"]["planter_alive"] for c in cases)}
    result = {"scope": "Conditional one-step counterfactual at retreat actions. Other recorded moves fixed; no replanning or terminal effect.",
              "selection": "First 50 single-target-model fixtures ordered by sha256('planter-raid-check/' + id-game). Each contributes at most its first eligible decision with exact next-state public players under historical actions; within step workers sorted by ID.",
              "eligibility": "Funded planter standing on an enemy tree with at least 10 tree and game steps left; opposing collector reaches current cell by staying/moving; planter actually moved. Full simulator carbon enters adjudication only.",
              "score_sha256": sha(raw_scores), "script_sha256": sha(Path(__file__).read_bytes()),
              "fastsim_sha256": sha((CODE / "fastsim.py").read_bytes()),
              "recorded_baseline": "Both seats' next-frame players arrays (cash, bases, workers and trees) must exactly equal the historical simulation.",
              "summary": summary, "coverage": coverage, "cases": cases}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
