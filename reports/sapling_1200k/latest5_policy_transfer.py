"""Summarize frozen policy predictions at the map, player, seat and role level."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize(observations):
    slots = sum(item[0] for item in observations)
    parent = sum(item[1] for item in observations)
    student = sum(item[2] for item in observations)
    return {"slots": slots, "parent_correct": parent, "student_correct": student,
            "change_percentage_points": 100 * (student - parent) / slots if slots else None}


def audit(predictions, manifest, manifest_digest):
    if predictions["inventory_sha256"] != manifest_digest or predictions["excluded"]:
        raise ValueError("predictions do not cover the frozen inventory")
    games = {g["id"]: g for g in manifest["games"]}
    if len(games) != len(manifest["games"]) or len({g["seed"] for g in games.values()}) != len(games):
        raise ValueError("duplicate fixture or map")
    totals = []
    groups = {column: defaultdict(list) for column in ["role", "player", "seat", "map"]}
    seen = set()
    top_names = {rank["name"] for rank in manifest["rankings"]}
    for game in predictions["games"]:
        key = game["id"]
        source = games[key]
        if key in seen or source["seed"] != game["seed"] or source["sha256"] != game["packed_sha256"]:
            raise ValueError("duplicate or mismatched scored game")
        if source["seat_names"] != game["seat_names"] or source["cohort"] != game["cohort"]:
            raise ValueError("incorrect seat identities or cohort")
        expected_seats = {str(i) for i, name in enumerate(source["seat_names"]) if name in top_names}
        if set(game["by_seat"]) != expected_seats:
            raise ValueError("missing or unexpected ranked seat")
        seen.add(key)
        for seat, scores in game["by_seat"].items():
            player = game["seat_names"][int(seat)]
            for role, before in scores["cash_v3"].items():
                after = scores["full_top5_decoder"][role]
                n = before["n"]
                if n != after["n"] or not (0 <= before["correct"] <= n and 0 <= after["correct"] <= n):
                    raise ValueError("invalid action denominator")
                row = (n, before["correct"], after["correct"])
                totals.append(row)
                for column, category in [("role", role), ("player", player), ("seat", seat), ("map", key)]:
                    groups[column][category].append(row)
    if seen != set(games):
        raise ValueError("missing game predictions")
    summaries = {column: {k: summarize(v) for k, v in sorted(group.items())}
                 for column, group in groups.items()}
    changes = [row["change_percentage_points"] for row in summaries["map"].values()]
    return {"inventory_sha256": manifest_digest, "weights": predictions["checkpoint_sha256"],
            "top_player_count": len(top_names), "all": summarize(totals), **summaries,
            "equal_map_mean_change_pp": sum(changes) / len(changes),
            "map_directions": {"positive": sum(x > 0 for x in changes),
                               "zero": sum(x == 0 for x in changes),
                               "negative": sum(x < 0 for x in changes)},
            "scope": "Frozen models, historical action prefixes. Map and player summaries are descriptive, and do not estimate the return from playing either policy."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = args.inventory.read_bytes()
    results = audit(json.loads(args.predictions.read_text()), json.loads(data), hashlib.sha256(data).hexdigest())
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
