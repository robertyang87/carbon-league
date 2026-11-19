"""Compare audited candidates on every seat of the two capacity-development maps."""
from pathlib import Path
from statistics import mean
import argparse
import hashlib
import json

from tree_win_gate import validate_game

MAPS = (620106811, 620106812)


def read_run(directory):
    report = json.loads((directory / "results.json").read_bytes())
    header = report["header"]
    if report["status"] != "complete" or header["examiner_md5"] != "57f9a594cce4f2ec9e3573d8d9d26ca1":
        raise ValueError("completed, source-audited Master3s2 games required")
    rows = {}
    for game in report["games"]:
        validate_game(game)
        key = (game["seed"], game["candidate_seat"])
        if key in rows or hashlib.sha256((directory/game["raw_file"]).read_bytes()).hexdigest() != game["raw_sha256"]:
            raise ValueError("duplicate seat or changed transcript")
        rows[key] = game
    if set(rows) != {(seed, seat) for seed in MAPS for seat in (0,1)}:
        raise ValueError("exactly two seats per preselected map required")
    return rows, header


def summarize(rows):
    games = list(rows.values())
    sides = (0, 1)
    return {
        "wins_by_seat": [sum(g["result"] == "win" and g["candidate_seat"] == seat for g in games) for seat in sides],
        "draws": sum(g["result"] == "draw" for g in games),
        "mean_official_margin": mean(g["margin"] for g in games),
        "mean_official_own_score": mean(g["official_score_by_seat"][g["candidate_seat"]] for g in games),
        "mean_official_opponent_score": mean(g["official_score_by_seat"][1-g["candidate_seat"]] for g in games),
        "mean_deposit": mean(g["final"]["candidate"]["own_positive_delivery_carbon"] for g in games),
        "mean_tree_income": mean(g["final"]["candidate"]["cumulative_tree_absorption_income"] for g in games),
        "time_limit_games": sum(g["termination"] == "official_episode_limit" for g in games),
        "opponent_search_calls": sum(g["search_invocations"] for g in games),
        "opponent_completed_comparisons": sum(g["search"]["searched"] for g in games),
        "opponent_depth8": sum(g["search"]["depth"].get("8",0) for g in games),
    }


def compare(small, large):
    a, ha = read_run(small)
    b, hb = read_run(large)
    if ha["pack_sha256"] == hb["pack_sha256"] or ha["configuration"] != hb["configuration"]:
        raise ValueError("different candidates under a common rules configuration required")
    per_seat = []
    for seed, seat in sorted(a):
        record = {"seed": seed, "seat": seat}
        for name, g in (("v5",a[seed,seat]),("capacity",b[seed,seat])):
            record[name] = {k:g[k] for k in ("result", "margin", "actual_transitions", "termination", "official_score_by_seat", "raw_sha256")}
        record["paired_delta_margin"] = b[seed,seat]["margin"] - a[seed,seat]["margin"]
        per_seat.append(record)
    return {"scope":"Two development maps, swapped seats. Original examiner samples independently under a wall-clock search budget.",
            "v5_pack_sha256":ha["pack_sha256"], "capacity_pack_sha256":hb["pack_sha256"],
            "v5":summarize(a), "capacity":summarize(b), "paired":per_seat,
            "paired_delta_margin_by_map":{str(seed):mean(r["paired_delta_margin"] for r in per_seat if r["seed"]==seed) for seed in MAPS}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v5", type=Path)
    parser.add_argument("capacity", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.v5, args.capacity)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
