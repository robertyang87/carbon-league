"""Count scarce-capital situations in complete Top Five head-to-head replays."""
import argparse
from collections import Counter
import gzip
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reports/master_family/deep_review"))
from recover_public import key_of, validate

REV = "499a8555c16d21eb56a78de1b8db1a1362b58f60"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", nargs="+", type=Path, required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a fresh output")
    data = json.loads((HERE / "top5_h2h_data.json").read_text())
    groups = {g["id"]: g["split"] for g in data["games"]}
    frozen = {g["id"] for g in data["mutual_matches"]}
    roster = set(data["strategy_ids"])
    official = {}
    for tag in ("master5s", "master6", "it720preview", "m9xj630", "fablev7"):
        raw = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{REV}:data/{tag}_matches.json"])
        for row in json.loads(raw):
            if row.get("state") == "done" and row["name_a"] in roster and row["name_b"] in roster and row["name_a"] != row["name_b"]:
                official[key_of(row)] = row
    by_game = {}
    for root in args.replays:
        for path in sorted(root.glob("*.json.gz")):
            key = path.name.removesuffix(".json.gz")
            if key in by_game or key not in official:
                continue
            with gzip.open(path, "rt") as src:
                replay = json.load(src)
            validate(replay, official[key])
            group = groups.get(key, "prior_exposure" if key in frozen else "later")
            frames = replay["steps"]
            cfg = replay["configuration"]
            recruit_cost = min(cfg["recCollectorCost"], cfg["recPlanterCost"])
            counts = Counter()
            seats_at_risk = set()
            for t in range(10, len(frames) - 1):
                players = frames[t][0]["observation"]["players"]
                for seat, (cash, bases, workers, trees) in enumerate(players):
                    counts["seat_steps_from_t10"] += 1
                    if len(workers) == 1 and not trees and cash < recruit_cost:
                        counts["last_worker_no_tree_sub_recruit_cash"] += 1
                        seats_at_risk.add(seat)
            by_game[key] = dict(split=group, decisions=len(frames)-1, seats_at_risk=sorted(seats_at_risk), **counts)
    if set(by_game) != set(official) or len(by_game) != 108 or not frozen.issubset(by_game):
        raise ValueError(f"expected all 108 verified matches, found {len(by_game)}")
    totals = {}
    for group in ("train", "val", "prior_exposure", "later", "all"):
        rows = [r for r in by_game.values() if group == "all" or r["split"] == group]
        totals[group] = {
            "games": len(rows),
            "seat_steps_from_t10": sum(r.get("seat_steps_from_t10", 0) for r in rows),
            "at_risk_seat_steps": sum(r.get("last_worker_no_tree_sub_recruit_cash", 0) for r in rows),
            "games_with_risk": sum(bool(r["seats_at_risk"]) for r in rows),
            "games_reaching_t299": sum(r["decisions"] == 299 for r in rows),
        }
    result = dict(definition="Pre-action from t10, exactly one worker, no trees, cash below cheapest recruit. "
                  "A seat-step may belong to a long spell in one game. All counts are observed, "
                  "not an estimate of the effect of a different action.", groups=totals, per_game=by_game)
    with args.out.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
