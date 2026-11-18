"""Audit recruitment for all five exact ranked names on available validated public replays.

Inputs are the archived Top-5 inventory, its scored match table and replay cache. Every model in a top-five duel is read at its actual seat. The aggregate
sum counts appearances; coverage separately counts unique matches. Missing bytes
are reported by match, never replaced by zeros.

  python top5_recruit.py \
    --replays /path/to/replays --out /tmp/top5_recruit.json
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path

from recruit_probe import BANDS, action_counts, digest, identity, key_of, validate

RANKED = (
    "claude-AlphaCarbon-Master5s",
    "claude-AlphaCarbon-Master6",
    "RyanX-AIPowercore-it720-preview",
    "RyanX-AIPowercore-M9XJ-630",
    "ClaudeFable5.2-v7",
)


def audit(task):
    key, row, replays, checksum = task
    path = replays / f"{key}.json.gz"
    expected = identity(row)
    appearances = [{"name": name, "seat": side} for side, (name, _) in enumerate(expected)
                   if name in RANKED]
    if key_of(row) != key or not appearances:
        raise ValueError(f"{key}: invalid score identity")
    result = {"key": key, "appearances": appearances}
    if not path.is_file():
        result["status"] = "missing"
        return result
    packed = path.read_bytes()
    env = json.loads(gzip.decompress(packed))
    actual_sha = digest(packed)
    if actual_sha != checksum["sha256"]:
        # The same decoded JSON may have a different gzip level or timestamp.
        canonical = json.dumps(env, ensure_ascii=False, separators=(",", ":"),
                               allow_nan=False).encode()
        if digest(canonical) != checksum["canonical_sha256"]:
            raise ValueError(f"{key}: replay differs from archived inventory")
    validate(env, row)
    for side in appearances:
        side.update(action_counts(env, side["seat"], key))
    result.update(status="validated", gzip_sha256=checksum["sha256"],
                  termination_status=env["statuses"])
    return result


def summarize(games, name):
    appearances = [p for g in games for p in g["appearances"] if p["name"] == name]
    available = [a for a in appearances if "opening" in a]
    bands = {}
    for lo, hi in BANDS:
        key = f"{lo}-{hi - 1}"
        cells = [a["bands"][key] for a in available]
        bands[key] = {
            "any_recruit": sum(c[0] for c in cells),
            "opportunities": sum(c[1] for c in cells),
            "games_with_opportunities": sum(c[1] > 0 for c in cells),
        }
    return {
        "scored_appearances": len(appearances), "validated_appearances": len(available),
        "by_seat": {str(s): sum(a["seat"] == s for a in available) for s in (0, 1)},
        "ordered_openings": dict(sorted(Counter(a["opening"] for a in available).items())),
        "openings_by_seat": {
            str(seat): dict(sorted(Counter(a["opening"] for a in available
                                          if a["seat"] == seat).items()))
            for seat in (0, 1)
        },
        "bands": bands,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    folder = Path(__file__).resolve().parent
    parser.add_argument("--scores", type=Path, default=folder / "top5_score_rows.json")
    parser.add_argument("--inventory", type=Path, default=folder / "top5_inventory.json")
    parser.add_argument("--replays", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    data = args.scores.read_bytes()
    rows = json.loads(data)
    inventory = json.loads(args.inventory.read_bytes())
    if digest(data) != inventory["score_rows_sha256"]:
        raise ValueError("score rows differ from archived inventory")
    ranks = tuple(r["name"] for r in inventory["ranks"])
    if ranks != RANKED or inventory["replays"].keys() != rows.keys():
        raise ValueError("ranked names or match list differ from inventory")
    with ProcessPoolExecutor(args.workers) as pool:
        games = list(pool.map(audit, [(k, r, args.replays, inventory["replays"][k])
                                     for k, r in sorted(rows.items())],
                              chunksize=4))
    result = {
        "source_score_table_sha256": digest(data),
        "leaderboard_http_date": inventory["leaderboard_http_date"],
        "unidentified_result": inventory["leaderboard_vs_score_exception"],
        "formula": "Each game is checked against archived score, both seats and contiguous "
                   "steps. At t=0 record the submitted actions of ordered base IDs 0,1. "
                   "C=collector, P=planter, -=no recruit. At t>0 count at least one "
                   "submitted recruit at a root with cash >=30 and workforce <10. "
                   "Grouped bands are inclusive; per appearance values are [any, eligible].",
        "distinct_matches": len(games), "by_status": dict(Counter(g["status"] for g in games)),
        "valid_status_pairs": dict(Counter("/".join(g["termination_status"]) for g in games
                                          if g["status"] == "validated")),
        "ranks": {name: summarize(games, name) for name in RANKED},
        "games": games,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
