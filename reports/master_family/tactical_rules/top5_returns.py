"""Evaluate the frozen adjacent-base return probe against a Top 5 match snapshot.

Example: python top5_returns.py --scores score_rows.json --public-dir replays --out top5_returns.json
Use --allow-partial to explicitly report missing replay IDs from an incomplete archive.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

from return_probe import observe, summarize

NAMES = ("claude-AlphaCarbon-Master5s", "claude-AlphaCarbon-Master6",
         "RyanX-AIPowercore-it720-preview", "RyanX-AIPowercore-M9XJ-630", "ClaudeFable5.2-v7")


def measure(task):
    match, path, expected = task
    data = path.read_bytes()
    replay = json.loads(gzip.decompress(data))
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected["sha256"]:
        canonical = json.dumps(replay, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != expected["canonical_sha256"]:
            raise ValueError(f"Replay differs from frozen inventory: {path.name}")
    if replay["rewards"] != [match["scores"][s] for s in match["order"]]:
        raise ValueError(f"Official scores differ for cache member {path.name}")
    key = f"{match['id']}-{match['game']}"
    sides = []
    for seat, side in enumerate(match["order"]):
        model = match["name_" + side]
        if model in NAMES:
            sides.append({"key": key, "seat": seat, "model": model, "bands": observe(replay, seat)})
    return key, digest, sides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path(__file__).with_name("top5_inventory.json"))
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--public-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    scores_raw = args.scores.read_bytes()
    matches = json.loads(scores_raw)
    inventory = json.loads(args.inventory.read_text())
    if inventory["score_rows_sha256"] != hashlib.sha256(scores_raw).hexdigest() or set(matches) != set(inventory["replays"]):
        raise ValueError("Score snapshot differs from archived inventory")
    expected = Counter()
    tasks, missing = [], []
    for key, game in sorted(matches.items()):
        if key != f"{game['id']}-{game['game']}" or game['state'] != 'done' or sorted(game['order']) != ['a','b']:
            raise ValueError(f"Incomplete or inconsistent source fixture: {key}")
        names = [game['name_' + s] for s in game['order'] if game['name_' + s] in NAMES]
        if not names:
            raise ValueError(f"Fixture does not involve a scoped player: {key}")
        expected.update(names)
        path = args.public_dir / (key + '.json.gz')
        if path.is_file():
            tasks.append((game, path, inventory["replays"][key]))
        else:
            missing.append(key)
    if missing and not args.allow_partial:
        parser.error(f"Missing {len(missing)} of {len(matches)} frozen replays, including {missing[0]}")
    if args.workers < 1 or not tasks:
        parser.error("Need a positive worker count and at least one replay")
    hashes, appearances = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        for key, digest, sides in workers.map(measure, tasks, chunksize=4):
            hashes[key] = digest
            appearances.extend(sides)
    result = {"scope": "All eligible nonterminal decisions in retrieved completed matches at this historical heavy-board snapshot. "
                       "Uses the clear one-step return definition and fixed cargo bands in return_probe.py. "
                       "The source corpus overlaps the previous Top 2 cohort. Repeated worker decisions are correlated.",
              "unique_indexed_games": len(matches), "unique_analyzed_games": len(hashes), "missing": missing,
              "score_rows_sha256": hashlib.sha256(scores_raw).hexdigest(), "gzip_sha256_by_key": hashes,
              "expected_appearances": dict(expected), "analyzed_appearances":len(appearances),
              "summary": {name: summarize([r for r in appearances if r["model"] == name]) for name in NAMES},
              "by_appearance": appearances}
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == '__main__':
    main()
