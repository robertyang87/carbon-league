"""Compute separate tactical readouts from the same verified replay pass."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import importlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
ANALYSES = ("recruitment", "harvest", "planting", "cargo_defense", "endgame")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def extract(task):
    names, key, match, expected, ranks, path = task
    raw = path.read_bytes()
    replay = json.loads(gzip.decompress(raw))
    if digest(raw) != expected["sha256"]:
        canonical = json.dumps(replay, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if digest(canonical.encode()) != expected["canonical_sha256"]:
            raise ValueError(f"Replay mismatch: {key}")
    if replay["rewards"] != [match["scores"][s] for s in match["order"]]:
        raise ValueError(f"Score/seat mismatch: {key}")
    frames = replay["steps"]
    if any(frame[0]["observation"]["step"] != t for t, frame in enumerate(frames)):
        raise ValueError(f"Noncontiguous replay: {key}")
    results = []
    for seat, side in enumerate(match["order"]):
        model = match["name_" + side]
        if model in ranks:
            results.append((key, seat, model, {
                name: importlib.import_module(name).observe(replay, seat, key) for name in names}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--analyses", nargs="+", choices=ANALYSES, default=ANALYSES)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--limit", type=int, help="Use a prefix of sorted match IDs for a debugging run")
    args = parser.parse_args()
    if args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("workers and sample limits must be positive")
    if args.limit is not None and args.out_dir.resolve() == HERE:
        parser.error("Use a separate --out-dir for partial runs")

    inventory = json.loads((ARCHIVE / "top5_inventory.json").read_bytes())
    raw_scores = (ARCHIVE / "top5_score_rows.json").read_bytes()
    scores = json.loads(raw_scores)
    if digest(raw_scores) != inventory["score_rows_sha256"] or set(scores) != set(inventory["replays"]):
        raise ValueError("Score inventory mismatch")
    ranks = tuple(r["name"] for r in inventory["ranks"])
    sources = {n: digest((HERE / f"{n}.py").read_bytes()) for n in args.analyses}
    keys = sorted(scores)[:args.limit]
    tasks = ((args.analyses, key, scores[key], inventory["replays"][key], ranks,
              args.replays / (key + ".json.gz")) for key in keys)
    by_module = {n: [] for n in args.analyses}
    with ProcessPoolExecutor(args.workers) as pool:
        for group in pool.map(extract, tasks, chunksize=3):
            for key, seat, model, observations in group:
                for name, observation in observations.items():
                    by_module[name].append(dict(key=key, seat=seat, model=model, observation=observation))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in by_module.items():
        if args.limit is None and len(rows) != inventory["appearances"]:
            raise ValueError(f"Incomplete {name} appearance set")
        if sources[name] != digest((HERE / f"{name}.py").read_bytes()):
            raise ValueError(f"Source changed during scan: {name}")
        report = {"score_sha256": digest(raw_scores), "script_sha256": sources[name],
                  "matches": len(keys), "appearances": len(rows), "ranks": ranks,
                  "scope": "Descriptive census of the pinned heavy-ladder snapshot. Same-game decisions are correlated.",
                  "analysis": importlib.import_module(name).summarize(rows)}
        (args.out_dir / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
