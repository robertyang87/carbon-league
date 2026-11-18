"""Measure adjacent-base return commands from public Top 2 replays.

The feature side of the probe only uses current worker locations, own cargo, and
own bases. Its label is the command carried by the following replay frame.
The optional newer cohort takes its hashes from the corresponding opening receipt.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

from elite_defense_evidence import MODELS, REVIEW, REV, moves, read_replay

NEW_FIXTURE_REV = "88b4b108a9d90cc817b9112fca2ed21d572acb42"
BANDS = ("empty", "light_0_30", "medium_30_100", "heavy_ge100")


def band(cargo):
    if cargo < 0:
        raise ValueError(f"Negative own cargo: {cargo}")
    if cargo == 0:
        return BANDS[0]
    if cargo < 30:
        return BANDS[1]
    if cargo < 100:
        return BANDS[2]
    return BANDS[3]


def observe(replay, seat):
    directions = moves(replay["configuration"]["size"])
    counts = {name: Counter() for name in BANDS}
    steps = replay["steps"]
    for tick, (current, following) in enumerate(zip(steps, steps[1:])):
        obs = current[0]["observation"]
        if obs["step"] != tick or following[0]["observation"]["step"] != tick + 1:
            raise ValueError(f"Unexpected replay step at {tick}")
        own, enemy = obs["players"][seat], obs["players"][1 - seat]
        bases = set(own[1].values())
        own_cells = {v[0] for v in own[2].values()}
        enemy_cells = {v[0] for v in enemy[2].values()}
        clear_bases = {b for b in bases if b not in own_cells and b not in enemy_cells
                       and enemy_cells.isdisjoint(directions[b].values())}
        commands = following[seat]["action"] or {}
        for worker, (cell, cargo, role) in own[2].items():
            if role != "COLLECTOR" or cell in bases:
                continue
            adjacent = bases.intersection(directions[cell].values())
            if len(adjacent) != 1:
                continue
            home = next(iter(adjacent))
            if home not in clear_bases:
                continue
            dest = directions[cell][commands[worker]] if commands.get(worker) else cell
            record = counts[band(cargo)]
            record["opportunities"] += 1
            record["moves_into_base"] += int(dest == home)
    return counts


def extract(task):
    row, match, cache, split = task
    replay = read_replay(row, cache)
    if replay["rewards"] != [match["scores"][side] for side in match["order"]]:
        raise ValueError(f"Rewards differ from official seat assignment: {row['key']}")
    result = []
    for seat, side in enumerate(match["order"]):
        model = MODELS.get(match["name_" + side])
        if model:
            result.append({"key": row["key"], "seat": seat, "model": model,
                           "split": split, "bands": observe(replay, seat)})
    return result


def summarize(games):
    out = {"appearances": len(games)}
    for name in BANDS:
        records = [game["bands"][name] for game in games if game["bands"][name]["opportunities"]]
        eligible = sum(r["opportunities"] for r in records)
        entered = sum(r["moves_into_base"] for r in records)
        rates = [r["moves_into_base"] / r["opportunities"] for r in records]
        out[name] = {"games": len(records), "opportunities": eligible, "moves_into_base": entered,
                     "pooled_rate": entered / eligible if eligible else None,
                     "mean_game_rate": statistics.mean(rates) if rates else None}
    differences = []
    for game in games:
        hi = game["bands"]["heavy_ge100"]
        lo = game["bands"]["light_0_30"]
        if hi["opportunities"] and lo["opportunities"]:
            differences.append(hi["moves_into_base"] / hi["opportunities"] -
                               lo["moves_into_base"] / lo["opportunities"])
    pair = {"games": len(differences), "positive": sum(d > 0 for d in differences),
            "mean": statistics.mean(differences) if differences else None}
    # The nine new games are too few for useful interval estimation by model.
    if len(differences) >= 10:
        rng = random.Random(86396)
        means = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
        pair["game_bootstrap_95pct"] = [means[49], means[1949]]
    out["within_game_heavy_minus_light"] = pair
    return out


def newer_tasks(manifest, cache, frozen):
    records = json.loads(manifest.read_text())["records"]
    cutoff = max(game["created"] for game in frozen.values())
    fixtures = {}
    for model in MODELS.values():
        path = f"data/{model.lower()}_matches.json"
        data = subprocess.check_output(["git", "-C", str(REVIEW), "show", f"{NEW_FIXTURE_REV}:{path}"])
        for match in json.loads(data):
            key = f"{match['id']}-{match['game']}"
            if key in fixtures and match != fixtures[key]:
                raise ValueError(f"Conflicting new score rows for {key}")
            fixtures[key] = match
    tasks = []
    seen = set()
    for record in records:
        key = record["key"]
        row = {"key": key, "source": record["url"], "sha256": record["gzip_sha256"]}
        match = fixtures[key]
        if match != record["source_match"]:
            raise ValueError(f"New fixture differs from frozen score row: {key}")
        if key in frozen or key in seen or match["state"] != "done" or match["created"] <= cutoff:
            raise ValueError(f"Invalid chronological holdout: {key}")
        seen.add(key)
        tasks.append((row, match, cache, "newer"))
    if len(tasks) != 9:
        raise ValueError("Expected nine newly archived games")
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", required=True, type=Path, help="92 frozen public replays")
    parser.add_argument("--new-manifest", type=Path, help="recruit_probe.json opening receipt for the new games")
    parser.add_argument("--new-public-dir", type=Path, help="Nine chronological confirmation replays")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.new_manifest) != bool(args.new_public_dir):
        parser.error("Supply both new-cohort arguments or neither")
    if args.workers < 1:
        parser.error("workers must be positive")
    with (REVIEW / "all_games.csv").open() as source:
        rows = list(csv.DictReader(source))
    official = json.loads((REVIEW / "inputs" / "official_matches.json").read_text())
    if len(rows) != 352 or {row['key'] for row in rows} != set(official) or len(official) != 352:
        raise ValueError("Unexpected frozen replay census")
    tasks = []
    for row in rows:
        match = official[row["key"]]
        digest = hashlib.sha256(match["id"].encode("ascii")).digest()
        split = "heldout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "discovery"
        tasks.append((row, match, args.public_dir, split))
    if args.new_manifest:
        tasks.extend(newer_tasks(args.new_manifest, args.new_public_dir, official))
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        appearances = [a for group in workers.map(extract, tasks, chunksize=3) for a in group]
    if len(appearances) != 354 + (9 if args.new_manifest else 0):
        raise ValueError("Unexpected Top 2 appearance count")
    groups = sorted({(row["split"], row["model"]) for row in appearances})
    summary = {f"{split}/{model}": summarize([r for r in appearances
               if (r["split"], r["model"]) == (split, model)]) for split,model in groups}
    result = {"frozen_replay_revision": REV, "new_score_index_revision": NEW_FIXTURE_REV,
              "frozen_games": len(official), "new_games": len(tasks) - len(official),
              "new_sources": [row for row, _, _, split in tasks if split == "newer"],
              "eligibility": "Own collector off base and exactly one step from one own base. "
                             "No worker currently on that base and no enemy worker on its four neighbours.",
              "label": "Command into the base on following transition, regardless of later collision or delivery.",
              "bands": "0; (0,30); [30,100); [100,+inf). 30 is base recruitment cost, 100 is default cell cap. "
                       "These are descriptive bins, not calibrated policy thresholds.",
              "split": "sha256 of ASCII match ID, first eight bytes unsigned big-endian modulo 5. "
                       "Remainder 0 is held out. Both players in a direct match share the split. "
                       "All newer games started after the frozen census maximum timestamp.",
              "scope": "Observed behavior. Per-game differences pair the heavy and light exposure buckets within "
                       "each game; a return command is not a guaranteed deposit or a causal gain in utility.",
              "summary": summary, "by_game": appearances}
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
