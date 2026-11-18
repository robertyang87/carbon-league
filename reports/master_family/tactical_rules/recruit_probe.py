"""Count base actions by budget and game phase in the Master5s/Master6 replays.

Run from the repository root. The old Git-backed replays need no checkout.
Recover old public files using deep_review/recover_public.py with its archived
deep_review/inputs/replay_sources.json. Recover the nine later public files with
the same program, using recruit_probe.json as --source-manifest and 88b4b108 as
--ref. For the probe, pass --public-cache for each of the two cache directories.
All reads are offline once the public replay caches exist.

Example:
  python reports/master_family/tactical_rules/recruit_probe.py \
    --public-cache /path/to/original-public --public-cache /path/to/later-public \
    --out /tmp/recruit_probe.json
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DEEP = HERE.parent / "deep_review"
sys.path.insert(0, str(DEEP))
from recover_public import identity, key_of, validate

ORIGINAL_REF = "37965ea578851b5f3ff88ffc42d7119ff716b5cc"
LATER_REF = "88b4b108a9d90cc817b9112fca2ed21d572acb42"
TARGETS = {
    "claude-AlphaCarbon-Master5s": "Master5s",
    "claude-AlphaCarbon-Master6": "Master6",
}
ACTION_NAMES = {"RECCOLLECTOR": "C", "RECPLANTER": "P", "NONE": "-"}
BANDS = [(1, 30), (30, 100), (100, 200), (200, 280), (280, 299)]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo, ref, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{path}"])


def action_counts(env, side, key):
    """Count only decisions at the player's two bases, in canonical base-ID order."""
    steps = env["steps"]
    cfg = env["configuration"]
    prices = [cfg["recCollectorCost"], cfg["recPlanterCost"]]
    increases = [cfg["recCollectorIncreaseCost"], cfg["recPlanterIncreaseCost"]]
    if prices != [30, 30] or increases != [0, 0] or cfg["workerLimit"] != 10:
        raise ValueError(f"{key}: unrecognised price or cap")

    bands = {f"{lo}-{hi - 1}": [0, 0] for lo, hi in BANDS}
    opening = None
    for t, (before, after) in enumerate(zip(steps, steps[1:])):
        cash, bases, workers, _ = before[0]["observation"]["players"][side]
        actions = after[side]["action"] or {}
        base_ids = sorted(bases)
        if len(base_ids) != 2 or [k.rsplit("-", 1)[-1] for k in base_ids] != ["0", "1"]:
            raise ValueError(f"{key}: unexpected base ids")
        purchases = [ACTION_NAMES[actions.get(base_id, "NONE")] for base_id in base_ids]
        if t == 0:
            if cash != 80 or workers:
                raise ValueError(f"{key}: nonstandard opening")
            opening = "+".join(purchases)
        elif cash >= 30 and len(workers) < cfg["workerLimit"]:
            band = next(f"{lo}-{hi - 1}" for lo, hi in BANDS if lo <= t < hi)
            bands[band][1] += 1
            bands[band][0] += any(action != "-" for action in purchases)
    return {"opening": opening, "bands": bands}


def checked_game(task):
    row, repo, cache_dirs = task
    key = row["key"]
    source = row["source"]
    if source.startswith("data/"):
        packed = git_bytes(repo, ORIGINAL_REF, source)
    else:
        candidates = [d / (key + ".json.gz") for d in cache_dirs]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"missing checked public replay {key}")
        packed = path.read_bytes()

    if row.get("expected_gzip") and digest(packed) != row["expected_gzip"]:
        raise ValueError(f"{key}: frozen replay digest mismatch")
    decoded = gzip.decompress(packed)
    env = json.loads(decoded)
    validate(env, row["match"])
    appearances = []
    for side, (name, _) in enumerate(identity(row["match"])):
        if name in TARGETS:
            appearances.append({
                "model": TARGETS[name], "seat": side, **action_counts(env, side, key)
            })
    game = {"key": key, "split": row["split"], "appearances": appearances}
    if row["split"] != "later":
        return game, None
    # Same trusted-source layout accepted by deep_review/recover_public.py.
    receipt = {
        "key": key, "url": source, "source_match": row["match"],
        "gzip_sha256": digest(packed), "replay_json_sha256": digest(decoded),
    }
    return game, receipt


def read_index(repo):
    index_path = DEEP / "all_games.csv"
    original = list(csv.DictReader(index_path.read_text().splitlines()))
    unique = {r["key"].split("-")[0] for r in original}
    if len(unique) != len(original) or len(original) != 352:
        raise ValueError("original census does not contain 352 distinct matches")
    metadata = {}
    for model in ("master5s", "master6"):
        for row in json.loads(git_bytes(repo, LATER_REF, f"data/{model}_matches.json")):
            if row["state"] != "done":
                continue
            key = key_of(row)
            if key in metadata and identity(metadata[key]) != identity(row):
                raise ValueError(f"{key}: score or seat mismatch across model manifests")
            metadata[key] = row
    original_keys = {r["key"] for r in original}
    if len(metadata) != 361 or not original_keys <= metadata.keys():
        raise ValueError("later manifests do not cover the original census plus nine matches")
    ordered = sorted(original, key=lambda r: (float(r["created"]), r["key"]))
    split_at = len(ordered) * 4 // 5
    rows = []
    for i, rec in enumerate(ordered):
        key = rec["key"]
        scored = metadata[key]
        seat = int(rec["seat"])
        names = [name for name, _ in identity(scored)]
        if names[seat] != rec["model_name"] or names[1 - seat] != rec["opponent"]:
            raise ValueError(f"{key}: focal side mismatch")
        rows.append({
            "key": key, "match": scored, "split": "earlier" if i < split_at else "last_fifth",
            "source": rec["source"], "expected_gzip": rec["sha256"],
        })
    last_original = max(float(r["created"]) for r in ordered)
    for key in sorted(metadata.keys() - original_keys):
        scored = metadata[key]
        if scored["created"] <= last_original:
            raise ValueError(f"{key}: extra match predates original cutoff")
        rows.append({
            "key": key, "match": scored, "split": "later",
            "source": f"https://117.72.210.134/replay/{scored['id']}/{scored['game']}",
        })
    return rows, digest(index_path.read_bytes()), last_original


def summarize(games):
    report = {}
    for model in TARGETS.values():
        report[model] = {}
        for split, allowed in {
            "original": {"earlier", "last_fifth"}, "earlier": {"earlier"},
            "last_fifth": {"last_fifth"}, "later": {"later"}
        }.items():
            selected = [p for g in games if g["split"] in allowed
                        for p in g["appearances"] if p["model"] == model]
            bands = {}
            for lo, hi in BANDS:
                key = f"{lo}-{hi - 1}"
                values = [p["bands"][key] for p in selected]
                bands[key] = {
                    "any_recruit": sum(v[0] for v in values),
                    "opportunities": sum(v[1] for v in values),
                    "appearances_with_opportunity": sum(v[1] > 0 for v in values)
                }
            report[model][split] = {
                "appearances": len(selected),
                "openings": dict(Counter(p["opening"] for p in selected)), "bands": bands,
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-cache", action="append", required=True, type=Path)
    parser.add_argument("--repo", default=HERE.parents[2], type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    rows, inventory_sha, last_original = read_index(args.repo)
    with ProcessPoolExecutor(args.workers) as pool:
        products = list(pool.map(checked_game, [(r, args.repo, args.public_cache) for r in rows],
                                 chunksize=2))
    games = [g for g, _ in products]
    output = {
        "scope": "Public Top-2 spectator replay audit. Observational action counts.",
        "original_replay_ref": ORIGINAL_REF, "metadata_ref": LATER_REF,
        "original_csv_sha256": inventory_sha, "latest_original_created": last_original,
        "matches_by_split": dict(Counter(g["split"] for g in games)),
        "method": {
            "opening": "Submitted actions at t=0 on ordered base IDs 0 and 1; C=collector, P=planter.",
            "bands": "Per appearance [roots with at least one submitted purchase, eligible roots]. "
                     "Eligible means t>0, pre-action cash >=30 and fewer than 10 own workers. "
                     "Root is one player decision across BOTH bases. No survival, profitability or "
                     "independence of repeated roots is implied. All bands are inclusive.",
            "original": "Previously studied; last fifth is a descriptive temporal split only.",
            "later": "Nine additional matches have timestamps after every original match. "
                     "Separate confirmation of the two frozen opening predictions.",
            "visibility": "Only own cash, workers, own bases, configuration and submitted actions "
                          "enter action counts. Observer scores and seat order validate replay identity."
        },
        "summary": summarize(games), "games": games,
        "records": [r for _, r in products if r is not None]
    }
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
