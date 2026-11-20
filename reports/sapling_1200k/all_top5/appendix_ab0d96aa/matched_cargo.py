"""Follow high-load enemy collectors on a fixed grid in same-opponent, same-seat games."""
import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = TOP.parents[2]
sys.path.insert(0, str(TOP))
from economy_extract import distances

OPPONENT = "RyanX-AIPowercore-it920-evo31"
ELITES = {"ClaudeFable5.2-v7", "claude-AlphaCarbon-Master5s"}


def read(path):
    return json.loads(path.read_bytes())


def archive_game(game, cache):
    source = game["archive_location"]
    if cache is not None and (cache / game["path"]).is_file():
        raw = (cache / game["path"]).read_bytes()
    else:
        with tarfile.open(REPO / source["archive"]) as archive:
            raw = archive.extractfile(source["member"]).read()
    if hashlib.sha256(raw).hexdigest() != game["sha256"]:
        raise ValueError(game["id"])
    return json.loads(gzip.decompress(raw))


def examine(game, ledger, cache):
    replay = archive_game(game, cache)
    frames = replay["steps"]
    turns = ledger["turns"]
    if len(turns) + 1 != len(frames) or ledger["replay_sha256"] != game["sha256"]:
        raise ValueError("cash ledger does not match source")
    dist = distances(replay["configuration"]["size"])
    seen = []
    for t in range(0, len(turns) - 9, 10):
        other, elite = frames[t][0]["observation"]["players"]
        chasers = [w for w in elite[2].values() if w[2] == "COLLECTOR"]
        banks = Counter()
        for turn in turns[t:t+10]:
            for seat, unit, amount in turn[3]:
                if seat == 0:
                    banks[unit] += amount
        for uid, (cell, load, role) in sorted(other[2].items()):
            if role != "COLLECTOR" or load < 100:
                continue
            home = min(dist[cell][h] for h in other[1].values())
            contested = any(c[1] <= load and dist[cell][c[0]] <= 1 for c in chasers)
            seen.append([t, uid, cell, round(load, 4), home, contested, round(banks[uid], 4)])
    return {"id": game["id"], "seat_names": game["seat_names"], "seed": game["seed"],
            "scores": game["score_by_seat"], "source_sha256": game["sha256"], "workers": seen}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, help="Optional verified replay cache in a sparse checkout.")
    args = parser.parse_args()
    checkpoint = read(HERE / "manifest.json")
    revisions = set(checkpoint["baseline_revisions"]) | {checkpoint["revision"]}
    found = []
    for manifest in [TOP / "manifest.json", *sorted(TOP.glob("appendix_*/manifest.json"))]:
        batch = read(manifest)
        if batch["revision"] not in revisions:
            continue
        games = [g for g in batch["games"] if g["seat_names"][0] == OPPONENT and g["seat_names"][1] in ELITES]
        if games:
            ledgers = json.loads(gzip.decompress((manifest.parent / "turn_cash.json.gz").read_bytes()))
            by_id = {g["id"]: g for g in ledgers["games"]}
            for game in games:
                found.append(examine(game, by_id[game["id"]], args.cache))
    if len(found) != len({g["id"] for g in found}) or len(found) != len({g["seed"] for g in found}):
        raise ValueError("joined a map twice")
    data = {"source_revision": checkpoint["revision"], "other_physical_seat": 0,
            "grid": "t=0,10,... with ten full future actions; load>=100",
            "columns": ["pre_step", "unit", "cell", "load", "distance_to_nearest_own_base",
                        "adjacent_equal_or_lighter_enemy_collector", "same_unit_normal_bank_next_10"],
            "games": sorted(found, key=lambda g:g["id"])}
    out = gzip.compress((json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(), mtime=0)
    (HERE / "matched_cargo.json.gz").write_bytes(out)


if __name__ == "__main__":
    main()
