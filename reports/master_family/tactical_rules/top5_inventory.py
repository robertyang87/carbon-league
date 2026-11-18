"""Restore and audit the archived Top 5 diagnostic replay set without model training.

  python reports/master_family/tactical_rules/top5_inventory.py --out /tmp/top5/replays
Pass --insecure for the spectator IP with a mismatched TLS hostname. Archived SHA-256
identities, score order and the terminal state are checked for every restored replay.
The scoreboard and match-table snapshots are frozen in the sibling JSON files.
"""
import argparse
import gzip
import hashlib
import json
import os
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "deep_review"))
from recover_public import canonical, extract, validate


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch_game(key, row, record, dest, context, cache_only):
    path = dest / (key + ".json.gz")
    if path.is_file():
        packed = path.read_bytes()
    else:
        packed = None
        if record.get("git_blob"):
            blob = subprocess.run(["git", "-C", str(HERE), "cat-file", "blob", record["git_blob"]],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if blob.returncode == 0:
                packed = blob.stdout
        if packed is None and not cache_only:
            request = urllib.request.Request(record["public_url"], headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(request, context=context, timeout=45) as response:
                env = extract(response.read())[0]
            packed = gzip.compress(canonical(env), compresslevel=1, mtime=0)
        if packed is None:
            raise ValueError("Local replay missing: " + key)
    env = json.loads(gzip.decompress(packed))
    if digest(canonical(env)) != record["canonical_sha256"]:
        raise ValueError("Replay identity mismatch: " + key)
    checked = validate(env, row)
    if checked["environment_uuid"] != record["environment_uuid"] or checked["step_count"] != record["steps"]:
        raise ValueError("Replay state mismatch: " + key)
    if not path.is_file():
        with tempfile.NamedTemporaryFile(dir=dest, delete=False) as temp:
            temp.write(packed)
        os.replace(temp.name, path)
    return key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--cache-only", action="store_true", help="Refuse public HTTP requests")
    args = p.parse_args()
    manifest = json.loads((HERE / "top5_inventory.json").read_bytes())
    scores = (HERE / "top5_score_rows.json").read_bytes()
    if digest(scores) != manifest["score_rows_sha256"]:
        raise ValueError("Score table differs from the frozen snapshot")
    rows = json.loads(scores)
    records = manifest["replays"]
    if set(rows) != set(records) or len(rows) != manifest["unique_identified_games"]:
        raise ValueError("Expected exactly one replay identity for each score row")
    args.out.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
    def restore(key):
        return fetch_game(key, rows[key], records[key], args.out, context, args.cache_only)
    with ThreadPoolExecutor(args.workers) as pool:
        restored = list(pool.map(restore, sorted(records)))
    print(json.dumps({"verified_games": len(restored), "out": str(args.out)}))


if __name__ == "__main__":
    main()
