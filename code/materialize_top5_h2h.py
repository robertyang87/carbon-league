"""Restore and verify all 108 head-to-head replays from the repository, offline."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("data/sapling_top5_h2h_corpus")
IDENTITIES = Path("reports/sapling_1200k/top5_h2h_data.json")


def read_repo(path, blob=None):
    local = ROOT / path
    if local.is_file():
        return local.read_bytes()
    # A sparse checkout can contain the source in Git without a working-tree file.
    rev = blob or "HEAD:" + path.as_posix()
    return subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", rev])


def check(data, expected, label):
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"{label}: SHA256 mismatch")


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"output contains different bytes: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        partial = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.link(partial, path)
        finally:
            partial.unlink()


def restore(destination):
    inventory = json.loads(read_repo(ARCHIVE / "sources.json"))
    frozen = read_repo(IDENTITIES)
    later = read_repo(ARCHIVE / "later7.json")
    check(frozen, inventory["frozen101_receipt_sha256"], "historical receipt")
    check(later, inventory["later7_receipt_sha256"], "later receipt")
    historical_ids = {g["id"] for g in json.loads(frozen)["mutual_matches"]}
    later_sha = {g["id"]: g["sha256"] for g in json.loads(later)["games"]}
    games = inventory["games"]
    if (len(historical_ids) != 101 or len(later_sha) != 7 or historical_ids & later_sha.keys()
            or len(games) != 108 or {g["id"] for g in games} != historical_ids | later_sha.keys()):
        raise ValueError("source index and score receipts disagree")
    counts = Counter()
    for game in games:
        key = game["id"]
        cohort = "frozen101" if key in historical_ids else "later7"
        if cohort != game["cohort"]:
            raise ValueError(f"{key}: cohort mismatch")
        data = read_repo(Path(game["path"]), game["git_blob"])
        check(data, game["packed_sha256"], key)
        if cohort == "later7":
            check(data, later_sha[key], key + " later receipt")
        publish(destination / cohort / (key + ".json.gz"), data)
        counts[cohort] += 1
    publish(destination / "later7/manifest.json", later)
    print(f'{counts["frozen101"]} frozen and {counts["later7"]} later replays verified in {destination}')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path, help="new directory, or resume with byte-identical files")
    restore(p.parse_args().destination)
