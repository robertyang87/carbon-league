"""Rebuild the frozen 628e28d0 archive, dual-engine validations, and per-game evidence."""
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
sys.path.insert(0, str(TOP))

import fetch
from appendix_97de1ebc.collect import inventories
from archive import prepare

from continue_appendix import check_archive, previous_manifests
from validation_delta import run as validate

REVISION = "628e28d06b0610f6ceebdac3cc632057766bddd2"
PREVIOUS = "56e1e6137fcff237aa212ea9423a29e8d486501d"


def restore_published_replays(manifest_path):
    if not manifest_path.is_file():
        return
    manifest = fetch.load(manifest_path)
    if manifest["revision"] != REVISION:
        raise ValueError("published revision changed")
    check_archive(manifest)
    keys = ("id", "revision", "source", "sha256", "canonical_sha256", "seed", "environment_uuid",
            "transitions", "terminal", "clean_terminal", "source_attempt_errors", "frame_verified")
    for entry in manifest["new_archives"]:
        with tarfile.open(fetch.ROOT / entry["archive"]) as archive:
            for game in manifest["games"]:
                if game["archive_location"]["archive"] != entry["archive"]:
                    continue
                fetch.atomic(fetch.CACHE / game["path"], archive.extractfile(game["path"]).read())
                fetch.dump(fetch.CACHE / "receipts" / (game["id"] + ".json"), {k: game[k] for k in keys})


def main():
    prior = previous_manifests(TOP / "appendix_56e1e613")
    old = fetch.load(prior[-1])
    if old["revision"] != PREVIOUS or old["accumulated_games"] != 1630:
        raise ValueError("unexpected predecessor")
    fetch.git("merge-base", "--is-ancestor", PREVIOUS, REVISION)
    names = [str(path.parent.relative_to(TOP)) for path in prior]
    delta = inventories(PREVIOUS, REVISION, names)
    if delta["unscored_new_ids"] or delta["new_count"] != 6 or delta["new_mutual"]:
        raise ValueError("frozen delta changed")
    fetch.dump(HERE / "score_delta.json", delta)
    faults = [p.parent / "unscored/manifest.json" for p in prior if (p.parent / "unscored/manifest.json").is_file()]
    manifest_path = HERE / "manifest.json"
    restore_published_replays(manifest_path)
    fault_cards = {}
    prepare(REVISION, prior, HERE, (delta["new_count"], delta["new_mutual"]), unscored=fault_cards)
    if set(fault_cards) != {g["id"] for p in faults for g in fetch.load(p)["games"]}:
        raise ValueError("historical fault roster changed")
    manifest = fetch.load(manifest_path)
    expected = {gid for gid, g in delta["all_added_references_by_id"].items() if g["class"] == "top"}
    if {g["id"] for g in manifest["games"]} != expected or manifest["accumulated_games"] != delta["union_games"]:
        raise ValueError("independent full catalog and frozen delta disagree")
    check_archive(manifest)
    ledgers = []
    for name, path in zip(names, prior, strict=True):
        ledger = path.parent / "validation_maps.jsonl.gz"
        blob = fetch.git("show", REVISION + ":" + str(ledger.relative_to(fetch.ROOT)))
        if ledger.read_bytes() != blob:
            raise ValueError("historical map registry changed: " + name)
        ledgers.append((name, fetch.sha(blob)))
    validate(fetch.CACHE, HERE, REVISION, PREVIOUS, ledgers, previous_fault_ledgers=faults)

    def run(script, *args):
        subprocess.run([sys.executable, str(TOP / script), *map(str, args)], check=True)

    index = HERE / "context_index.tsv.gz"
    receipt = HERE / "context_receipt.json"
    run("appendix_499886f1/cases.py", "--manifest", manifest_path, "--cache", fetch.CACHE)
    run("appendix_0137949d/cash.py", "--manifest", manifest_path, "--cache", fetch.CACHE)
    run("turn_action_windows.py", HERE, "--cache", fetch.CACHE)
    run("context_catalog.py", "--manifest", manifest_path, "--output", index)
    run("context_audit.py", "--index", index, "--validation", HERE / "validation_games.jsonl.gz", "--out", receipt, "--jobs", "1")
    run("appendix_628e28d0/exposure.py")
    run("appendix_628e28d0/development_maps.py")
    run("appendix_628e28d0/horizons.py")
    run("appendix_628e28d0/report.py")


if __name__=="__main__":
    with TemporaryDirectory(prefix="sapling-score-628e28d0-") as scratch:
        fetch.CACHE=Path(scratch)
        main()
