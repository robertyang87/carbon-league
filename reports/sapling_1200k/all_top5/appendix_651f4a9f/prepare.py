"""Archive new appearances of the five frozen players and verify every public transition."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
sys.path.insert(0, str(TOP))

from appendix_batch import prepare
from continue_appendix import check_archive, previous_manifests
import fetch
from validation_delta import run as validate


def main():
    revision = fetch.git("rev-parse", "651f4a9fd^{commit}").decode().strip()
    parents = previous_manifests(TOP / "appendix_9e2a81cf")
    manifests = [fetch.load(p) for p in parents]
    if sum(len(m["games"]) for m in manifests) != 1606:
        raise ValueError("the frozen predecessor must have 1606 numeric games")
    prior = manifests[-1]["revision"]
    faults = [p.parent / "unscored/manifest.json" for p in parents if (p.parent / "unscored/manifest.json").exists()]
    excluded = {}
    prepare(revision, parents, HERE, unscored=excluded)
    if set(excluded) != {g["id"] for p in faults for g in fetch.load(p)["games"]}:
        raise ValueError("changed fault cohort")
    check_archive(fetch.load(HERE / "manifest.json"))
    ledgers = []
    for p in parents:
        maps = p.parent / "validation_maps.jsonl.gz"
        raw = maps.read_bytes()
        if raw != fetch.git("show", revision + ":" + str(maps.relative_to(fetch.ROOT))):
            raise ValueError("historical registry differs from frozen revision")
        ledgers.append((str(p.parent.relative_to(TOP)), fetch.sha(raw)))
    validate(fetch.CACHE, HERE, revision, prior, ledgers, previous_fault_ledgers=faults)


if __name__ == "__main__":
    main()
