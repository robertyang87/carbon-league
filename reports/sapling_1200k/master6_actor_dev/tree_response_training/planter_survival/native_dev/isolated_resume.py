"""Use authentic files from the first arm's Git revision if the live ladder ledger advances mid-comparison."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parents[1]
ROOT = TRIAL.parents[3]
REL = HERE.relative_to(ROOT)
REVISION = "bd9ed3fb1cd397b8050b5ba863a36c1828fe3f9a"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def frozen_parent():
    base = HERE / "parent"
    n = read(base / "results.json")
    a = read(base / "official_replay.json")["runs"]["parent"]["games"]
    require = {(s, p) for s in read(HERE / "protocol.json")["maps"] for p in (0, 1)}
    check(
        n["status"] == "complete"
        and {(r["seed"], r["candidate_seat"]) for r in n["games"]} == require
        and len(n["games"]) == 4
        and {(g["seed"], g["seat"]) for g in a} == require
        and len(a) == 4,
        "parent must have four completed official outcomes",
    )
    files = [base / "results.json", base / "official_replay.json"]
    for r in n["games"]:
        path = base / r["raw_file"]
        check(sha(path) == r["raw_sha256"], "changed parent transcript")
        files.append(path)
    return {p.name: sha(p) for p in files}


def local_registration():
    sys.path.insert(0, str(ROOT / "reports/sapling_1200k"))
    import map_identity

    fp, seeds, terrains = map_identity.ladder_boards()
    declared = read(HERE / "slot_request.json")
    check(
        not (seeds & set(declared["seeds"]) or terrains & set(declared["boards"].values())),
        "new ladder terrain conflict",
    )
    return dict(
        fingerprint=fp,
        registered_games=sum(x[1] for x in map_identity.SOURCES),
        excluded_actual_boards=declared["boards"],
        current_map_identity_sha256=sha(Path(map_identity.__file__)),
    )


def extract(root):
    # Use git archive for the exact source tree. No hardlinks or imports from the live repo.
    archive = root / "snapshot.tar"
    areas = [
        "code",
        "reports/sapling_1200k",
        "reports/sapling_bootstrap",
        "data/master3s2_source.py",
        "data/weights/sapling_1200k/master6_actor_hybrid.py",
        "data/weights/sapling_1200k/terminal_return_fit_v1.py",
        "data/weights/sapling_1200k/master6_actor_v1.npz",
    ]
    subprocess.run(["git", "archive", "-o", str(archive), REVISION, *areas], cwd=ROOT, check=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(root)], check=True)
    archive.unlink()


def capsule_worker(root, ledger):
    sys.path.insert(0, str(root / REL))
    import run as frozen

    check(
        frozen.HERE == root / REL and frozen.sg.ROOT == root and frozen.gate.ROOT == root,
        "candidate or arena resolved outside capsule",
    )
    header = read(HERE / "parent/results.json")["header"]
    source = header["source_sha256"]
    check(
        frozen.gate.protocol_hashes() == source and all(sha(root / k) == v for k, v in source.items()),
        "Git snapshot does not reproduce whole parent source fingerprint",
    )
    manifest = read(HERE / "release_manifest.json")
    for f in ("run.py", "build.py", "runtime.py", "release_manifest.json", "protocol.json", "candidate.py"):
        check(sha(root / REL / f) == sha(HERE / f), f"development source differs in capsule: {f}")
    original = frozen_parent()
    req = read(HERE / "slot_request.json")
    plan = read(HERE / "protocol.json")
    # Only output destinations and the already-granted live slot are external; all imported code and simulator
    # processes are from the snapshot. Pass the frozen candidate file itself to the isolated arena.
    frozen.HERE = HERE
    frozen.slot.SLOT = TRIAL / "resources/veto_dev_slot"
    signed = {
        "snapshot_commit": REVISION,
        "native_driver": "Git snapshot of the father run and the preregistered candidate",
        "frozen_sources": source,
        "frozen_source_count": len(source),
        "parent_verified_artifacts": original,
        "later_map_registration": json.loads(ledger),
        "live_files_differing_from_first_arm": {
            k: sha(ROOT / k) for k, v in source.items() if sha(ROOT / k) != v
        },
        "snapshot_driver_sha256": sha(root / REL / "run.py"),
        "resume_code_sha256": sha(Path(__file__)),
        "risk_candidate_sha256": manifest["pack_sha256"],
    }
    frozen.trial.frozen_file(HERE / "isolated_execution.json", signed)
    with tempfile.TemporaryDirectory() as t:
        parent, weights, _ = frozen.trial.recipe("learned", Path(t))
        parent_rows = frozen.run_arm("parent", parent, weights, req, plan)
        check(original == frozen_parent(), "parent evidence changed during readback")
        candidate_rows = frozen.run_arm(
            "candidate", root / REL / "candidate.py", root / REL / "release_manifest.json", req, plan
        )
        frozen.trial.frozen_file(
            HERE / "comparison.json",
            frozen.compare({"parent": parent_rows, "candidate": candidate_rows}, req),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capsule", type=Path)
    p.add_argument("--ledger")
    a = p.parse_args()
    if a.capsule:
        capsule_worker(a.capsule, a.ledger)
        return
    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[k] = "1"
    sys.path.insert(0, str(TRIAL))
    import resource_slot as slot

    frozen_parent()
    if (slot.SLOT / "release.json").exists():
        return
    with slot.native_session(), tempfile.TemporaryDirectory(prefix="survival-native-revision-") as tmp:
        root = Path(tmp)
        extract(root)
        checked = local_registration()
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--capsule",
                str(root),
                "--ledger",
                json.dumps(checked),
            ],
            cwd=root,
            check=True,
        )
    slot.release()
    subprocess.run([sys.executable, str(HERE / "report.py")], check=True)
    print("All eight games signed. The training owner may resume.")


if __name__ == "__main__":
    main()
