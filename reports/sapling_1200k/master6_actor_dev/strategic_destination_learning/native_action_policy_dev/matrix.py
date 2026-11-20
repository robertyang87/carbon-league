"""Freeze the first eligible physical-action Sapling release and use the established ladder referee."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "development"
TRAIN = HERE.parent / "native_action_policy"
MECHANISM = HERE.parent / "native_action_policy_economy"
MECHANISM_PROTOCOL_SHA = "98456d4c72bc2f206d673b32344ba9100db444ba6660d114d0c677b1d100569e"
PLAN = HERE / "protocol.json"
REFERENCE = HERE.parent / "elite_direct_policy/development/protocol.json"
WORKFLOW = "sapling-native-action-policy-dev.yml"
SEEDS = (621758560, 621758561)
ARMS = ("parent", "candidate")
DESIGN_SHA = "832e9cf9838ec709a82c0b34088d2f18738bbef1538c917784db6af7a3f1019b"
REFERENCE_SHA = "a97d156a28b039ff54e6e0cf55a6cf1c114fdafbfeee7b2c72456fab872949eb"

spec = importlib.util.spec_from_file_location("registered_dev_runner", BASE / "matrix.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ROOT, play, timing = runner.ROOT, runner.play, runner.timing
require, sha, read, record = runner.require, runner.sha, runner.read, runner.record
PLATFORM_FILES = (
    "opponent", "official_helpers", "rules", "environment", "ladder_runner", "worker",
    "container", "base_registration", "original_game_configuration", "current_guide",
    "match_runner", "replayer", "native_timing_audit", "previous_exposure_guard",
)


def training_artifacts():
    return {
        "candidate": TRAIN / "candidate.py", "candidate_weights": TRAIN / "head.npz",
        "gate": TRAIN / "gate.json", "training_registration": TRAIN / "protocol.json",
        "training_fit": TRAIN / "fit.json", "training_source": TRAIN / "train.py",
        "candidate_runtime": TRAIN / "runtime.py", "training_reader": TRAIN / "receipt.py",
        "gate_checker": TRAIN / "check.py", "held_prediction": TRAIN / "held.json.gz",
        "training_maps": HERE.parent / "sources.json", "mechanism_registration": MECHANISM / "protocol.json",
        "mechanism_checker": MECHANISM / "audit.py", "mechanism": MECHANISM / "result.json",
    }


def check_eligibility():
    candidate = sha(TRAIN / "candidate.py")
    gate = read(TRAIN / "gate.json")
    require(gate["candidate_sha256"] == candidate and gate["parent_sha256"] == sha(HERE.parent / "bc.py")
            and gate["head_sha256"] == sha(TRAIN / "head.npz") and gate["physical_change_proven"] is True
            and gate["eligible_for_original_development"] is True, "Held physical-action gate did not pass")
    response = read(MECHANISM / "result.json")
    require(response["candidate_sha256"] == candidate and response["warrant_development"] is True
            and response["protocol_sha256"] == sha(MECHANISM / "protocol.json") == MECHANISM_PROTOCOL_SHA,
            "Held native-opponent economic mechanism did not pass")


def contract():
    design = read(HERE / "design.json")
    require(sha(HERE / "design.json") == DESIGN_SHA and sha(REFERENCE) == REFERENCE_SHA,
            "Development schedule or original referee changed")
    base = read(REFERENCE)
    files = {key: base["files"][key] for key in PLATFORM_FILES}
    for key, identity in files.items():
        require(record(ROOT / identity["path"]) == identity, f"Referee component changed: {key}")
    paths = {
        "reference": REFERENCE, "registered_runner": BASE / "matrix.py",
        "registered_sparse": BASE / "workflow_cli.py", "registered_settler": BASE / "settle.py",
        "workflow": ROOT / ".github/workflows" / WORKFLOW, "matrix": HERE / "matrix.py",
        "cli": HERE / "workflow_cli.py", "settler": HERE / "settle.py",
        "design": HERE / "design.json", "clearance": HERE / "clearance.json",
        "clearance_at_freeze": HERE / "clearance_at_freeze.json", "census_verifier": HERE / "clearance.py",
        "release": HERE / "release.json", "parent": HERE.parent / "bc.py", "parent_weights": HERE.parent / "bc.npz",
        **training_artifacts(), "economic_auditor": HERE / "economy.py",
    }
    files.update({key: record(path) for key, path in paths.items()})
    census = read(paths["clearance"])
    increment = read(paths["clearance_at_freeze"])
    require(census["by_seed"] == design["physical_boards"] == increment["by_seed"]
            and not (census["seed_intersections"] or census["physical_intersections"])
            and len(census["top5_canonical_ledgers"]) > 0 and census["top5_unique_openings"] >= 1650
            and census["verifier_sha256"] == files["census_verifier"]["sha256"]
            and increment["source_census_sha256"] == files["clearance"]["sha256"],
            "Registered maps and independently reproduced ground terrain must agree")
    require(files["opponent"]["sha256"] == design["original_sha256"]
            and files["current_guide"]["sha256"] == design["guide_sha256"]
            and files["parent"]["sha256"] == design["parent_sha256"]
            and files["parent"]["sha256"] != files["candidate"]["sha256"]
            and all(0 < files[key]["bytes"] <= 10_000_000 for key in (*ARMS, "opponent")),
            "Wrong parent/candidate/examiner/guide or oversized release")
    require(files["training_registration"]["sha256"] == design["training_registration_sha256"], "Wrong learning plan")
    check_eligibility()
    cohort = read(paths["training_maps"])
    require(all(r["terrain"] not in design["physical_boards"].values() and r["seed"] not in SEEDS
                for r in cohort["maps"]), "Training/validation cohort used a development map")
    fit = read(paths["training_fit"])
    for role, field in {"training_registration": "protocol", "candidate": "candidate", "candidate_weights": "head",
                        "parent": "parent", "training_source": "train_code", "candidate_runtime": "runtime",
                        "training_reader": "receipt", "training_maps": "sources", "held_prediction": "held_sha"}.items():
        require(fit[field] == files[role]["sha256"], f"Training provenance differs: {field}")
    release = read(paths["release"])
    require({key: files[key] for key in training_artifacts()} == release["files"], "Published artifacts drifted")
    return {**design, "platform": base["platform"], "files": files}


def bind(commit):
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Full GitHub main release commit required")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "origin/main"], check=True)
    check_eligibility()
    artifacts = training_artifacts()
    proof = {}
    for key, path in artifacts.items():
        payload = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + path.relative_to(ROOT).as_posix()])
        proof[key] = record(path)
        require(hashlib.sha256(payload).hexdigest() == proof[key]["sha256"], "Published bytes differ: " + key)
    for path, result in ((HERE / "release.json", {"training_commit": commit, "files": proof}), (PLAN, None)):
        if result is None:
            result = contract()
        if path.exists():
            require(read(path) == result, "Refusing to replace frozen " + path.name)
        else:
            play.live.save(path, result)
    return sha(PLAN)


def check(digest):
    require(sha(PLAN) == digest and read(PLAN) == contract(), "Different protocol or inputs")
    return read(PLAN)


runner.HERE, runner.PLAN, runner.SEEDS, runner.ARMS, runner.check = HERE, PLAN, SEEDS, ARMS, check
verify_block = runner.verify_block


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("bind", "list", "check", "play", "verify"))
    p.add_argument("--training-commit")
    p.add_argument("--matrix-sha")
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--seat", type=int, choices=(0, 1))
    p.add_argument("--out", type=Path)
    p.add_argument("--image")
    args = p.parse_args()
    if args.mode == "bind":
        print(bind(args.training_commit))
        return 0
    plan = check(args.matrix_sha)
    if args.mode == "list":
        print(json.dumps({"include": [{"seed": r["seed"], "seat": r["seat"]} for r in plan["schedule"]]}))
    elif args.mode == "play":
        return runner.run(args.matrix_sha, args.seed, args.seat, args.out, args.image)
    elif args.mode == "verify":
        verified = verify_block(args.out, args.seed, args.seat)
        print(json.dumps(verified, allow_nan=False))
        return int(bool(verified["verification_errors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
