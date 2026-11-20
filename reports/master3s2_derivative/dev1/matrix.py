"""Run one frozen Master3s2 derivative against the intact published original, switching seats on four development maps."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / ".git").exists())
STRATEGY = ROOT / "reports/sapling_1200k/master6_actor_dev/strategic_destination_learning"
BASE = STRATEGY / "development"
REFERENCE = STRATEGY / "elite_direct_policy/development/protocol.json"
SPEC = importlib.util.spec_from_file_location("ladder_matrix", BASE / "matrix.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
read, sha, require, record, play, timing = runner.read, runner.sha, runner.require, runner.record, runner.play, runner.timing
PLAN = HERE / "protocol.json"
SEEDS = tuple(range(621758580, 621758584))
ARMS = ("opponent_policy",)
ARM = ARMS[0]
WORKFLOW = "master3s2-derivative-dev1.yml"
TRAIN = HERE.parent / "opponent_policy"
PACKAGE = TRAIN / "candidate.py"
ORIGINAL_SHA = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
GUIDE_SHA = "97adbc81e39ef2b1834af96e261cd10e76424352abde5e04868b3467c2a5cec8"
REF_SHA = "a97d156a28b039ff54e6e0cf55a6cf1c114fdafbfeee7b2c72456fab872949eb"


def contract():
    require(sha(REFERENCE) == REF_SHA, "Frozen referee reference changed")
    reference = read(REFERENCE)
    shared = ("opponent", "official_helpers", "rules", "environment", "ladder_runner", "worker", "container",
              "base_registration", "original_game_configuration", "current_guide", "match_runner", "replayer",
              "native_timing_audit", "previous_exposure_guard")
    files = {key: reference["files"][key] for key in shared}
    require(all(record(ROOT / row["path"]) == row for row in files.values()), "Original or referee no longer matches the registered bytes")
    require(files["opponent"]["sha256"] == ORIGINAL_SHA and files["current_guide"]["sha256"] == GUIDE_SHA, "Changed original or platform guide")
    release = read(HERE / "release.json")
    require(release["package"]["path"] == PACKAGE.relative_to(ROOT).as_posix() and
            release["original_sha256"] == ORIGINAL_SHA and release["eligible_for_development"] is True,
            "Exam registration requires the independently accepted, locally trained derivative")
    weights = ROOT / release["weights"]["path"]
    require(weights.parent == TRAIN and weights.suffix == ".npz", "Invalid learned weights path")
    entries = {ARM: PACKAGE, ARM + "_weights": weights, "release": HERE / "release.json",
               "matrix": HERE / "matrix.py", "release_builder": HERE / "bind.py", "local_settler": HERE / "settle.py",
               "direct_source_auditor": HERE.parent / "verify_source.py", "full_search_verifier": TRAIN / "search_probe/full_search_check.py",
               "clearance_increment": HERE / "clearance_at_freeze.json",
               "clearance": HERE / "clearance.json", "clearance_verifier": HERE / "clearance.py", "cli": HERE / "workflow_cli.py",
               "workflow": ROOT / ".github/workflows" / WORKFLOW, "registered_runner": BASE / "matrix.py",
               "registered_settler": BASE / "settle.py", "registered_sparse": BASE / "workflow_cli.py",
               "observation_replay": STRATEGY / "native_action_onpolicy_intake/observations.py", "reference": REFERENCE}
    evidence = release["inputs"]
    require(evidence and not set(evidence) & set(entries) and not set(evidence) & set(shared), "Invalid release evidence keys")
    for key, value in evidence.items():
        path = ROOT / value["path"]
        require(path.is_relative_to(HERE.parent) and ".." not in path.parts and record(path) == value, "Different training or eligibility record")
        entries[key] = path
    files.update({key: record(path) for key, path in entries.items()})
    require(files[ARM] == release["package"] and files[ARM + "_weights"] == release["weights"] and
            files[ARM]["sha256"] != ORIGINAL_SHA and 0 < files[ARM]["bytes"] <= 10_000_000, "Unbound or oversize derivative")
    src_spec = importlib.util.spec_from_file_location("original_source_auditor", entries["direct_source_auditor"])
    src_audit = importlib.util.module_from_spec(src_spec)
    src_spec.loader.exec_module(src_audit)
    original_path = ROOT / files["opponent"]["path"]
    original_bytes = original_path.read_bytes() if original_path.is_file() else subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "HEAD:" + files["opponent"]["path"]])
    source_proof = src_audit.verify(original_bytes, PACKAGE.read_bytes())
    require(source_proof == read(TRAIN / "source_identity.json") and source_proof["original_restored"] and source_proof["single_opponent_root_call"],
            "Candidate does not restore the original parent or differs from the provenance receipt")
    control = read(TRAIN / "physical.json")
    require(control["candidate_sha256"] == source_proof["candidate_sha256"] and control["max_own_root_logit_change"] == 0
            and control["max_unchanged_critic_change"] == 0 and control["zero_opponent_logits_bitwise_equal"] is True
            and control["all_original_rollout_logits_bitwise_equal"] is True, "Own policy, unchanged critic or rollout drifted")
    fit = read(TRAIN / "fit.json")
    require(fit["source"] == ORIGINAL_SHA and fit["candidate_sha256"] == files[ARM]["sha256"]
            and fit["head_sha256"] == files[ARM + "_weights"]["sha256"]
            and fit["held_sha"] == sha(TRAIN / "held.json") and fit["protocol"] == sha(TRAIN / "protocol.json")
            and fit["observational_gate"] is release["observational_gate"], "Frozen label source, package or held outcome differs")
    mechanism = json.loads(gzip.decompress((TRAIN / "search_probe/full_search.json.gz").read_bytes()))
    require(mechanism["candidate_sha256"] == files[ARM]["sha256"] and mechanism["dispatch_eligible"] is release["mechanism_gate"]
            and mechanism["original_sha256"] == ORIGINAL_SHA and mechanism["verifier_sha256"] == files["full_search_verifier"]["sha256"],
            "Frozen full original search has a different mechanism receipt")
    if release["registration_class"] == "qualified_development":
        require(fit["observational_gate"] and mechanism["dispatch_eligible"], "Unqualified development")
    else:
        require(release["registration_class"] == "diagnostic_development" and release.get("diagnostic_decision"), "Missing diagnosis or decision")
    clearance = read(HERE / "clearance.json")
    update = read(HERE / "clearance_at_freeze.json")
    require(update["source_census_sha256"] == files["clearance"]["sha256"] and not update["unresolved"]
            and not update["seed_intersections"] and not update["physical_intersections"]
            and update["opponent_training_and_holdout_checked"] == 56, "Unresolved new training or ladder exposure")
    boards = clearance["by_seed"]
    require(set(boards) == {str(s) for s in SEEDS} and len(set(boards.values())) == len(SEEDS), "Unregistered or duplicate maps")
    train_protocol = TRAIN / "protocol.json"
    require(record(train_protocol)["sha256"] == "16e77a290cfe1c22503981f34cff934aa3fccd6fc69b0ba41d9383e1e078e502"
            and any(r["path"] == train_protocol.relative_to(ROOT).as_posix() for r in evidence.values()),
            "Missing frozen original-opponent training inventory")
    training = read(train_protocol)["games"]
    require(training and not set(SEEDS).intersection(row["seed"] for row in training)
            and not set(boards.values()).intersection(row["terrain"] for row in training),
            "A training or held opening intersects the development panel")
    require(clearance["seed_intersections"] == [] and clearance["physical_intersections"] == [], "Known map overlap")
    return dict(schema="master3s2-derivative-development/1", registration_class=release["registration_class"],
                observational_gate=fit["observational_gate"], purpose="Our learned opponent root-action predictor in the original source vs intact Master3s2 on four registered development maps. Never use these maps as unseen final tests.",
                seeds=list(SEEDS), arms=list(ARMS), physical_boards=boards, platform=reference["platform"], files=files,
                schedule=[dict(seed=s, seat=p, order=[ARM]) for s in SEEDS for p in (0, 1)],
                budget="Exactly eight games in one dispatch. No retries after exposure or outcome-based replacements. Both seats play to official termination or a measured forfeit.",
                score="Full observed W/D/L and official margin from the candidate perspective; draws are not wins. A forfeit is a counted outcome without an invented numeric margin. Report both seats and physical-map clusters.",
                uncertainty="Show all four per-map results and enumerate the empirical percentile bootstrap over four whole paired maps. These development estimates have only four independent map clusters.",
                runtime="Same public module and agent(observation, configuration) from the registered bytes, isolated ladder cgroups, randomized T=1 original with native 0.58s search. The candidate's entire policy and weights are embedded in the same single file; the weight path records the standalone head checkpoint embedded by our source builder. Both seats have full IPC/cgroup timing. Opponent _S_STATS supplies the original examiner's search counts.")


def check(digest):
    require(sha(PLAN) == digest and read(PLAN) == contract(), "Requested registration differs from the frozen code")
    return read(PLAN)


runner.HERE, runner.PLAN, runner.SEEDS, runner.ARMS, runner.check = HERE, PLAN, SEEDS, ARMS, check
verify_block = runner.verify_block


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "check", "list", "play", "verify", "observations"))
    parser.add_argument("--matrix-sha")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--seat", type=int, choices=(0, 1))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--image")
    args = parser.parse_args()
    if args.mode == "freeze":
        value = contract()
        if PLAN.exists():
            require(read(PLAN) == value, "Different existing freeze")
        else:
            play.live.save(PLAN, value)
        print(sha(PLAN))
    else:
        plan = check(args.matrix_sha)
        if args.mode == "list":
            print(json.dumps({"include": [{"seed": row["seed"], "seat": row["seat"]} for row in plan["schedule"]]}))
        elif args.mode in ("play", "verify", "observations"):
            require(any((r["seed"],r["seat"]) == (args.seed,args.seat) for r in plan["schedule"]), "Unregistered seed/seat")
            if args.mode == "play":
                sys.exit(runner.run(args.matrix_sha, args.seed, args.seat, args.out, args.image))
            if args.mode == "verify":
                result = verify_block(args.out, args.seed, args.seat)
                print(json.dumps(result, allow_nan=False))
                sys.exit(int(bool(result["verification_errors"])))
            sys.path.append(str(STRATEGY / "native_action_onpolicy_intake"))
            from observations import recover
            folder = args.out / ARM
            (folder / "observations.jsonl.gz").write_bytes(recover(folder))
