"""Qualify fixed Sapling packages for a four-map original-opponent replication."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "development"
module = importlib.util.spec_from_file_location("strategic_original_runner", PREVIOUS / "matrix.py")
runner = importlib.util.module_from_spec(module)
module.loader.exec_module(runner)
ROOT, play, timing = runner.ROOT, runner.play, runner.timing
read, sha, record, require = runner.read, runner.sha, runner.record, runner.require
PLAN = HERE / "protocol.json"
SEEDS = tuple(range(621758510, 621758514))
ARMS = ("parent", "bc")
WORKFLOW = "sapling-strategic-bc-replication.yml"
DESIGN = "5c7902de9fefc8f5f5704a5e9239e5ce3024ccc1db068e05533d3cb2098c9ca4"
ORIGINAL_REGISTRATION = "551171013eada2e4e8c3b0cff9585456c38c6f3b16208c8c9752ef97f891d142"


def contract():
    require(sha(HERE / "design.json") == DESIGN, "Replication design changed")
    require(sha(PREVIOUS / "protocol.json") == ORIGINAL_REGISTRATION, "Original evaluation inputs changed")
    design = read(HERE / "design.json")
    expected = design["physical_boards"]
    original = read(PREVIOUS / "protocol.json")
    clear, extra = (read(HERE / n) for n in ("clearance.json", "score_delta.json"))
    require(clear["by_seed"] == extra["physical_boards"] == expected
            and clear["source_snapshot"] == extra["global_census_snapshot"]
            and len(extra["through"]) == 40
            and not any(row[key] for row in (clear, extra) for key in ("seed_intersections", "physical_intersections")),
            "Missing or overlapping map clearance evidence")
    require(not set(original["physical_boards"].values()).intersection(expected.values()), "Earlier development map")
    require(set(design["seeds"]) == set(SEEDS) and
            [(r["seed"], r["seat"], r["order"]) for r in design["schedule"]] ==
            [(s, p, list(ARMS if (i+p)%2 == 0 else ARMS[::-1])) for i, s in enumerate(SEEDS) for p in (0,1)],
            "Unbalanced or incomplete tickets")
    shared = ("parent", "parent_weights", "bc", "bc_weights", "opponent", "rules", "environment", "official_helpers",
              "ladder_runner", "worker", "container", "base_registration", "original_game_configuration",
              "current_guide", "match_runner", "replayer", "native_timing_audit", "previous_exposure_guard")
    files = {}
    for key in shared:
        entry = original["files"][key]
        require(record(ROOT / entry["path"]) == entry, f"Frozen dependency drift: {key}")
        files[key] = entry
    for k in ("parent", "bc", "opponent"):
        require(files[k]["bytes"] <= 10_000_000, f"Oversize package: {k}")
    require([files[k]["sha256"] for k in ("parent", "bc", "opponent", "current_guide")] ==
            [design[k] for k in ("parent_sha256", "bc_sha256", "original_sha256", "current_platform_guide_sha256")],
            "Candidate, examiner or guide identity changed")
    add = {"design": HERE/"design.json", "clearance": HERE/"clearance.json", "clearance_census":HERE/"clearance.py",
           "score_delta": HERE/"score_delta.json", "matrix": HERE/"matrix.py",
           "replication_settle": HERE/"settle.py", "previous_settle": PREVIOUS/"settle.py",
           "previous_matrix": PREVIOUS/"matrix.py", "previous_prepare": PREVIOUS/"workflow_cli.py",
           "previous_protocol": PREVIOUS/"protocol.json", "workflow": ROOT/".github/workflows"/WORKFLOW,
           "cli": HERE/"workflow_cli.py"}
    files.update({k:record(p) for k,p in add.items()})
    return {**design, "schema":"sapling-frozen-bc-four-map-replication/frozen/1",
            "platform":original["platform"], "files":files}


def check(digest):
    require(sha(PLAN) == digest and read(PLAN) == contract(), "Incorrect or changed replication registration")
    return read(PLAN)


def bind():
    obj = contract()
    if PLAN.exists():
        require(read(PLAN) == obj, "Refusing to replace frozen protocol")
    else:
        play.live.save(PLAN, obj)
    return sha(PLAN)


# The signed two-map runner implements a general per-ticket loop and per-arm replay.
# Its globals belong to this isolated module instance, never to a prior on-disk registration.
runner.HERE, runner.PLAN, runner.SEEDS, runner.ARMS = HERE, PLAN, SEEDS, ARMS
runner.check = check
verify_block = runner.verify_block


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("bind", "check", "list", "play", "verify"))
    p.add_argument("--matrix-sha")
    p.add_argument("--seed", type=int)
    p.add_argument("--seat", type=int, choices=(0, 1))
    p.add_argument("--out", type=Path)
    p.add_argument("--image")
    args = p.parse_args()
    if args.mode == "bind":
        print(bind())
    elif args.mode == "play":
        sys.exit(runner.run(args.matrix_sha, args.seed, args.seat, args.out, args.image))
    else:
        plan = check(args.matrix_sha)
        if args.mode == "list":
            print(json.dumps({"include":[{"seed":r["seed"], "seat":r["seat"]} for r in plan["schedule"]]}))
        if args.mode == "verify":
            result = verify_block(args.out, args.seed, args.seat)
            print(json.dumps(result, allow_nan=False))
            sys.exit(int(bool(result["verification_errors"])))
