"""Verify the frozen H24 continuation gate and play one same-host paired session."""
import argparse
import asyncio
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CARGO = HERE.parents[1]
ROOT = HERE.parents[5]
LIVE = CARGO / "ladder_live"
sys.path.insert(0, str(LIVE))
import live


def load(path):
    return json.loads(path.read_bytes())


def check(plan):
    for entry in plan["files"].values():
        path = ROOT / entry["path"]
        if path.stat().st_size != entry["bytes"] or live.sha(path) != entry["sha256"]:
            raise ValueError(f"Frozen artifact changed: {path}")
    if max(plan["files"][arm]["bytes"] for arm in ("student", "parent", "opponent")) > 10_000_000:
        raise ValueError("Oversize submission")
    base = load(LIVE / "protocol.json")
    for role in ("official_helpers", "rules", "environment"):
        entry = base["files"][role]
        if live.sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Official dependency changed: {role}")
    prev = HERE.parent / "response"
    results = load(prev / "results.json")
    prereg = load(prev / "protocol.json")
    hashes = {arm: plan["files"][arm]["sha256"] for arm in ("student", "parent", "opponent")}
    if results["binding"]["student"] != hashes["student"] or results["binding"]["parent"] != hashes["parent"]:
        raise ValueError("Conditional study arms differ")
    if prereg["examiner_sha256"] != hashes["opponent"]:
        raise ValueError("Conditional examiner differs")
    cases = prereg["cases"]
    recorded = {(r["seed"], r["seat"]): r for r in results["rows"]}
    if len(cases) != 2 or len(recorded) != 2:
        raise ValueError("Wrong number of continuation cases")
    ground_deltas = []
    for case in cases:
        seed, seat = case["seed"], case["seat"]
        arms = {}
        for arm in ("parent", "student"):
            tape = json.loads(gzip.decompress((prev / f"{seed}_{seat}_{arm}.json.gz").read_bytes()))
            if tape["identity"]["binding"] != results["binding"] or tape["physical_seat"] != seat:
                raise ValueError("Continuation tape identity mismatch")
            scores = tape["official_scores"]
            margin = scores[seat] - scores[1-seat]
            decisions = sum(bool(step["changed"]) for step in tape["comparisons"])
            rent = tape["cash"]["through_199"]
            net_rent = rent[seat].get("ground_rent", 0) - rent[1-seat].get("ground_rent", 0)
            arms[arm] = (margin, decisions, net_rent)
        old, new = arms["parent"], arms["student"]
        row = recorded[seed, seat]
        delta = new[0] - old[0]
        rent_delta = new[2] - old[2]
        if (not math.isclose(delta, row["margin_delta"], abs_tol=1e-6)
                or not math.isclose(rent_delta, row["ground_margin_delta"], abs_tol=1e-6)
                or new[1] != row["actual_changed_turns"] or new[1] < 2 or delta <= 0):
            raise ValueError(f"Conditional gate failed at {seed}, seat {seat}")
        ground_deltas.append(rent_delta)
    if not sum(ground_deltas) > 0 or results["passed"] is not True:
        raise ValueError("Pooled ground-rent gate failed")
    if load(HERE / "clearance.json")["physical_boards"] != plan["physical_boards"]:
        raise ValueError("Map clearance differs from registered matrix")
    return base


def game_plan(matrix, base, arm, seed, seat):
    files = {role: base["files"][role] for role in ("environment", "official_helpers", "rules")}
    files.update(candidate=matrix["files"][arm], candidate_weights=matrix["files"][arm+"_weights"],
                 opponent=matrix["files"]["opponent"])
    return {"matrix_sha256": live.sha(HERE / "protocol.json"), "arm": arm,
            "seed": seed, "candidate_seat": seat, "opponent_seat": 1-seat,
            "physical_map_sha256": matrix["physical_boards"][str(seed)],
            "files": files, "platform": base["platform"], "workflow_run": os.environ.get("GITHUB_RUN_ID")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seat", type=int, choices=[0, 1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--image", default="sapling-h24:registered")
    args = parser.parse_args()
    matrix = load(HERE / "protocol.json")
    base = check(matrix)
    if args.check:
        return 0
    schedule = [s for s in matrix["schedule"] if (s["seed"], s["seat"]) == (args.seed, args.seat)]
    if len(schedule) != 1 or args.out is None:
        parser.error("Supply exactly one registered --seed/--seat and an --out directory")
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) != 2:
        raise RuntimeError("Registered two-core CPU set is unavailable")
    image = subprocess.check_output(["docker","image","inspect","--format","{{.Id}}",args.image],text=True).strip()
    args.out.mkdir(parents=True, exist_ok=True)
    status = []
    for arm in schedule[0]["order"]:
        dest = args.out / arm
        dest.mkdir(exist_ok=False)
        reg = dest / "registration.json"
        live.save(reg, game_plan(matrix, base, arm, args.seed, args.seat))
        live.configure(reg, dest/"attempt.json", args.out/".match.lock")
        try:
            rc = asyncio.run(live.run(argparse.Namespace(out=dest, image=image, play=True,
                                             cpus=",".join(map(str, cpus)))))
            status.append(rc)
        except Exception as exc:
            live.save(dest/"orchestration_error.json", {"error":repr(exc), "arm":arm})
            status.append(2)
    return max(status)


if __name__ == "__main__":
    sys.exit(main())
