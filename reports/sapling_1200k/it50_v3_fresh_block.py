"""Two prespecified fresh maps in both seats for the frozen 75/25 policy."""
import argparse
from pathlib import Path
import json
import subprocess
import sys

from it50_v3_exam import CPU, PACK, PRIOR, SHA

SEEDS = (620098121, 620098122)
SEATS = (0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()
    args.dir.mkdir(exist_ok=False, parents=True)
    scorer = Path(__file__).with_name("it50_v3_audited_seat.py")
    plan = {"candidate_sha256": SHA, "pack_bytes": PACK.stat().st_size, "seeds": SEEDS,
            "seats": SEATS, "single_worker_cpu": CPU, "examiner_md5": PRIOR.digest(PRIOR.SOURCE,"md5"),
            "scorer_sha256": PRIOR.digest(scorer), "plan_sha256": PRIOR.digest(Path(__file__)),
            "utc_frozen": PRIOR.utc(), "opponent": "source, TEMP=1, SEED=None, native wall-clock search",
            "scope": "four fixed development games; all seats run irrespective of previous outcomes",
            "games": [], "status": "running"}
    report = args.dir / "index.json"
    PRIOR.shared.atomic_json(report, plan)
    for seed in SEEDS:
        for seat in SEATS:
            dest = args.dir / f"{seed}_{seat}.json"
            status = subprocess.run([sys.executable, str(scorer), "--seed", str(seed), "--seat", str(seat),
                                     "--out", str(dest)]).returncode
            r = json.loads(dest.read_text()) if dest.exists() else {"status": "failed before artifact"}
            plan["games"].append({"seed":seed, "seat":seat, "exit":status, "artifact":dest.name,
                                  "status":r["status"], "result":r.get("result"), "margin":r.get("margin"),
                                  "audit_passed":r.get("audit_passed"), "transitions":r.get("actual_transitions")})
            PRIOR.shared.atomic_json(report, plan)
    plan["status"] = ("complete" if all(g["status"] == "complete" and g["audit_passed"]
                                         for g in plan["games"]) else "finished_with_failed_audits")
    plan["utc_finished"] = PRIOR.utc()
    PRIOR.shared.atomic_json(report, plan)


if __name__ == "__main__":
    main()
