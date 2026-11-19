"""One new map, both seats, against the original searching Master3s2.

Each invocation evaluates one frozen arm. Reuse the same CPU pair for both.
Record extinction, official cash, and real deposit/tree income separately.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile

import external_3s2 as exam


SEED = 620081731
ARMS = {"sqrt64": "sqrt64.npz", "elite_bc_v1": "elite_bc_v1.npz"}
SOURCE_FILES = [
    Path(__file__), Path(exam.__file__), Path(exam.prior.__file__),
    Path(exam.prior.shared.__file__), exam.prior.SOURCE,
    *(exam.ROOT / "code" / name for name in
      ("eval_v53.py", "fastsim.py", "cnet2_pack.py", "cnet2_agent.py", "cnet2_feat.py", "cnet2_np.py")),
]


def run_seat(job):
    label, pack, side, cpu = job
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    row = exam.play((label, pack, SEED, side))
    if not row["full_game"] or any(row["agent_errors_by_seat"]):
        raise RuntimeError(f"Incomplete game or agent errors: {row['errors']}")
    keys = ("state_t", "cash", "cargo", "alive_workers", "alive_trees", "new_trees_planted",
            "own_positive_delivery_carbon", "cumulative_tree_absorption_income", "recruited_total")

    def economy(snapshot):
        return {role: {key: state[key] for key in keys} for role, state in snapshot.items()}

    search = row["opponent_search_stats"]
    return {
        "seed": SEED, "candidate_seat": side, "worker_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "actual_transitions": row["actual_transitions"], "termination": row["termination"],
        "extinct_by_seat": row["extinct_by_seat"], "win_draw_loss": row["result"],
        "official_score_by_seat": row["official_score_by_seat"],
        "candidate_official_margin": row["candidate_official_margin"],
        "pre_last_transition_cash_by_seat": row["pre_last_transition_cash_by_seat"],
        "agent_calls_by_seat": row["agent_calls_by_seat"], "agent_errors_by_seat": row["agent_errors_by_seat"],
        "final": economy(row["final_observed"]),
        "snapshots": {t: economy(s) for t, s in row["snapshots"].items()},
        "search": {key: search[key] for key in ("searched", "changed", "single", "aborted", "rolls", "depth")},
        "opponent_torch_threads": row["opponent_torch_threads"], "blas_pools": row["threadpools"],
        "call_seconds_by_seat": row["call_seconds_by_seat"], "started_at_utc": row["started_at_utc"],
        "finished_at_utc": row["finished_at_utc"], "wall_seconds": row["wall_seconds"],
        "actions_sha256": row["actions_sha256"], "final_state_sha256": row["final_state_sha256"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cpus", nargs=2, type=int, metavar=("SEAT0_CPU", "SEAT1_CPU"))
    args = ap.parse_args()
    if args.out.exists():
        ap.error("choose a fresh output")
    cpus = args.cpus or [None, None]
    if args.cpus and (len(set(cpus)) != 2 or not set(cpus) <= os.sched_getaffinity(0)):
        ap.error("two distinct available CPUs required")
    weight_dir = exam.ROOT / "data/weights/sapling_1200k"
    weight = weight_dir / ARMS[args.arm]
    prior = exam.prior
    import cnet2_pack
    import numpy
    import torch

    if prior.digest(prior.SOURCE, "md5") != prior.EXPECTED_MD5:
        raise ValueError("unexpected examiner source identity")

    with tempfile.TemporaryDirectory(prefix="sapling-elite-exam-") as tmp:
        local = Path(tmp) / weight.name
        shutil.copyfile(weight, local)
        pack = local.with_suffix(".py")
        cnet2_pack.QMODE = "int8"
        cnet2_pack.main(str(local), str(pack), teacher=2)
        if args.arm == "sqrt64" and prior.digest(pack) != prior.digest(weight_dir / "agent_int8.py"):
            raise ValueError("baseline repack differs from original checked-in release")
        result = {
            "schema": "sapling-elite-direct-3s2/1", "arm": args.arm, "map": SEED,
            "candidate_weight_sha256": prior.digest(local), "candidate_pack_sha256": prior.digest(pack),
            "examiner_md5": prior.digest(prior.SOURCE, "md5"),
            "configuration": exam.CFG, "mode": "full research diagnostic, two seats on one map",
            "examiner": "Original agent; TEMP=1, SEED=None, production search on, no registered wrapper",
            "timing": "Two seat workers at once. The examiner uses a wall clock deadline; load may differ by run.",
            "python": sys.version, "numpy": numpy.__version__, "torch": torch.__version__,
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
            "source_sha256": {str(p.relative_to(exam.ROOT)): prior.digest(p) for p in SOURCE_FILES},
        }
        jobs = [(args.arm, str(pack), side, cpus[side]) for side in (0, 1)]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            result["games"] = list(pool.map(run_seat, jobs))
        prior.shared.atomic_json(args.out, result)
        for game in result["games"]:
            print(args.arm, game["candidate_seat"], game["actual_transitions"],
                  game["termination"], game["candidate_official_margin"], flush=True)


if __name__ == "__main__":
    main()
