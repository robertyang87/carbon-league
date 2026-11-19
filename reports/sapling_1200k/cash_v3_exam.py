"""Complete unseeded-search Master3s2 games on the frozen cash-V3 research maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

import external_3s2 as exam

MAPS = (620095811, 620095812)
ARMS = {"top5_h2h_v2", "cash_v3"}
ECONOMY = ("state_t", "cash", "cargo", "alive_workers", "alive_trees", "new_trees_planted",
           "positive_delivery_events", "own_positive_delivery_carbon", "closed_loop_events",
           "collected_carbon", "cumulative_tree_absorption_income", "recruited_total")


def game(job):
    arm, pack, seed, side, cpu = job
    os.sched_setaffinity(0, {cpu})
    row = exam.play((arm, pack, seed, side))
    if not row["full_game"] or any(row["agent_errors_by_seat"]):
        raise RuntimeError(f"Incomplete or erroneous outcome: {row['errors']}")
    deliveries = [event for event in row["positive_delivery_and_plant_events"] if event["event"] == "deposit"]
    def economy(snapshot, t):
        result = {}
        for role, seat in (("candidate", side), ("opponent", 1-side)):
            state = snapshot[role]
            events = [e for e in deliveries if e["player"] == seat and e["step"] <= t]
            if (len(events) != state["positive_delivery_events"] or
                    abs(sum(e["amount"] for e in events) - state["own_positive_delivery_carbon"]) > 1e-6):
                raise RuntimeError(f"Deposit accounting disagreement at {seed}/{seat}/{t}")
            result[role] = {key: state[key] for key in ECONOMY}
            result[role]["distinct_delivery_turns"] = len({e["step"] for e in events})
        return result
    snapshots = {t: economy(s, int(t)) for t, s in row["snapshots"].items()}
    return {
        "seed": seed, "candidate_seat": side, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "actual_transitions": row["actual_transitions"], "termination": row["termination"],
        "extinct_by_seat": row["extinct_by_seat"], "result": row["result"],
        "official_score_by_seat": row["official_score_by_seat"], "margin": row["candidate_official_margin"],
        "snapshots": snapshots, "final": economy(row["final_observed"], row["actual_transitions"]),
        "agent_errors_by_seat": row["agent_errors_by_seat"], "agent_calls_by_seat": row["agent_calls_by_seat"],
        "search": row["opponent_search_stats"], "threadpools": row["threadpools"],
        "opponent_torch_threads": row["opponent_torch_threads"], "wall_seconds": row["wall_seconds"],
        "call_seconds_by_seat": row["call_seconds_by_seat"], "actions_sha256": row["actions_sha256"],
        "final_state_sha256": row["final_state_sha256"],
        "positive_deposits": [{"t": e["step"], "seat": e["player"], "amount": e["amount"]}
                              for e in deliveries],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpus", nargs=2, required=True, type=int)
    args = p.parse_args()
    if args.out.exists() or len(set(args.cpus)) != 2 or not set(args.cpus) <= os.sched_getaffinity(0):
        p.error("a fresh output and two distinct available CPUs are required")
    import cnet2_pack
    import numpy
    import torch
    prior = exam.prior
    if prior.digest(prior.SOURCE, "md5") != prior.EXPECTED_MD5:
        raise ValueError("unexpected original Master3s2")
    directory = exam.ROOT / "data/weights/sapling_1200k"
    weight = directory / (args.arm + ".npz")
    published = directory / (args.arm + "_agent.py")
    sources = [Path(__file__), Path(exam.__file__), Path(prior.__file__), Path(prior.shared.__file__), prior.SOURCE]
    sources += [exam.ROOT / "code" / name for name in
                ("eval_v53.py", "fastsim.py", "cnet2_pack.py", "cnet2_agent.py", "cnet2_feat.py", "cnet2_np.py")]
    with tempfile.TemporaryDirectory(prefix="sapling-cash-exam-") as tmp:
        local = Path(tmp) / weight.name
        shutil.copyfile(weight, local)
        pack = local.with_suffix(".py")
        cnet2_pack.QMODE = "int8"
        cnet2_pack.main(str(local), str(pack), teacher=2)
        if args.arm == "cash_v3" and not published.exists():
            with published.open("xb") as f:
                f.write(pack.read_bytes())
        if prior.digest(published) != prior.digest(pack):
            raise ValueError("packed bytes differ from published version")
        report = {
            "schema": "sapling-cash-v3-full-matches/1", "arm": args.arm, "maps": list(MAPS), "seats": [0,1],
            "status": "running", "candidate_weight_sha256": prior.digest(local), "pack_sha256": prior.digest(pack),
            "examiner_md5": prior.digest(prior.SOURCE, "md5"), "source_sha256":
                {str(path.relative_to(exam.ROOT)): prior.digest(path) for path in sources},
            "configuration": exam.CFG, "utc_started": prior.utc(), "python": sys.version,
            "numpy": numpy.__version__, "torch": torch.__version__, "workers": 2, "cpus": args.cpus,
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "scope": "Two preselected research maps, both seats; not a registered strength exam. "
                     "Master3s2 original sourced agent, TEMP=1, SEED=None, time-limited search. "
                     "Maps and CPU pinning match across arms; actual examiner trajectories and workloads can differ. "
                     "Normal deliveries are positive smelted cargo at an owned base.",
            "games": [],
        }
        prior.shared.atomic_json(args.out, report)
        for seed in MAPS:
            # Finish both seats of a map before starting the next pair; equal worker/affinity load shape.
            jobs = [(args.arm, str(pack), seed, side, args.cpus[side]) for side in (0,1)]
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
                for result in pool.map(game, jobs):
                    report["games"].append(result)
                    prior.shared.atomic_json(args.out, report)
                    print(args.arm, seed, result["candidate_seat"], result["actual_transitions"],
                          result["margin"], result["termination"], flush=True)
        report["utc_finished"] = prior.utc()
        report["status"] = "complete"
        report["mean_margin"] = statistics.fmean(r["margin"] for r in report["games"])
        report["t100_available"] = sum("100" in r["snapshots"] for r in report["games"])
        prior.shared.atomic_json(args.out, report)


if __name__ == "__main__":
    main()
