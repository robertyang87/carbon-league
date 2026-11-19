"""Resumable complete-game research comparison against the genuine Master3s2."""
import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import statistics
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cash_v3_exam as cash

DEV_MAPS = (620106811, 620106812)
EXAMINER = "57f9a594cce4f2ec9e3573d8d9d26ca1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True, help="published agent, or fresh destination for repacking")
    p.add_argument("--run", type=Path, required=True, help="output directory; same command resumes")
    p.add_argument("--cpus", type=int, nargs=2, default=[0, 1])
    p.add_argument("--final-plan", type=Path, help="reserved one-use 100-map plan bound to frozen hashes")
    args = p.parse_args()
    if len(set(args.cpus)) != 2 or not set(args.cpus) <= os.sched_getaffinity(0):
        p.error("supply two distinct available CPU cores")
    args.run.mkdir(parents=True, exist_ok=True)
    import cnet2_pack
    import numpy
    import torch
    prior = cash.exam.prior
    sha = prior.digest
    if sha(prior.SOURCE, "md5") != EXAMINER:
        raise ValueError("incorrect examiner source")
    sources = [Path(__file__), Path(cash.__file__), Path(cash.exam.__file__), Path(prior.__file__),
               Path(prior.shared.__file__), prior.SOURCE]
    sources.extend(cash.exam.ROOT / "code" / name for name in
                   ("eval_v53.py", "fastsim.py", "cnet2_pack.py", "cnet2_agent.py", "cnet2_feat.py", "cnet2_np.py"))
    with (args.run / ".lock").open("a") as lock, tempfile.TemporaryDirectory(prefix="sapling-game-") as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        weights = Path(tmp) / args.weights.name
        shutil.copyfile(args.weights, weights)
        agent = weights.with_suffix(".py")
        cnet2_pack.QMODE = "int8"
        cnet2_pack.main(str(weights), str(agent), teacher=2)
        if not args.pack.exists():
            args.pack.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=args.pack.parent) as candidate:
                candidate.write(agent.read_bytes())
                candidate.flush()
                os.link(candidate.name, args.pack)
        if sha(args.pack) != sha(agent):
            raise ValueError("published and evaluated pack differ")
        plan = None
        if args.final_plan:
            from tree_win_gate import validate
            plan = json.loads(args.final_plan.read_text())
            validate(plan, sha(weights), sha(agent), EXAMINER, args.run)
            maps = [row["seed"] for row in plan["maps"]]
        else:
            maps = list(DEV_MAPS)
        header = {
            "version": 1, "mode": "final" if plan else "development", "seeds": maps, "cpus": args.cpus,
            "weight_sha256": sha(weights), "pack_sha256": sha(agent), "examiner_md5": EXAMINER,
            "configuration": cash.exam.CFG, "numpy": numpy.__version__, "torch": torch.__version__,
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "sources_sha256": {str(x.relative_to(cash.exam.ROOT)): sha(x) for x in sources},
            "plan_sha256": sha(args.final_plan) if plan else None,
            "scope": "Two persistent single-core seat workers, fresh examiner per game, no seed override. "
                     "The examiner uses the wall clock; underlying opponent trajectories can differ. "
                     "No platform resource certification.",
        }
        report_path = args.run / "results.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["header"] != header:
                raise ValueError("resume identities/environment differ")
        else:
            report = dict(header=header, status="running", started=prior.utc(), games=[])
        index = {(g["seed"],g["candidate_seat"]):g for g in report["games"]}
        allowed = {(seed, seat) for seed in maps for seat in (0,1)}
        if not set(index) <= allowed or len(index) != len(report["games"]):
            raise ValueError("unexpected or duplicate stored game")
        prior.shared.atomic_json(report_path, report)
        for seed in maps:
            missing = [side for side in (0,1) if (seed,side) not in index]
            if not missing:
                continue
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(cash.game, (weights.stem, str(agent), seed, side, args.cpus[side]))
                           for side in missing]
                for future in as_completed(futures):
                    result = future.result()
                    report["games"].append(result)
                    report["games"].sort(key=lambda r:(r["seed"],r["candidate_seat"]))
                    prior.shared.atomic_json(report_path, report)
                    print(seed, result["candidate_seat"], result["result"], result["margin"], flush=True)
        if report.get("status") == "complete":
            return
        report.update(status="complete", finished=prior.utc(), mean_margin=statistics.mean(g["margin"] for g in report["games"]))
        if plan:
            from tree_win_gate import verdict
            report["verdict"] = verdict(plan, report["games"])
        prior.shared.atomic_json(report_path, report)


if __name__ == "__main__":
    main()
