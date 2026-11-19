"""Resumable, source-audited match against original Master3s2 for frozen candidate packs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile

import tree_audited_game as arena
import map_identity
import native_clock

PRIOR = arena.prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True,
                   help="source NPZ for standard candidate or frozen build manifest for a compound pack")
    p.add_argument("--pack-sha256", required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--cpus", type=int, nargs=2, default=[0,1])
    maps = p.add_mutually_exclusive_group()
    maps.add_argument("--maps", type=int, nargs="+", default=[620106811])
    maps.add_argument("--final-plan", type=Path)
    args = p.parse_args()
    weight_sha = PRIOR.digest(args.weights)
    if PRIOR.digest(args.pack) != args.pack_sha256 or PRIOR.digest(PRIOR.SOURCE, "md5") != PRIOR.EXPECTED_MD5:
        p.error("candidate or examiner identity mismatch")
    if len(set(args.cpus)) != 2 or not set(args.cpus) <= os.sched_getaffinity(0):
        p.error("two distinct available CPU cores are required")
    from tree_win_gate import RESERVED_RANGE, prove_candidate, protocol_hashes, validate, validate_game, verdict, verify_raw
    plan = None
    if args.final_plan:
        plan = json.loads(args.final_plan.read_text())
        rebuilt_weights, rebuilt_pack = prove_candidate(args.weights, args.pack)
        seeds = validate(plan, rebuilt_weights, rebuilt_pack, PRIOR.EXPECTED_MD5, args.run)
    else:
        seeds = args.maps
        if any(RESERVED_RANGE[0] <= s < RESERVED_RANGE[1] for s in seeds):
            p.error("these seeds are reserved for independent formal exams")
    terrains = {r["seed"]:r["initial_terrain_u8_sha256"] for r in plan["maps"]} if plan else {}
    cpus = native_clock.assignments(plan, args.cpus)
    play_match = arena.play
    if native_clock.required(plan):
        play_match = native_clock.clocked_play
    elif plan is not None:
        from compound_formal import D1
        if args.pack_sha256 == D1:
            from d1_sapling_arena import play as play_match
    if len(set(seeds)) != len(seeds):
        p.error("duplicate map")
    args.run.mkdir(parents=True, exist_ok=True)
    scorefile = args.run / "results.json"
    with (args.run / ".lock").open("a") as lock, tempfile.TemporaryDirectory() as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pack = Path(tmp)/args.pack.name
        shutil.copyfile(args.pack, pack)
        if PRIOR.digest(pack) != args.pack_sha256:
            raise ValueError("changed while copying")
        meta = dict(weight_sha256=weight_sha, pack_sha256=args.pack_sha256, examiner_md5=PRIOR.EXPECTED_MD5,
                    source_sha256=protocol_hashes(), seeds=seeds, cpus=args.cpus, configuration=arena.exam.CFG,
                    cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
                    plan_sha256=PRIOR.digest(args.final_plan) if plan else None)
        if scorefile.exists():
            report = json.loads(scorefile.read_text())
            if report["header"] != meta:
                raise ValueError("resume identities differ")
        else:
            report = dict(header=meta, started=PRIOR.utc(), status="running", games=[])
        allowed = {(s,k) for s in seeds for k in (0,1)}
        done = {(r["seed"],r["candidate_seat"]) for r in report["games"]}
        if not done <= allowed or len(done) != len(report["games"]):
            raise ValueError("duplicate or unassigned stored game")
        for r in report["games"]:
            if PRIOR.digest(args.run / r["raw_file"]) != r["raw_sha256"]:
                raise ValueError("raw evidence changed")
        PRIOR.shared.atomic_json(scorefile, report)
        for seed in seeds:
            missing = [s for s in (0,1) if (seed,s) not in done]
            if not missing:
                continue
            orphans = [args.run / f"raw_{seed}_{s}.json.gz" for s in missing]
            if any(p.exists() for p in orphans):
                report.update(status="interrupted_after_raw_save", finished=PRIOR.utc())
                PRIOR.shared.atomic_json(scorefile, report)
                raise ValueError("preserved an already played raw outcome; recover that record before resuming")
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
                jobs = [pool.submit(play_match, (str(pack), weight_sha, seed, s, cpus.get((seed, s), args.cpus[s]), str(args.run), args.pack_sha256)) for s in missing]
                for future in as_completed(jobs):
                    row = future.result()
                    report["games"].append(row)
                    report["games"].sort(key=lambda r:(r["seed"],r["candidate_seat"]))
                    PRIOR.shared.atomic_json(scorefile, report)
                    print(seed,row["candidate_seat"],row["result"],row["margin"],row["audit_messages"],flush=True)
            if any(not r["valid"] for r in report["games"]):
                break
        invalid = []
        for row in report["games"]:
            try:
                require_income = plan is not None and plan["schema"] == map_identity.SCHEMA
                account = verify_raw(args.run, row, args.pack_sha256, weight_sha, formal=plan is not None,
                                     income=require_income, record_compound=plan is not None,
                                     expected_terrain=terrains.get(row["seed"]),
                                     clock_cpu=cpus.get((row["seed"], row["candidate_seat"])))
                if require_income:
                    row["income_account"] = account
                validate_game(row)
            except (KeyError, TypeError, ValueError) as exc:
                invalid.append([row["seed"], row["candidate_seat"], str(exc)])
        if invalid:
            report.update(status="invalid", errors=invalid, finished=PRIOR.utc())
            PRIOR.shared.atomic_json(scorefile, report)
            raise ValueError("game evidence was preserved; source or score audit failed")
        if len(report["games"]) != len(allowed):
            raise ValueError("incomplete scheduled games")
        if plan:
            report["verdict"] = verdict(plan, report["games"])
        if report.get("status") != "complete":
            report.update(status="complete", finished=PRIOR.utc())
            PRIOR.shared.atomic_json(scorefile, report)


if __name__ == "__main__":
    main()
