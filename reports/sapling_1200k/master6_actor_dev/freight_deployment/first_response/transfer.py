"""Run one registered two-seat arm and verify both action tapes with the original interpreter."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent
DEV = DEPLOY.parent
BASE = DEV.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from map_identity import local_board
from official_game_replay import audit
from tree_win_gate import verify_raw

PLAN_HASH = "cbc1e4868d8256b35af3b9d2f582c44b1c7139b6e8ae959dd84edcd698dfc7be"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(arm):
    plan_path = HERE / "transfer_plan.json"
    plan = json.loads(plan_path.read_bytes())
    if sha(plan_path) != PLAN_HASH:
        raise ValueError("registered transfer plan changed")
    receipt = json.loads((HERE / "map_preflight.json").read_bytes())
    if (receipt["verdict"] != "eligible_in_checked_scope" or receipt["candidate_seed"] != plan["seed"]
            or receipt["current_declaration"]["sha256"] != PLAN_HASH or receipt["conflicts"]
            or receipt["same_terrain_matches"] or receipt["reserved"] or receipt["ladder_seed_or_terrain_match"]):
        raise ValueError("development map clearance or plan identity mismatch")
    terrain = local_board(plan["seed"])
    if receipt["official_opening_sha256"] != terrain or receipt["candidate_initial_terrain_u8_sha256"] != terrain:
        raise ValueError("actual generated opening differs from sealed clearance")
    if arm == "parent":
        pack = ROOT / "data/weights/sapling_1200k/master6_actor_hybrid.py"
        manifest = DEV / "hybrid_manifest.json"
        catalog = json.loads(manifest.read_bytes())["input_and_output_sha256"]
        expected = plan["arms"]["parent"]
    else:
        pack = DEPLOY / "agent.py"
        manifest = DEPLOY / "manifest.json"
        build = json.loads(manifest.read_bytes())
        catalog = build["inputs_sha256"]
        expected = plan["arms"]["first_eligible"]
        if build["agent_sha256"] != expected:
            raise ValueError("candidate build disagrees with transfer plan")
    if sha(pack) != expected or any(sha(ROOT / key) != digest for key, digest in catalog.items()):
        raise ValueError("frozen package or source of weights differs")
    return plan["seed"], terrain, pack, manifest


def verify(run, arm):
    seed, terrain, pack, manifest = inputs(arm)
    report = json.loads((run / "results.json").read_bytes())
    if report["header"]["pack_sha256"] != sha(pack) or report["header"]["weight_sha256"] != sha(manifest):
        raise ValueError("completed match differs from the registered arm")
    if sorted((g["seed"], g["candidate_seat"]) for g in report["games"]) != [(seed, 0), (seed, 1)]:
        raise ValueError("missing or unexpected map/seat")
    official = audit(run)
    for row in report["games"]:
        raw = json.loads(gzip.decompress((run / row["raw_file"]).read_bytes()))
        if raw["configuration"].get("SEARCH_TEMP") not in (None, 1):
            raise ValueError("original search temperature changed")
        from tree_win_gate import validate_examiner_configuration
        validate_examiner_configuration(raw["configuration"])
        account = verify_raw(run, row, sha(pack), sha(manifest), income=True, expected_terrain=terrain)
        dest_row = next(g for g in official["games"] if g["seat"] == row["candidate_seat"])
        if dest_row["map_sha256"] != terrain:
            raise ValueError("official interpreter used different opening terrain")
        dest_row["income"] = account
    out = dict(arm=arm, plan_sha256=PLAN_HASH, receipt_sha256=sha(HERE / "map_preflight.json"),
               manifest_sha256=sha(manifest), examiner_md5=report["header"]["examiner_md5"],
               native_results_sha256=sha(run / "results.json"), **official)
    target = run / "official_check.json"
    if target.exists() and json.loads(target.read_bytes()) != out:
        raise ValueError("prior official receipt differs")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=["parent", "candidate"])
    ap.add_argument("--cpus", type=int, nargs=2, required=True, metavar=("SEAT0", "SEAT1"))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="verify frozen identities without starting any games")
    mode.add_argument("--verify-only", action="store_true", help="check completed action tapes without replaying the actors")
    opts = ap.parse_args()
    seed, _, pack, manifest = inputs(opts.arm)
    if len(set(opts.cpus)) != 2 or not set(opts.cpus) <= os.sched_getaffinity(0):
        ap.error("select two distinct available logical CPUs")
    folder = HERE / ("transfer_" + opts.arm)
    if not opts.check_only:
        if not opts.verify_only:
            environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            subprocess.run([sys.executable, "-u", str(BASE / "tree_pack_arena.py"), "--pack", str(pack),
                            "--weights", str(manifest), "--pack-sha256", sha(pack), "--run", str(folder),
                            "--cpus", *map(str, opts.cpus), "--maps", str(seed)], env=environment, check=True)
        verify(folder, opts.arm)
