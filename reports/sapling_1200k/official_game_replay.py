"""Replay recorded agent commands in the repository's original game interpreter."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "code/env_src/carbon_challenge-master"
sys.path.insert(0, str(ENGINE))
from zerosum_env import make


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(run, row):
    payload = (run / row["raw_file"]).read_bytes()
    if digest(payload) != row["raw_sha256"]:
        raise ValueError("transcript checksum mismatch")
    raw = json.loads(gzip.decompress(payload))
    seed, seat = row["seed"], row["candidate_seat"]
    if (raw["seed"], raw["candidate_seat"]) != (seed, seat):
        raise ValueError("unexpected map or seat")
    # populate_board samples the initial carbon from the global NumPy RNG.
    # make() generates map one, a subsequent env.reset() would draw map two.
    np.random.seed(seed)
    env = make("carbon", configuration={**raw["game_configuration"], "randomSeed": seed})
    initial = env.states[0].observation.full_carbon
    initial_map = digest(bytes(initial))
    actions = raw["actions"]
    if not actions:
        raise ValueError("no actions")
    for turn, commands in enumerate(actions, 1):
        if env.done:
            raise ValueError(f"extra turn {turn} after official terminal")
        env.step(commands)
    statuses = [agent.status for agent in env.states]
    rewards = [agent.reward for agent in env.states]
    step = env.states[0].observation.step
    if (not env.done or statuses != ["DONE", "DONE"] or step != len(actions)
            or step != row["actual_transitions"] or step != raw["steps"]
            or rewards != row["official_score_by_seat"] or rewards != raw["official_score_by_seat"]):
        raise ValueError(f"official outcome mismatch for {seed}/{seat}: {step}, {statuses}, {rewards}")
    diff = rewards[seat] - rewards[1-seat]
    result = "win" if diff > 0 else "loss" if diff < 0 else "draw"
    if result != row["result"] or abs(row["margin"] - diff) > 1e-7:
        raise ValueError(f"winner or margin mismatch on {seed}/{seat}")
    return {"seed": seed, "seat": seat, "map_sha256": initial_map, "transcript_sha256": row["raw_sha256"],
            "steps": step, "statuses": statuses, "official_reward_by_seat": rewards, "result": result}


def audit(run):
    report = json.loads((run / "results.json").read_bytes())
    if report["status"] != "complete":
        raise ValueError(f"incomplete run {run}")
    header = report["header"]
    seeds = header.get("seeds")
    # V4r used an earlier ledger schema. Its recorded development block was one map.
    if seeds is None and header["pack_sha256"] == "12aeeb3718b62b1271fadea63debb71a9c22454cd61dd97b91aeab59590870e0":
        seeds = [620106811]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("missing or repeated map in run header")
    expected = {(s, p) for s in seeds for p in (0, 1)}
    checked = [check(run, row) for row in report["games"]]
    if len(checked) != len(expected) or {(r["seed"], r["seat"]) for r in checked} != expected:
        raise ValueError(f"incomplete map and seat set in {run}")
    for s in seeds:
        if len({r["map_sha256"] for r in checked if r["seed"] == s}) != 1:
            raise ValueError(f"seat exchange did not use the same map for {s}")
    return {"pack_sha256": report["header"]["pack_sha256"], "games": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or len({p.name for p in args.run}) != len(args.run):
        parser.error("supply a new output path and distinct run names")
    paths = ["zerosum_env/core.py", "zerosum_env/envs/carbon/carbon.py", "zerosum_env/envs/carbon/helpers.py",
             "zerosum_env/envs/carbon/idgen.py", "zerosum_env/envs/carbon/carbon.json"]
    runs = {run.name: audit(run) for run in args.run}
    with args.out.open("x") as target:
        json.dump({"engine_source_sha256": {p:digest((ENGINE/p).read_bytes()) for p in paths},
                   "runs":runs}, target, indent=2, allow_nan=False)
        target.write("\n")
