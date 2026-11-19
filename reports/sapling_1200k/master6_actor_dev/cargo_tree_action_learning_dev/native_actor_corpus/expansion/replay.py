"""Reconstruct the archived physical action tapes in the official interpreter, with no agents loaded."""
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

FREEZE = "5567aaddaabd3cc86eb7863fdc4c38a640878e48"
REPORT = Path("reports/sapling_1200k")
ENGINE = Path("code/env_src/carbon_challenge-master")
ENGINE_INPUTS = [
    "zerosum_env/core.py", "zerosum_env/envs/carbon/carbon.py", "zerosum_env/envs/carbon/helpers.py",
    "zerosum_env/envs/carbon/idgen.py", "zerosum_env/envs/carbon/carbon.json",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache
def tracked(repo):
    tree = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-rz", FREEZE, "--", str(REPORT), str(ENGINE)])
    return {Path(name.decode()): attrs.split()[2].decode() for row in tree.split(b"\0") if row
            for attrs, name in [row.split(b"\t", 1)]}


def checked_bytes(repo, path):
    expected = tracked(repo)[path]
    local = repo / path
    data = (local.read_bytes() if local.is_file() else
            subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", expected]))
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != expected:
        raise ValueError(f"working file differs from {FREEZE}: {path}")
    return data


@lru_cache
def document(repo, path):
    content = checked_bytes(repo, path)
    return json.loads(content), {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


@lru_cache
def official_rules(repo):
    for source in ENGINE_INPUTS:
        checked_bytes(repo, ENGINE / source)
    sys.path.insert(0, str(repo / ENGINE))
    from zerosum_env import make
    return make


@lru_cache
def batch_receipts(repo):
    index = defaultdict(list)
    for path in sorted(tracked(repo)):
        if path.suffix != ".json" or "official" not in path.name or not (
                path.name in {"official_replay.json", "official_check.json", "official_and_prefix.json"} or path.parent == REPORT):
            continue
        data, reference = document(repo, path)
        if "official_replay" in data:
            data = data["official_replay"]
        runs = [data] if "games" in data else data.get("runs", {}).values()
        for run in runs:
            for game in run["games"]:
                if "transcript_sha256" in game:
                    index[game["transcript_sha256"]].append((game, run.get("pack_sha256"), reference))
    return index


def receipts(repo, path, raw, raw_sha):
    seed, seat = raw["seed"], raw["candidate_seat"]
    root = path.parent
    native, historical = [], list(batch_receipts(repo)[raw_sha])
    for name in (f"receipt_{seed}_{seat}.json", f"official_seat{seat}.json", f"native_seat{seat}.json", "results.json"):
        file = root / name
        if file not in tracked(repo):
            continue
        data, reference = document(repo, file)
        if "official" in data:
            historical.append((data["official"], data.get("pack_sha256"), reference))
        if "game" in data:
            games = [data["game"]]
        elif isinstance(data.get("games"), list):
            games = [g for g in data["games"] if g.get("raw_sha256") == raw_sha]
        elif "raw_sha256" in data and "actual_transitions" in data:
            games = [data]
        else:
            games = []
        for g in games:
            native.append((g, reference))
        if "native_receipt_sha256" in data:
            if data["native_receipt_sha256"] is not None:
                sibling = root / f"native_seat{seat}.json"
                if document(repo, sibling)[1]["sha256"] != data["native_receipt_sha256"]:
                    raise ValueError(f"detached native receipt mismatch at {file}")
    return native, historical


def replay(raw, path, repo):
    """Return physical identities, official terminal cash and diagnostics for one frozen archive path."""
    import numpy as np
    repo = Path(repo).resolve()
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(repo)
    payload = checked_bytes(repo, path)
    if raw != json.loads(gzip.decompress(payload)):
        raise ValueError(f"input differs from frozen compressed payload: {path}")
    raw_sha = hashlib.sha256(payload).hexdigest()
    seed, seat = raw["seed"], raw["candidate_seat"]
    actions, cfg = raw["actions"], raw["game_configuration"]
    issues = []
    def check(label, actual, expected):
        if actual != expected:
            issues.append(label)

    make = official_rules(repo)
    np.random.seed(seed)
    env = make("carbon", configuration={**cfg, "randomSeed": seed})
    state = env.states[0].observation
    initial = list(state.full_carbon)
    physical = hashlib.sha256(bytes(initial)).hexdigest()
    check("physical_seat", seat in (0, 1), True)
    check("starts_at_zero_with_active_seats", [state.step] + [a.status for a in env.states], [0, "ACTIVE", "ACTIVE"])
    if "initial_full_carbon" in raw:
        check("raw_opening_cells", raw["initial_full_carbon"], initial)
        check("raw_opening_sha256", raw["initial_terrain_u8_sha256"], physical)
    check("raw_action_count", len(actions), raw["steps"])
    money = [[p[0] for p in state.players]]
    unknown = [0, 0]
    for tick, joint in enumerate(actions[:raw["steps"]]):
        if env.done or env.states[0].observation.step != tick:
            issues.append("action_after_terminal_or_wrong_tick")
            break
        if not isinstance(joint, list) or len(joint) != 2 or any(not isinstance(d, dict) for d in joint):
            issues.append("malformed_joint_action")
            break
        for s in (0, 1):
            _, bases, workers, _ = env.states[0].observation.players[s]
            unknown[s] += sum(uid not in bases and uid not in workers for uid in joint[s])
        env.step(joint)
        money.append([p[0] for p in env.states[0].observation.players])
    n = len(money) - 1
    state = env.states[0].observation
    check("replayed_every_action", n, len(actions))
    check("official_terminal", env.done, True)
    statuses = [a.status for a in env.states]
    check("official_done_both", statuses, ["DONE", "DONE"])
    check("official_step", state.step, n)
    rewards = [a.reward for a in env.states]
    extinct = [not workers and not trees and (not bases or cash < min(cfg["recCollectorCost"], cfg["recPlanterCost"]))
               for cash, bases, workers, trees in state.players]
    check("extinction_by_seat", extinct, raw["extinct_by_seat"])
    final_pre = money[-2] if n else None
    check("recorded_last_pre_cash", final_pre, raw["last_pre_cash"])
    check("recorded_last_post_cash", money[-1], raw["last_post_cash"])
    check("official_rewards", rewards, raw["official_score_by_seat"])
    margin = rewards[seat] - rewards[1-seat]
    result = "win" if margin > 0 else "loss" if margin < 0 else "draw"
    check("signed_physical_result", result, raw["result"])
    check("signed_physical_margin", raw["margin"] is not None and abs(margin - raw["margin"]) < 1e-7, True)
    actions_sha = digest(actions)
    native, historical = receipts(repo, path, raw, raw_sha)
    native_snapshot_steps = set()
    for game, _ in native:
        for key, value in {"raw_sha256": raw_sha, "seed": seed, "candidate_seat": seat, "actual_transitions": n,
                           "official_score_by_seat": rewards, "result": result, "extinct_by_seat": extinct}.items():
            check("native_" + key, game.get(key), value)
        if "actions_sha256" in game:
            check("native_actions_sha256", game["actions_sha256"], actions_sha)
        for snap in [*game.get("snapshots", {}).values(), game.get("final", {})]:
            for role, s in [("candidate", seat), ("opponent", 1-seat)]:
                if role in snap and "state_t" in snap[role]:
                    t = snap[role]["state_t"]
                    check("archived_cash_snapshot", money[t][s] if 0 <= t <= n else None, snap[role]["cash"])
                    native_snapshot_steps.add(t)
    for game, pack, _ in historical:
        for key, value in {"seed": seed, "seat": seat, "map_sha256": physical, "transcript_sha256": raw_sha,
                           "steps": n, "statuses": statuses, "official_reward_by_seat": rewards, "result": result}.items():
            check("historical_official_" + key, game.get(key), value)
        if pack is not None:
            check("historical_official_pack", pack, raw["pack_sha256"])
    return {
        "path": str(path), "raw_sha256": raw_sha, "seed": seed, "candidate_seat": seat, "expert_seat": 1 - seat,
        "physical_sha256": physical, "trajectory_sha256": digest([cfg, physical, actions]),
        "actions_sha256": actions_sha, "physical_seat_action_sha256": [digest([a[s] for a in actions]) for s in (0, 1)],
        "transitions": n, "unexecuted_captured_action_count": len(actions) - n, "full_official_terminal": bool(env.done and statuses == ["DONE", "DONE"] and n == len(actions)),
        "verified": not issues, "issues": sorted(set(issues)), "statuses": statuses,
        "official_score_by_seat": rewards if env.done else None, "recorded_result": raw["result"],
        "recorded_score_by_seat": raw["official_score_by_seat"],
        "result": result if env.done else None, "extinct_by_seat": extinct,
        "opening_evidence": "raw_225_and_replay" if "initial_full_carbon" in raw else "seed_configuration_reconstruction",
        "unrecognized_or_unowned_command_ids_by_seat": unknown,
        "official_cash_path_sha256": digest(money), "official_cash_observations": len(money),
        "terminal_pre_cash": final_pre, "terminal_post_cash": money[-1] if env.done else None,
        "historical_cash_snapshot_ticks_checked": sorted(native_snapshot_steps),
        "historical_native_receipts": [ref for _, ref in native],
        "historical_official_receipts": [ref for _, _, ref in historical],
    }


def replay_path(path):
    repo = Path(__file__).resolve().parents[6]
    payload = checked_bytes(repo, path)
    return replay(json.loads(gzip.decompress(payload)), path, repo)


if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[6]
    files = sorted(p for p in tracked(base) if p.match("raw_*_[01].json.gz"))
    with multiprocessing.Pool(args.workers) as pool:
        for row in pool.imap(replay_path, files):
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
