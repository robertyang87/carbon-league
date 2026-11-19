"""Replay the two registered hosted matches from their public command receipts."""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOST = HERE.parents[1] / "ladder_live"
sys.path.insert(0, str(HOST))
import live


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read(path):
    return json.loads(path.read_bytes())


def search_summary(rows, seat):
    """An aborted comparison may still finish a depth. Use counter deltas per call."""
    previous = Counter()
    types, depths, candidates = Counter(), Counter(), Counter()
    cpu, wall = [], []
    for row in rows:
        call = row["calls"][seat]
        if call["status"] != "ok":
            types["no_successful_reply"] += 1
            continue
        response = call["response"]
        if "public_cpu_seconds" in response:
            cpu.append(response["public_cpu_seconds"])
            wall.append(response["public_wall_seconds"])
        counts = Counter(response["search_counts"])
        require(all(previous[k] <= counts[k] <= previous[k] + 1 for k in counts),
                "invalid cumulative original search counts")
        delta = counts - previous
        previous = counts
        kind = "searched" if delta["searched"] else "single" if delta["single"] else (
            "zero_depth_return" if delta["aborted"] else "unclassified")
        types[kind] += 1
        last = response.get("search_last")
        if last is not None and kind != "unclassified":
            depths[last["depth"]] += 1
            candidates[last["candidates"]] += 1
    return {"replies_by_type": dict(types), "last_cumulative_counts": dict(previous),
            "depth_histogram": dict(depths), "candidate_histogram": dict(candidates),
            "instrumented_cpu_wall_calls": len(cpu), "public_cpu_seconds": sum(cpu) if cpu else None,
            "public_wall_seconds": sum(wall) if wall else None}


def verify_game(directory, seed, seat, arm, matrix_file):
    matrix = read(matrix_file)
    ticket = directory / "registration.json"
    registration = read(ticket)
    require(seed in matrix["seeds"] and seat in (0, 1), "unregistered map or seat")
    require(registration["matrix_sha256"] == live.sha(matrix_file), "wrong matrix")
    require((registration["seed"], registration["candidate_seat"], registration["arm"]) ==
            (seed, seat, arm), "different registered game")
    terrain = registration["physical_map_sha256"]
    require(terrain == matrix["physical_boards"][str(seed)], "wrong physical terrain")
    base = read(HOST / "protocol.json")
    require(registration["platform"] == base["platform"], "platform settings differ")
    for role, package in (("candidate", arm), ("candidate_weights", arm + "_weights"), ("opponent", "opponent")):
        entry = registration["files"][role]
        frozen = matrix["files"][package]
        require(entry["sha256"] == frozen["sha256"] and entry["path"] == frozen["path"],
                f"wrong {role} package")
    for key, entry in registration["files"].items():
        if key in ("official_helpers", "environment", "rules"):
            require(entry == base["files"][key], f"wrong {key}")
        path = live.ROOT / entry["path"]
        require(path.stat().st_size == entry["bytes"] and live.sha(path) == entry["sha256"],
                f"registered file mismatch: {key}")

    # Selecting the match config only sets globals. The replay never loads either policy.
    live.configure(ticket, directory / "attempt.json", directory / ".unused-replay-lock")
    receipt = read(directory / "receipt.json")
    require(receipt["seed"] == seed and receipt["candidate_seat"] == seat, "wrong receipt")
    require(receipt["protocol_sha256"] == live.sha(ticket), "unbound receipt")
    for name, key in (("live.py", "runner_sha256"), ("worker.py", "worker_sha256")):
        require(receipt[key] == live.sha(HOST / name), f"{name} SHA differs")
    status = receipt["status"]
    scored_statuses = {"complete", "forfeit", "bilateral_failure"}
    if "probes" in receipt:
        issues = live.eligibility(receipt["probes"], receipt["containers"])
        require(issues == receipt["blockers"], "cgroup probe contradicts receipt")
    if status in scored_statuses:
        require(receipt.get("blockers") == [] and receipt["strategy_initializations"] == 2,
                "attempted game lacks eligible containers")
        for container in receipt["containers"]:
            require(container["image"] == receipt["image_id"], "image mismatch")

    initial = receipt.get("initialization", [])
    if initial:
        require(len(initial) == 2, "missing physical seat on initialization")
    for s, call in enumerate(initial):
        require(call["limit"] == 20 and math.isfinite(call["elapsed"]) and call["elapsed"] >= 0,
                "invalid initialization clock")
        if call["status"] == "ok":
            answer = call["response"]
            require(call["elapsed"] <= 20 and answer["ready"] is True, "late or unready")
            require(answer["sha256"] == registration["files"][live.ROLES[s]]["sha256"],
                    "wrong package at cold start")
            if s != seat:
                require(answer["original"] == {"_TEMP": 1, "_SEED": None, "_S_ABORT": .58, "_SEARCH_OK": True},
                        "examiner runtime configuration differs")
    terminal = live.failures(initial) if initial else None
    tape = directory / "turns.jsonl"
    if tape.exists():
        require(live.sha(tape) == receipt["turns_sha256"], "bad tape digest")
        rows = [json.loads(line) for line in tape.read_bytes().splitlines()]
    else:
        require("turns_sha256" not in receipt, "missing tape")
        rows = []

    env, _ = live.official()  # checks the initial full-board SHA against the registration
    bank, slow = [5., 5.], [0, 0]
    for index, row in enumerate(rows):
        require(not terminal and not env.done and row["step"] == env.states[0].observation.step == index,
                "recorded action after termination, missing step, or duplicate step")
        require(len(row["calls"]) == len(row["public_observation_sha256"]) == 2, "missing seat")
        for s, call in enumerate(row["calls"]):
            obs = live.observe(env, s, bank[s])
            digest = hashlib.sha256(live.encoded(obs)).hexdigest()
            require(digest == row["public_observation_sha256"][s], f"observation mismatch at {index}/{s}")
            elapsed = call["elapsed"]
            limit = live.allowance(slow[s], bank[s])
            require(math.isfinite(elapsed) and elapsed >= 0 and call["limit"] == limit, "invalid step clock")
            require(call["status"] in ("ok", "timeout", "error"), "unrecognized call status")
            if call["status"] == "ok":
                require(elapsed <= limit, "successful call exceeded time limit")
                checked = call.copy()
                live.check_action(checked, obs)
                require(checked["status"] == "ok", "invalid successful action")
                if elapsed > 1:
                    slow[s] += 1
                    bank[s] = max(0., bank[s] - (elapsed - 1))
        require(slow == row["slow_steps"] and bank == row["overage_remaining_seconds"], "overage mismatch")
        terminal = live.failures(row["calls"])
        if not terminal:
            env.step([call["response"]["action"] for call in row["calls"]])
    transitions = env.states[0].observation.step
    require(transitions == receipt["official_transitions"], "official transition count mismatch")
    if status in scored_statuses:
        require(len(initial) == 2 and receipt["action_calls"] == [len(rows)] * 2, "call count mismatch")
    if initial:
        require(live.timing_by_seat(receipt, rows) == receipt["timing_by_seat"], "per-seat clock mismatch")

    scores = None
    if status == "complete":
        require(not terminal and env.done and all(s.status == "DONE" for s in env.states), "premature score")
        scores = [s.reward for s in env.states]
        winner = None if scores[0] == scores[1] else int(scores[1] > scores[0])
        require(receipt["counted"] is True and scores == receipt["official_scores_by_seat"]
                and winner == receipt["winner_seat"], "official terminal result mismatch")
    elif status in ("forfeit", "bilateral_failure"):
        require(terminal and all(receipt.get(k) == v for k, v in terminal.items()), "wrong forfeit")
        require(receipt.get("official_scores_by_seat") is None, "spurious forfeit score")
    else:
        require(status in ("whole_task_timeout", "BLOCKED_BEFORE_AGENT_LOAD", "infrastructure_error")
                and receipt["counted"] is False, "unknown or incorrectly scored infrastructure status")
    return {"arm": arm, "seed": seed, "seat": seat, "terrain": terrain, "status": status,
            "counted": receipt["counted"], "winner_seat": receipt.get("winner_seat"),
            "official_scores_by_seat": scores, "official_transitions": transitions,
            "replayed_public_observations": 2 * len(rows), "image_id": receipt["image_id"],
            "workflow_run": registration.get("workflow_run"), "examiner_search": search_summary(rows, 1 - seat),
            "timing_by_seat": receipt.get("timing_by_seat"), "whole_wall_seconds": receipt["whole_wall_seconds"],
            "receipt_sha256": live.sha(directory / "receipt.json")}


def verify_pair(directory, seed, seat):
    games, errors = {}, {}
    for arm in ("parent", "student"):
        try:
            games[arm] = verify_game(directory / arm, seed, seat, arm, HERE / "protocol.json")
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            errors[arm] = f"{type(exc).__name__}: {exc}"
    if len(games) == 2:
        for key in ("image_id", "workflow_run"):
            if games["parent"][key] != games["student"][key]:
                errors["pair"] = f"different {key}"
    return {"seed": seed, "seat": seat, "games": games, "verification_errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--seat", required=True, choices=(0, 1), type=int)
    args = parser.parse_args()
    report = verify_pair(args.out, args.seed, args.seat)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    sys.exit(int(bool(report["verification_errors"])))
