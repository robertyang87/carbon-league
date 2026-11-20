"""Verify frozen ticket outputs by replaying every executed action in the official referee."""
import argparse
import gzip
import json
import math
from pathlib import Path

import run


def load(path):
    content = path.read_bytes()
    return json.loads(gzip.decompress(content) if path.suffix == ".gz" else content)


def match_header(obj, header):
    for field in ("seed", "seat", "kind", "protocol", "terrain", "actors", "configuration", "runner_sha256"):
        run.need(obj[field] == header[field], "Header drift in " + field)


def replay(header, rows, expected):
    cfg, seat = header["configuration"], header["seat"]
    env = run.collect.begin(header["seed"], cfg)
    run.need(run.sha(bytes(env.states[0].observation.full_carbon)) == header["terrain"], "Wrong terrain")
    bank = [5., 5.]
    for i, row in enumerate(rows):
        run.need(not env.done and row["step"] == i and row["bank_before"] == bank, "Missed turn or bank")
        v = [run.public(env, s, bank) for s in (0, 1)]
        run.need(run.digest(row["public"]) == run.digest(v[seat]) and
                 row["enemy_public_sha256"] == run.digest(v[1-seat]) and row["board_before"] == run.board_id(env),
                 f"Wrong pre-action state at {i}")
        for s, clock in enumerate(row["clocks"]):
            run.need(all(math.isfinite(clock[k]) and clock[k] >= 0 for k in ("wall", "cpu")), "Invalid clock")
            if s != seat:
                run.need(clock["K"] >= 1 and clock["H"] >= 0 and clock["rolls"] >= 0
                         and (clock["single"] or clock["searched"] or clock["aborted"]), "Missing native search")
        bank = [b - max(c["wall"] - 1., 0.) for b, c in zip(bank, row["clocks"])]
        env.step(row["orders"])
        run.need(row["bank_after"] == bank and row["board_after"] == run.board_id(env)
                 and [p[0] for p in env.states[0].observation.players] == row["cash_after"], f"Wrong transition at {i}")
    run.need(run.terminal(env, seat) == expected, "Wrong terminal outcome")


def verify(path, plan):
    result = load(path / "summary.json")
    seed, seat = result["seed"], result["seat"]
    source_file = path / "source.json.gz"
    source = load(source_file)
    match_header(source, result)
    tickets = [e for e in load(plan)["schedule"] if (e["seed"], e["seat"]) == (seed, seat)]
    run.need(len(tickets) == 1 and tickets[0]["role"] == result["kind"], "Ticket role mismatch")
    run.need(result["terrain"] == load(plan)["physical_boards"][str(seed)] and
             result["protocol"] == run.sha(plan.read_bytes()) and
             result["source_sha256"] == run.sha(source_file.read_bytes()) and result["actors"] == run.SIGNED and
             result["runner_sha256"] == run.sha(Path(run.__file__).read_bytes()), "Unbound ticket or input")
    run.need(result["roots"] and [r["step"] for r in result["roots"]] == list(run.ROOT_STEPS), "Missing roots")
    replay(source, source["rows"], source["terminal"])
    run.need(source["terminal"] == result["source_terminal"] and
             source["inprocess_time_limits"] == run.local_time_limits(source["rows"]), "Source accounting")
    num_suffixes = 0
    for info in result["roots"]:
        start = info["step"]
        root_file = path / f"root_{start}.json.gz"
        root = load(root_file)
        match_header(root, result)
        run.need(info["sha256"] == run.sha(root_file.read_bytes()) and
                 root["source_sha256"] == result["source_sha256"] and root["root_step"] == start and
                 root["status"] == info["status"] and root["source_terminal"] == source["terminal"], "Unbound root")
        if start >= len(source["rows"]):
            run.need(root["status"] == "unreachable_source_terminal" and info["margin_delta"] is None,
                     "Unreachable root was scored")
            continue
        run.need(root["status"] == "complete" and sorted(root["arms"]) == ["bc", "elite"], "Missing branch")
        expected_order = tickets[0].get("order") or (
            ["elite", "bc"] if (seed - 540 + run.ROOT_STEPS.index(start)) % 2 else ["bc", "elite"])
        run.need(root["order"] == expected_order, "Wrong paired arm order")
        source_root = source["rows"][start]
        run.need(root["root_public_sha256"] == run.digest(source_root["public"])
                 and root["root_board_sha256"] == source_root["board_before"], "Root chronology")
        for name, arm in root["arms"].items():
            rows = arm["rows"]
            head = rows[0]
            enemy = 1-seat
            run.need(head["orders"][enemy] == source_root["orders"][enemy] and
                     head["clocks"][enemy] == source_root["clocks"][enemy] and
                     head["bc_order"] == source_root["orders"][seat], "Shared root mismatch")
            for i, row in enumerate(rows):
                run.need(row["common_enemy_root"] == (i == 0) and row["policy"] == (name if i < run.WINDOW else "bc"),
                         "Policy window error")
                if i < run.WINDOW:
                    run.need(row["orders"][seat] == row[f"{name}_order"] and
                             row["option_disagreement"] == (row["bc_order"] != row["elite_order"]), "Action attribution")
            run.need(arm["within_branch_disagreements"] == sum(bool(r.get("option_disagreement")) for r in rows)
                     and arm["window_decisions"] == min(run.WINDOW, len(rows)), "Dropped identical actions")
            full = source["rows"][:start] + rows
            replay(source, full, arm["terminal"])
            run.need(arm["inprocess_time_limits"] == run.local_time_limits(full), "Branch timing accounting")
            num_suffixes += 1
        a, b = (root["arms"][n]["terminal"] for n in ("bc", "elite"))
        for key, delta in (("win_delta", b["win"] - a["win"]), ("margin_delta", b["margin"] - a["margin"])):
            run.need(info[key] == root[key] == delta, "Outcome delta mismatch")
        run.need(root["intervention"] == run.compare_window(root["arms"], seat), "Causal chronology mismatch")
    return {"seed": seed, "seat": seat, "role": result["kind"], "verified_source_games": 1,
            "verified_paired_suffixes": num_suffixes, "source_sha256": result["source_sha256"],
            "original_requests_reexecuted": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    verdict = verify(args.out, args.protocol)
    if args.receipt is not None:
        run.save(args.receipt, verdict)
    print(json.dumps(verdict, ensure_ascii=False))
