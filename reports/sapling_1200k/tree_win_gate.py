"""Freeze one 100-map final exam and compute its map-clustered win-rate interval."""
import argparse
import hashlib
import gzip
import json
import math
import fcntl
import datetime
import subprocess
import sys
from pathlib import Path
import random
import secrets

N = 100
RESERVED_RANGE = (1_887_000_000, 1_888_000_000)
EXAMINER = "57f9a594cce4f2ec9e3573d8d9d26ca1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = (
        "reports/sapling_1200k/tree_win_gate.py", "reports/sapling_1200k/tree_v4_exam.py",
        "reports/sapling_1200k/cash_v3_exam.py", "reports/sapling_1200k/external_3s2.py",
        "reports/sapling_1200k/tree_audited_game.py", "reports/sapling_1200k/tree_pack_arena.py",
        "reports/sapling_1200k/it50_v3_audited_seat.py", "reports/sapling_1200k/it50_v3_exam.py",
        "reports/sapling_bootstrap/transfer_3s2.py", "reports/sapling_bootstrap/smoke_ab.py",
        "code/eval_v53.py", "code/fastsim.py", "code/cnet2_pack.py", "code/cnet2_agent.py",
        "code/cnet2_feat.py", "code/cnet2_np.py", "data/master3s2_source.py",
    )
    return {p: digest(root / p) for p in paths}


def seed_list(nonce):
    generator = random.Random(int(nonce, 16))
    return sorted(generator.sample(range(*RESERVED_RANGE), N))


def validate(plan, weight_hash, pack_hash, examiner_hash, run):
    if (plan["schema"] != "sapling-one-shot-90/1" or plan["weight_sha256"] != weight_hash
            or plan["pack_sha256"] != pack_hash or plan["examiner_md5"] != examiner_hash
            or plan["protocol_sha256"] != protocol_hashes()
            or plan["run_directory"] != Path(run).name):
        raise ValueError("final plan does not match this run and exact frozen candidate")
    seeds = [row["seed"] for row in plan["maps"]]
    if len(plan["nonce"]) != 64 or seeds != seed_list(plan["nonce"]):
        raise ValueError("map allocation differs from the original draw")
    return seeds


def cp_interval(n, k, error=0.025):
    def tail(k, p):
        return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))

    def lower(successes):
        if successes == 0:
            return 0.
        lo, hi = 0., 1.
        for _ in range(65):
            midpoint = (lo + hi) / 2
            if tail(successes, midpoint) < error/2:
                lo = midpoint
            else:
                hi = midpoint
        return (lo + hi) / 2

    return lower(k), 1 - lower(n-k)


def verify_raw(directory, row, pack_sha):
    path = Path(directory) / f"raw_{row['seed']}_{row['candidate_seat']}.json.gz"
    if row["raw_file"] != path.name or digest(path) != row["raw_sha256"]:
        raise ValueError("raw transcript missing or changed")
    raw = json.loads(gzip.decompress(path.read_bytes()))
    if (raw["seed"], raw["candidate_seat"], raw["pack_sha256"], raw["examiner_md5"]) != (
            row["seed"], row["candidate_seat"], pack_sha, EXAMINER):
        raise ValueError("raw transcript identity mismatch")
    steps = row["actual_transitions"]
    if raw["errors"] or len(raw["actions"]) != steps or raw["steps"] != steps:
        raise ValueError("incomplete action history")
    ext = raw["extinct_by_seat"]
    cash = raw["last_pre_cash"] if any(ext) else raw["last_post_cash"]
    scores = [steps - 301 if ext[seat] else float(cash[seat]) for seat in (0,1)]
    if scores != row["official_score_by_seat"] or scores != raw["official_score_by_seat"]:
        raise ValueError("score differs from the terminal cash and extinction rule")
    events = raw["search_calls"]
    kinds = ("searched", "single", "depth_zero_return_base")
    if [e[0] for e in events] != list(range(steps)) or any(e[1] not in kinds for e in events):
        raise ValueError("missing or invalid original search call")
    if any(sum(e[1] == k for e in events) != row["search_return_types"].get(k,0) for k in kinds):
        raise ValueError("search trace differs from per-game ledger")


def validate_game(row):
    seat = row["candidate_seat"]
    scores = row["official_score_by_seat"]
    if seat not in (0, 1) or len(scores) != 2 or not all(math.isfinite(v) for v in scores):
        raise ValueError("invalid official scores")
    margin = scores[seat] - scores[1-seat]
    result = "win" if margin > 0 else "loss" if margin < 0 else "draw"
    if row["result"] != result or not math.isclose(margin, row["margin"], rel_tol=0., abs_tol=1e-6):
        raise ValueError("reported outcome differs from the official score")
    steps = row["actual_transitions"]
    if (row["agent_errors_by_seat"] != [0, 0] or row["agent_calls_by_seat"] != [steps, steps]
            or row["termination"] not in ("extinction", "official_episode_limit")
            or not 0 < steps <= 299 or (row["termination"] == "official_episode_limit" and steps != 299)):
        raise ValueError("missing or invalid official terminal")
    search = row["search"]
    kinds = row["search_return_types"]
    searched, single, no_depth = (kinds.get(k, 0) for k in ("searched", "single", "depth_zero_return_base"))
    if (row["audit_messages"] or not row["valid"] or row["search_invocations"] != steps
            or searched != search["searched"] or single != search["single"] or searched <= 0
            or min(searched, single, no_depth) < 0 or no_depth > search["aborted"]
            or searched + single + no_depth != steps):
        raise ValueError("original examiner search coverage not confirmed")


def verdict(plan, games):
    expected = {(r["seed"], seat) for r in plan["maps"] for seat in (0, 1)}
    if len(expected) != 2*N or len(games) != len(expected):
        raise ValueError("incomplete frozen map set")
    rows = {(g["seed"], g["candidate_seat"]):g for g in games}
    if set(rows) != expected:
        raise ValueError("duplicate, unknown or missing map and seat")
    for row in games:
        validate_game(row)
    seat_wins = [sum(g["candidate_seat"] == seat and g["result"] == "win" for g in games)
                 for seat in (0, 1)]
    counts = [sum(rows[(r["seed"], s)]["result"] == "win" for s in (0, 1))
              for r in plan["maps"]]
    any_win = sum(k >= 1 for k in counts)
    both_win = sum(k == 2 for k in counts)
    left = cp_interval(N, any_win)
    right = cp_interval(N, both_win)
    ci = [(x+y)/2 for x,y in zip(left,right)]
    total = sum(counts)
    seat_intervals = [list(cp_interval(N, k)) for k in seat_wins]
    return {"games": 2*N, "independent_map_units": N, "wins": total, "draws":
            sum(g["result"] == "draw" for g in games), "seat_wins": seat_wins,
            "observed_win_rate": total / (2*N), "observed_90_percent_met": 10*total >= 9*(2*N),
            "each_seat_observed_90_percent_met": all(k >= .90 * N for k in seat_wins),
            "seat_win_rates_simultaneous_95pct_intervals": seat_intervals,
            "maps_with_at_least_one_win": any_win, "maps_with_two_wins": both_win,
            "win_rate_95pct_map_interval": ci, "population_lower_bound_at_least_90_percent": ci[0] >= .90}


def execute():
    parser = argparse.ArgumentParser(description="Run or resume the frozen competition and preserve failed attempts")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs=2, default=[0, 1])
    args = parser.parse_args(sys.argv[2:])
    plan = json.loads(args.plan.read_text())
    validate(plan, digest(args.weights), digest(args.pack), EXAMINER, args.run)
    args.run.mkdir(parents=True, exist_ok=True)
    failed = args.run / "invalid_attempt.json"
    with (args.run / ".official.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if failed.exists():
            parser.error("this final attempt has a recorded failure; retain it and retire these maps")
        def stamp():
            return datetime.datetime.now(datetime.timezone.utc).isoformat()
        command = [sys.executable, "-u", str(Path(__file__).with_name("tree_pack_arena.py")), "--pack-sha256", plan["pack_sha256"],
                   "--final-plan", str(args.plan), "--weights", str(args.weights),
                   "--pack", str(args.pack), "--run", str(args.run), "--cpus", *map(str, args.cpus)]
        with (args.run / "execution.log").open("a", buffering=1) as log:
            log.write(f"{stamp()} starting or resuming plan {digest(args.plan)}\n")
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            log.write(f"{stamp()} runner exit {status}\n")
        if status < 0:
            raise SystemExit("interrupted by signal; resume with the same command")
        if status != 0:
            with failed.open("x") as file:
                json.dump(dict(utc=stamp(), plan_sha256=digest(args.plan), returncode=status), file)
            raise SystemExit("formal run invalid: keep execution.log; do not reroll a failed game")
        report = json.loads((args.run / "results.json").read_text())
        for row in report["games"]:
            verify_raw(args.run, row, plan["pack_sha256"])
        if (report["status"] != "complete" or report["header"]["plan_sha256"] != digest(args.plan)
                or report["verdict"] != verdict(plan, report["games"])):
            raise ValueError("reported final verdict does not match the full game ledger")
        print(json.dumps(report["verdict"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="new, unused run-directory name")
    parser.add_argument("--out", type=Path, required=True, help="new plan file")
    parser.add_argument("--exclusions", type=Path, required=True,
                        help="JSON array of prior training, validation, diagnostic and exam seeds")
    args = parser.parse_args()
    used = json.loads(args.exclusions.read_text())
    if not isinstance(used, list) or any(type(s) is not int for s in used):
        parser.error("exclusions must be a JSON array of integer seeds")
    # The reserved range belongs only to final tests. A seed-list collision retires
    # this independent random draw before any outcomes are generated.
    used = set(used)
    if not {620106811, 620106812}.issubset(used):
        parser.error("exclusion ledger is missing known V4 development maps")
    if args.out.exists() or args.run.exists():
        parser.error("allocate a fresh plan path and run directory")
    weights, pack = digest(args.weights), digest(args.pack)
    while True:
        nonce = secrets.token_hex(32)
        seeds = seed_list(nonce)
        if not used.intersection(seeds):
            break
    plan = {"schema":"sapling-one-shot-90/1", "weight_sha256":weights, "pack_sha256":pack,
            "examiner_md5":EXAMINER, "protocol_sha256":protocol_hashes(), "exclusion_count":len(used),
            "exclusions_sha256":digest(args.exclusions), "nonce":nonce,
            "run_directory":args.run.name, "maps":[{"seed": s} for s in seeds]}
    validate(plan, weights, pack, EXAMINER, args.run)
    with args.out.open("x") as out:
        json.dump(plan, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        execute()
    else:
        main()
