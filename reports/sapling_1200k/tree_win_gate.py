"""Freeze one 100-map final exam and compute its map-clustered win-rate interval."""
import argparse
import hashlib
import gzip
import json
import math
import os
import shutil
import tempfile
import fcntl
import datetime
import subprocess
import sys
from pathlib import Path
import random
import secrets

ROOT = Path(__file__).resolve().parents[2]
GUARDED_V5 = "89df51540512d187ad64e82e21890e9f301083fc07ba06357eddaf43fc6692a2"
RULES = "code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"
N = 100
RESERVED_RANGE = (1_887_000_000, 1_888_000_000)
EXAMINER = "57f9a594cce4f2ec9e3573d8d9d26ca1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_hashes():
    paths = (
        "reports/sapling_1200k/tree_win_gate.py", "reports/sapling_1200k/tree_v4_exam.py",
        "reports/sapling_1200k/cash_v3_exam.py", "reports/sapling_1200k/external_3s2.py",
        "reports/sapling_1200k/tree_audited_game.py", "reports/sapling_1200k/tree_pack_arena.py",
        "reports/sapling_1200k/it50_v3_audited_seat.py", "reports/sapling_1200k/it50_v3_exam.py",
        "reports/sapling_bootstrap/transfer_3s2.py", "reports/sapling_bootstrap/smoke_ab.py",
        "code/eval_v53.py", "code/fastsim.py", "code/cnet2_pack.py", "code/cnet2_agent.py",
        "code/cnet2_feat.py", "code/cnet2_np.py", "data/master3s2_source.py",
        "code/init_sapling.py", "code/cnet2_model.py", "code/cnet2_search_pack.py", "code/cnet2_search.py",
        "code/cnet2_comparator.py", "reports/sapling_1200k/build_v5_guard.py",
        "reports/sapling_1200k/v5_guard_runtime.py", RULES,
    )
    return {p: digest(ROOT / p) for p in paths}


def sapling_shapes(meta):
    """Construct the declared model on the meta device for its exact tensor schema."""
    if str(ROOT / "code") not in sys.path:
        sys.path.insert(0, str(ROOT / "code"))
    import torch
    from cnet2_model import CNet2
    from init_sapling import SAPLING_PRESETS, resolve_preset_metadata
    cfg = meta.get("cfg", {})
    if cfg.get("NB") == 12:
        expected_cfg = dict(SAPLING_PRESETS["sapling-1.2m"]["cfg"], NB=12)
        if (cfg != expected_cfg or any(type(v) is not int for v in cfg.values())
                or meta.get("architecture_id") != "sapling-experimental-c93-nb12-deepening-v1"
                or meta.get("parent_preset") != "sapling-1.2m"
                or meta.get("params") != 2_143_916):
            raise ValueError("unknown deep Sapling architecture")
    else:
        _, cfg, _ = resolve_preset_metadata(meta)
    with torch.device("meta"):
        model = CNet2(**cfg)
    return {name: tuple(t.shape) for name, t in model.state_dict().items()}


def rebuild_guarded_v5(source, directory):
    # The frozen builder uses repo-relative input/output paths. Preserve that
    # layout in scratch so an exam cannot overwrite its own candidate.
    root = directory / "guard-build"
    actor = root / "data/weights/sapling_1200k"
    support = root / "reports/sapling_1200k"
    actor.mkdir(parents=True)
    support.mkdir(parents=True)
    (root / "code").symlink_to(ROOT / "code", target_is_directory=True)
    shutil.copyfile(source, actor / "soft_it50_v5.npz")
    for name in ("build_v5_guard.py", "v5_guard_runtime.py"):
        shutil.copyfile(Path(__file__).with_name(name), support / name)
    subprocess.run([sys.executable, str(support / "build_v5_guard.py")],
                   check=True, stdout=subprocess.DEVNULL)
    return actor / "soft_it50_v5_guard_agent.py"


def prove_single_pack(weights, pack):
    """Rebuild the exact one-actor candidate from the given Sapling NPZ."""
    if Path(weights).suffix.lower() != ".npz":
        raise ValueError("compound packs need a verified build and attribution adapter before formal use")
    with tempfile.TemporaryDirectory(prefix="sapling-freeze-") as tmp:
        source = Path(tmp) / "frozen.npz"
        output = Path(tmp) / "rebuilt.py"
        shutil.copyfile(weights, source)
        import numpy as np
        with np.load(source, allow_pickle=False) as current:
            shapes = sapling_shapes(json.loads(str(current["meta"])))
            if set(current.files) != set(shapes) | {"meta"}:
                raise ValueError("candidate tensor keys differ from its declared Sapling state")
            for key, shape in shapes.items():
                value = current[key]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid Sapling parameter {key}")
        if digest(pack) == GUARDED_V5:
            output = rebuild_guarded_v5(source, Path(tmp))
        else:
            subprocess.run([sys.executable, str(ROOT / "code/cnet2_pack.py"), str(source), str(output), "2"],
                           check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, CNET2_QMODE="int8"))
        if digest(output) != digest(pack):
            raise ValueError("candidate pack differs from the exact rebuild of its supplied source weights")
        return digest(source), digest(output)


def seed_list(nonce):
    generator = random.Random(int(nonce, 16))
    return sorted(generator.sample(range(*RESERVED_RANGE), N))


def validate(plan, weight_hash, pack_hash, examiner_hash, run):
    if (plan["schema"] != "sapling-one-shot-90/2" or plan["weight_sha256"] != weight_hash
            or plan["pack_sha256"] != pack_hash or plan["examiner_md5"] != examiner_hash
            or plan["protocol_sha256"] != protocol_hashes()
            or plan["run_directory"] != Path(run).name):
        raise ValueError("final plan does not match this run and exact frozen candidate")
    seeds = [row["seed"] for row in plan["maps"]]
    if len(plan["nonce"]) != 64 or seeds != seed_list(plan["nonce"]):
        raise ValueError("map allocation differs from the original draw")
    if set(seeds).intersection(plan["excluded_reserved_seeds"]):
        raise ValueError("exam overlaps previously exposed reserved maps")
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


def verify_terminal_replay(row, raw):
    # Replay recorded actions only. This never reruns the wall-clock opponent.
    from tree_audited_game import exam, prior
    if raw["game_configuration"] != exam.CFG or prior.hashed(raw["actions"]) != row["actions_sha256"]:
        raise ValueError("game configuration or action digest differs")
    sim = exam.FastCarbon(exam.CFG).reset(seed=row["seed"])
    for index, actions in enumerate(raw["actions"], 1):
        pre_cash = list(sim.cash)
        sim.step(*actions)
        state = sim.get_state()
        extinct = [bool(exam._extinct(state, p)) for p in (0, 1)]
        terminal = any(extinct) or index == exam.EPISODE_STEPS - 1
        if terminal != (index == len(raw["actions"])):
            raise ValueError("action history does not end on the first official terminal")
    if (prior.hashed(state) != row["final_state_sha256"] or raw["extinct_by_seat"] != extinct
            or row["extinct_by_seat"] != extinct or row["termination"] != (
                "extinction" if any(extinct) else "official_episode_limit")
            or raw["last_pre_cash"] != pre_cash or raw["last_post_cash"] != state["cash"]):
        raise ValueError("recorded end state differs from replayed actions")


def verify_raw(directory, row, pack_sha, weights_sha, formal=False):
    path = Path(directory) / f"raw_{row['seed']}_{row['candidate_seat']}.json.gz"
    if row["raw_file"] != path.name or digest(path) != row["raw_sha256"]:
        raise ValueError("raw transcript missing or changed")
    raw = json.loads(gzip.decompress(path.read_bytes()))
    if (raw["seed"], raw["candidate_seat"], raw["pack_sha256"], raw["examiner_md5"]) != (
            row["seed"], row["candidate_seat"], pack_sha, EXAMINER) or raw["weight_identity"] != weights_sha:
        raise ValueError("raw transcript identity mismatch")
    if formal and pack_sha == GUARDED_V5:
        stats = raw.get("candidate_guard_stats")
        if (not isinstance(stats, dict) or stats != row.get("candidate_guard_stats")
                or stats["calls"] != row["actual_transitions"]
                or not 0 <= stats["unchanged_slots"] <= stats["active_slots"]):
            raise ValueError("guarded candidate has no valid network-action attribution")
        witnesses = raw["unchanged_network_action_steps"]
        if (witnesses != row["unchanged_network_action_steps"] or witnesses != sorted(set(witnesses))
                or len(witnesses) > stats["calls"] - stats["changed_roots"]
                or any(type(t) is not int or not 0 <= t < len(raw["actions"])
                       or not any(raw["actions"][t][row["candidate_seat"]].values()) for t in witnesses)):
            raise ValueError("invalid unchanged non-idle network decisions")
    steps = row["actual_transitions"]
    if raw["errors"] or len(raw["actions"]) != steps or raw["steps"] != steps:
        raise ValueError("incomplete action history")
    if steps <= 0:
        raise ValueError("empty action history")
    verify_terminal_replay(row, raw)
    ext = raw["extinct_by_seat"]
    cash = raw["last_pre_cash"] if any(ext) else raw["last_post_cash"]
    scores = [steps - 301 if ext[seat] else float(cash[seat]) for seat in (0,1)]
    if scores != row["official_score_by_seat"] or scores != raw["official_score_by_seat"]:
        raise ValueError("score differs from the terminal cash and extinction rule")
    if raw["margin"] != row["margin"] or raw["result"] != row["result"]:
        raise ValueError("raw outcome differs from the game ledger")
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
    if plan.get("pack_sha256") == GUARDED_V5:
        if not any(row["unchanged_network_action_steps"] for row in games):
            raise ValueError("guarded candidate executed no unmodified network decisions")
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
            verify_raw(args.run, row, plan["pack_sha256"], plan["weight_sha256"], formal=True)
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
    weights, pack = prove_single_pack(args.weights, args.pack)
    while True:
        nonce = secrets.token_hex(32)
        seeds = seed_list(nonce)
        if not used.intersection(seeds):
            break
    plan = {"schema":"sapling-one-shot-90/2", "weight_sha256":weights, "pack_sha256":pack,
            "examiner_md5":EXAMINER, "protocol_sha256":protocol_hashes(), "exclusion_count":len(used),
            "exclusions_sha256":digest(args.exclusions), "nonce":nonce,
            "excluded_reserved_seeds":sorted(s for s in used if RESERVED_RANGE[0] <= s < RESERVED_RANGE[1]),
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
