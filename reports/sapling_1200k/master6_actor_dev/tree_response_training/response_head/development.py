"""Run the registered two-map tree decision comparison only after the real training release passes."""

import argparse
from contextlib import ExitStack
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import types

import deploy
import evidence
import train

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
BASE = deploy.BASE
ROOT = deploy.ROOT
RUN = HERE / "development_runs"
ARMS = ("parent", "candidate")
REGISTRATION = "1a1981ade6d8746a74672f56bc93ee61e01860f4d94a48e2ad572e13b872bac9"
sys.path.insert(0, str(INTAKE))
import batch_clearance
from batch_cli import other_native
import source_games as source
from source_games import ENV, gate, sha, trial


def check(ok, message):
    if not ok:
        raise ValueError(message)


def check_registration():
    check(sha(HERE / "protocol.json") == REGISTRATION, "development decision registration changed")
    check(sha(INTAKE / "protocol.json") == evidence.FIRST_PROTOCOL, "source map registration changed")
    return source.PLAN["development_maps"]


def verify_maps(own_results):
    maps = check_registration()
    initial = trial.read(INTAKE / "clearance.json")
    check(
        initial["declaration_sha256"] == evidence.FIRST_PROTOCOL
        and initial["verdict"] == "eligible_in_checked_scope",
        "initial map clearance changed",
    )
    boards = {str(seed): batch_clearance.official_board(seed) for seed in maps}
    check(all(initial["boards"][key] == v for key, v in boards.items()), "reserved board changed")
    for key in boards:
        audit = initial["full_audits"][key]
        check(sha(INTAKE / audit["archive"]) == audit["sha256"], "original historical audit changed")

    known_seeds, known_boards, ladder = batch_clearance.ladder()
    # A newly delivered ladder ledger may predate its addition to the map_identity table.
    extra_ledgers = {}
    catalog = BASE / "all_top5"
    available = set(ladder["extra"])
    for path in sorted(catalog.rglob("validation_maps.jsonl.gz")):
        rel = str(path.relative_to(ROOT))
        if rel not in ladder["ledger_files_sha256"]:
            extra_ledgers[rel] = sha(path)
        for line in gzip.decompress(path.read_bytes()).splitlines():
            r = json.loads(line)
            available.add(r["id"])
            known_seeds.add(r["seed"])
            known_boards.add(r["initial_terrain_u8_sha256"])
    for man in [catalog / "manifest.json", *sorted(catalog.glob("appendix_*/manifest.json"))]:
        for game in trial.read(man)["games"]:
            check(game["id"] in available, f"registered ladder replay lacks an opening: {game['id']}")
            known_seeds.add(game["seed"])
    historical = batch_clearance.historical_catalogs(initial, boards)
    queue_path = INTAKE / "next_intake.json"
    queue = trial.read(queue_path)
    check(queue["protected_unopened_development_maps"] == maps, "changed protected reservation")
    known_seeds.update(source.PLAN["train_maps"])
    known_boards.update(initial["boards"][str(seed)] for seed in source.PLAN["train_maps"])
    for block in queue["batches"]:
        known_seeds.update(block["maps"])
        known_boards.update(block["terrain_at_queue_time"].values())
    local = {}
    for tape in sorted(BASE.rglob("raw_*.json.gz")):
        if tape.parent in own_results:
            continue
        raw = evidence.read(tape)
        terrain = (
            gate.map_identity.terrain_sha256(raw["initial_full_carbon"])
            if "initial_full_carbon" in raw
            else gate.map_identity.local_board(raw["seed"])
        )
        check(raw.get("initial_terrain_u8_sha256", terrain) == terrain, "raw opening digest mismatch")
        known_boards.add(terrain)
        known_seeds.add(raw["seed"])
        local[str(tape.relative_to(ROOT))] = sha(tape)
    check(
        not set(maps) & known_seeds
        and not set(boards.values()) & known_boards
        and len(set(boards.values())) == 2
        and all(not gate.RESERVED_RANGE[0] <= s < gate.RESERVED_RANGE[1] for s in maps),
        "reserved map overlaps ladder, training, native games or formal block",
    )
    return boards, {
        "historical": historical,
        "ladder": ladder,
        "extra_ladder_ledgers": extra_ledgers,
        "training_queue_sha256": sha(queue_path),
        "native_tapes": local,
    }


def schedule_locks(stack):
    paths = [
        INTAKE / ".continuation.lock",
        INTAKE / ".response_schedule.lock",
        INTAKE / "source" / ".source_schedule.lock",
    ]
    for area in sorted((INTAKE / "batches").glob("intake_*")):
        paths.extend([area / ".intake.schedule.lock", area / ".response_schedule.lock"])
        if (area / "source").is_dir():
            paths.append(area / "source" / ".source_schedule.lock")
    for path in paths:
        handle = stack.enter_context(path.open("a"))
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BlockingIOError(f"registered worker allocation busy: {path}") from None


def freeze_packages(temp):
    pack, weights, expected = trial.recipe("learned", temp)
    check(
        expected == sha(pack) == evidence.PARENT
        and sha(weights) == evidence.WEIGHTS
        and sha(ROOT / "data/weights/sapling_1200k/master6_actor_v1.npz") == evidence.STUDENT
        and sha(ROOT / "data/master3s2_source.py") == evidence.MASTER,
        "wrong base model or examiner",
    )
    candidate = RUN / "candidate_agent.py"
    new_sha = deploy.publish(candidate)
    check(new_sha == sha(candidate) and new_sha != expected, "candidate release identity mismatch")
    return {"parent": (pack, weights), "candidate": (candidate, HERE / "weights.json")}


def recorded_edits(pack, tape):
    """Replay the released policy on its own recorded public trajectory; verify the exact one-worker edit."""
    from fastsim import FastCarbon

    agent = types.ModuleType("tree_response_development_audit")
    exec(compile(pack.read_bytes(), str(pack), "exec"), agent.__dict__)
    original = agent._tree_original
    edits = []
    bases = []

    def remember(obs, cfg):
        base = original(obs, cfg)
        bases.append(base.copy())
        return base

    agent._tree_original = remember
    seat = tape["candidate_seat"]
    sim = FastCarbon(tape["game_configuration"]).reset(seed=tape["seed"])
    for step, joint in enumerate(tape["actions"]):
        action = agent.agent(sim.obs(seat), sim.config)
        check(action == joint[seat] and len(bases) == 1, f"released policy does not replay turn {step}")
        parent = bases.pop()
        altered = [uid for uid in parent.keys() | action.keys() if parent.get(uid) != action.get(uid)]
        event = agent._tree_head.last
        if altered:
            uid = event.get("worker")
            check(
                altered == [uid]
                and event["changed"]
                and not edits
                and event["old"] == parent.get(uid, "NONE")
                and event["new"] == action.get(uid, "NONE"),
                "release changed an unregistered worker or changed twice",
            )
            edits.append(dict(event))
        else:
            check(not event["changed"], "false policy change report")
        sim.step(*joint)
    return edits


def official_arm(arm, files, binding):
    directory = RUN / arm
    pack, weights = files
    pkg = binding["packages"][arm]
    busy = other_native(RUN)
    if busy:
        raise BlockingIOError(f"shared native workers busy: {busy}; rerun play to resume")
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(BASE / "tree_pack_arena.py"),
            "--pack",
            str(pack),
            "--weights",
            str(weights),
            "--pack-sha256",
            pkg["pack_sha256"],
            "--maps",
            *map(str, binding["maps"]),
            "--cpus",
            *map(str, binding["cpus"]),
            "--run",
            str(directory),
        ],
        check=True,
        env=ENV,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "official.json"
        subprocess.run(
            [sys.executable, str(BASE / "official_game_replay.py"), str(directory), "--out", str(path)],
            check=True,
            env=ENV,
        )
        trial.frozen_file(directory / "official_replay.json", trial.read(path))
    native = trial.read(directory / "results.json")
    official = trial.read(directory / "official_replay.json")
    header = native["header"]
    replay = official["runs"][arm]
    checks = {
        **pkg,
        "examiner_md5": source.PLAN["examiner_md5"],
        "cpus": binding["cpus"],
        "seeds": binding["maps"],
        "source_sha256": binding["arena_sources"],
        "plan_sha256": None,
    }
    check(
        native["status"] == "complete"
        and all(header[k] == v for k, v in checks.items())
        and official["engine_source_sha256"] == binding["official_engine"]
        and replay["pack_sha256"] == pkg["pack_sha256"],
        f"{arm} engine, source, allocation or policy mismatch",
    )
    expected = {(s, p) for s in binding["maps"] for p in (0, 1)}
    games = {(r["seed"], r["candidate_seat"]): r for r in native["games"]}
    rows = {(r["seed"], r["seat"]): r for r in replay["games"]}
    check(
        set(games) == set(rows) == expected and len(native["games"]) == len(replay["games"]) == 4,
        f"{arm} missing or duplicate physical game",
    )
    output = {}
    for key, r in sorted(games.items()):
        seed, seat = key
        raw = evidence.read(directory / r["raw_file"])
        gate.validate_examiner_configuration(raw["configuration"])
        gate.validate_game(r)
        account = gate.verify_raw(
            directory,
            r,
            pkg["pack_sha256"],
            pkg["weight_sha256"],
            income=True,
            expected_terrain=binding["boards"][str(seed)],
        )
        o = rows[key]
        scores = o["official_reward_by_seat"]
        margin = scores[seat] - scores[1 - seat]
        check(
            o["map_sha256"] == binding["boards"][str(seed)]
            and o["transcript_sha256"] == r["raw_sha256"]
            and o["statuses"] == ["DONE", "DONE"]
            and o["steps"] == r["actual_transitions"]
            and raw["audit_messages"] == []
            and o["result"] == r["result"]
            and r["official_score_by_seat"] == scores
            and math.isclose(margin, r["margin"])
            and r["cpu_affinity"] == [binding["cpus"][seat]],
            "terminal, source trace or processor differs",
        )
        output[key] = dict(
            o,
            margin=margin,
            U=trial.terminal(margin)[0],
            search_calls=r["search_return_types"],
            search_invocations=r["search_invocations"],
            call_seconds=r["call_seconds_by_seat"],
            wall_seconds=r["wall_seconds"],
            termination=r["termination"],
            cash_account=account,
        )
        if arm == "candidate":
            output[key]["verified_edits"] = recorded_edits(pack, raw)
    return output


def compare(maps, arms):
    keys = {(s, p) for s in maps for p in (0, 1)}
    check(
        len(maps) == len(set(maps)) == 2
        and set(arms) == set(ARMS)
        and all(set(arms[arm]) == keys for arm in ARMS),
        "requires two complete paired maps and both seats",
    )
    mean = statistics.mean
    pairs = []
    for seed, seat in sorted(keys):
        parent, candidate = (arms[arm][seed, seat] for arm in ARMS)
        pairs.append(
            dict(
                seed=seed,
                seat=seat,
                parent=parent,
                candidate=candidate,
                delta_margin=candidate["margin"] - parent["margin"],
                delta_U=candidate["U"] - parent["U"],
            )
        )
    per_map = {str(s): mean(r["delta_margin"] for r in pairs if r["seed"] == s) for s in maps}
    average = mean(per_map.values())
    wins = {arm: sum(r[arm]["result"] == "win" for r in pairs) for arm in ARMS}
    tests = {
        "both_map_margin_gains_positive": all(v > 0 for v in per_map.values()),
        "equal_map_mean_at_least_100": average >= 100,
        "positive_U_in_at_least_three_pairs": sum(r["delta_U"] > 0 for r in pairs) >= 3,
        "no_fewer_wins": wins["candidate"] >= wins["parent"],
    }
    se = statistics.stdev(per_map.values()) / math.sqrt(2)
    changed = sum(bool(r["candidate"]["verified_edits"]) for r in pairs)
    return dict(
        games=8,
        pairs=pairs,
        per_map_margin_gain=per_map,
        equal_map_margin_gain=average,
        illustrative_map_t95=[average - 12.706204736 * se, average + 12.706204736 * se],
        independent_maps=2,
        candidate_games_with_verified_edit=changed,
        candidate_games_without_edit=len(pairs) - changed,
        wins=wins,
        criteria=tests,
        verdict="development_pass" if all(tests.values()) else "retain_v1",
        formal=False,
    )


def play():
    maps = check_registration()
    try:
        train.rebuild_release(intake_root=INTAKE)
    except ValueError as exc:
        if str(exc) not in (
            "no eligible release: insufficient_realized_support",
            "no eligible release: heldout_gate_failed",
        ):
            raise
        return {
            "stage": str(exc).split(": ", 1)[1],
            "started_this_call": 0,
            "opened_development_maps_this_call": 0,
            "formal": False,
        }

    with ExitStack() as stack, tempfile.TemporaryDirectory() as tmp:
        schedule_locks(stack)
        cpus = source.PLAN["cpu_affinity"]
        check(set(cpus) <= os.sched_getaffinity(0), "registered CPU cores unavailable")
        own = {RUN / arm for arm in ARMS}
        seal = RUN / "release.json"
        check(seal.exists() or not any(p.exists() for p in own), "unsealed development outputs already exist")
        RUN.mkdir(exist_ok=True)
        files = freeze_packages(Path(tmp))
        boards, scanned = verify_maps(own)
        engine = trial.read(INTAKE / "source/official_replay.json")["engine_source_sha256"]
        engine_dir = ROOT / "code/env_src/carbon_challenge-master"
        check(
            all(sha(engine_dir / path) == value for path, value in engine.items()),
            "official interpreter changed",
        )
        bound = {
            "development_protocol": REGISTRATION,
            "source_protocol": evidence.FIRST_PROTOCOL,
            "master3s2_sha256": evidence.MASTER,
            "student_npz_sha256": evidence.STUDENT,
            "maps": maps,
            "boards": boards,
            "cpus": cpus,
            "arena_sources": gate.protocol_hashes(),
            "official_engine": engine,
            "scripts": {
                name: sha(HERE / name)
                for name in (
                    "development.py",
                    "policy.py",
                    "deploy.py",
                    "train.py",
                    "evidence.py",
                    "test_deploy.py",
                )
            },
            "official_replayer_sha256": sha(BASE / "official_game_replay.py"),
            "release_evidence": {
                name: sha(HERE / name)
                for name in ("weights.json", "evaluation.json", "status.json", "samples.json.gz")
            },
            "packages": {
                a: {"pack_sha256": sha(pack), "weight_sha256": sha(w)} for a, (pack, w) in files.items()
            },
        }
        trial.frozen_file(seal, bound)
        # Persist every clearance check. A change to an executable or its registered arena identity blocks resume.
        scan_id = hashlib.sha256(train.json_bytes(scanned)).hexdigest()
        trial.frozen_file(RUN / f"clearance_{scan_id}.json", scanned)
        arms = {arm: official_arm(arm, files[arm], bound) for arm in ARMS}
        result = dict(compare(maps, arms), release_sha256=sha(seal))
        trial.frozen_file(RUN / "comparison.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["play"])
    parser.parse_args()
    print(json.dumps(play(), ensure_ascii=False, allow_nan=False), flush=True)
