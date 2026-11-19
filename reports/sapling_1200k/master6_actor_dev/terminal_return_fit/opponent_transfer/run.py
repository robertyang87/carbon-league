"""One private-information-safe external development game per fresh process; verify with official replay."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
FIT = HERE.parent
BASE = FIT.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
import map_identity
from official_game_replay import ENGINE, check, make
from root_recovery_branches.branches import public, utility
from tree_audited_game import exam
from tree_win_gate import RESERVED_RANGE
import numpy as np

PLAN = HERE / "protocol.json"
PLAN_SHA = "6b91c9d0f971bb40b05a8f978bb385aa83487be0e3bd0a0d2bfede721f88df6a"
ENGINE_FILES = ["zerosum_env/core.py", "zerosum_env/envs/carbon/carbon.py",
                "zerosum_env/envs/carbon/helpers.py", "zerosum_env/envs/carbon/idgen.py",
                "zerosum_env/envs/carbon/carbon.json"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def write(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=None if path.suffix == ".gz" else 2,
                       allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing unequal artifact: {path}")
        return
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def plan():
    if sha(PLAN) != PLAN_SHA:
        raise ValueError("plan identity mismatch")
    return read(PLAN)


def module(spec, name):
    path, expected = spec
    if sha(ROOT / path) != expected:
        raise ValueError(f"source identity mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clearance():
    d = plan()
    witness, seeds, terrains = map_identity.ladder_boards()
    inputs = {}
    for p in sorted(FIT.glob("**/map_audit*.json.gz")) + sorted((FIT / "replication").glob("map_clearance*.json.gz")):
        record = read(p)
        for row in record["local_native_or_training_maps"]:
            seeds.add(row["seed"])
            terrains.add(row["terrain"])
        seeds.add(record["candidate_seed"])
        terrains.add(record["candidate_initial_terrain_u8_sha256"])
        inputs[str(p.relative_to(ROOT))] = sha(p)
    train = read(FIT.parent / "data_split.json")["games"]
    for row in train:
        seeds.add(row["seed"])
        terrains.add(row["terrain_sha256"])
    mappings = map_identity.checked_maps(d["maps"], seeds, terrains)
    for row in mappings:
        s = row["seed"]
        if RESERVED_RANGE[0] <= s < RESERVED_RANGE[1]:
            raise ValueError("reserved seed")
        np.random.seed(s)
        e = make("carbon", configuration={**exam.CFG, "randomSeed": s})
        if map_identity.terrain_sha256(e.states[0].observation.full_carbon) != row["initial_terrain_u8_sha256"]:
            raise ValueError("independent engine opening mismatch")
    if any(n not in d["maps"] for n in map(int, [r["seed"] for r in mappings])):
        raise ValueError("map list mismatch")
    names = sorted({r["opponent"] for r in train if r["split"] == "train"})
    if len(names) != 51 or sum(r["split"] == "train" for r in train) != 218:
        raise ValueError("changed cohort")
    manifest_paths = [BASE / f for f in ["cash_dagger_data.json", "tree_dagger_data.json", "elite_bc_data.json",
                                         "top5_h2h_data.json", "soft_it50_data_manifest.json"]]
    for p in manifest_paths:
        if b"policy_v0_8_replace" in p.read_bytes() or d["opponent"][1].encode() in p.read_bytes():
            raise ValueError(f"opponent already in accessible manifest: {p}")
    write(HERE / "clearance.json", {
        "protocol_sha256": PLAN_SHA, "boards": mappings, "ladder": witness,
        "checked_unique_seeds": len(seeds), "checked_unique_terrain_digests": len(terrains),
        "local_registries": inputs, "cohort_manifest_sha256": sha(FIT.parent / "data_split.json"),
        "cohort_218_opponent_names": names, "historical_manifest_sha256": {p.name: sha(p) for p in manifest_paths},
        "scope": "Accessible training and development map ledgers. Undisclosed outside schedules and alias identities are unknown.",
        "engine_sha256": {p: sha(ENGINE / p) for p in ENGINE_FILES}})


def normalize(action, obs, side):
    if not isinstance(action, dict):
        raise ValueError("action is not a dictionary")
    _, bases, workers, _ = obs["players"][side]
    for key, cmd in action.items():
        if key in workers:
            allowed = {None, "UP", "DOWN", "LEFT", "RIGHT"}
        elif key in bases:
            allowed = {None, "RECCOLLECTOR", "RECPLANTER"}
        else:
            raise ValueError(f"unexpected object {key}")
        if cmd not in allowed:
            raise ValueError(f"unexpected command on {key}: {cmd}")
    return {k: v for k, v in action.items() if v is not None}


def play(arm, seed, seat):
    d = plan()
    if arm not in d["arms"] or seed not in d["maps"] or seat not in d["candidate_seats"]:
        raise ValueError("unregistered game")
    c = read(HERE / "clearance.json")
    if c["protocol_sha256"] != PLAN_SHA or any(sha(ENGINE / p) != v for p, v in c["engine_sha256"].items()):
        raise ValueError("stale clearance/engine")
    out = HERE / arm
    out.mkdir(exist_ok=True)
    receipt = out / f"{seed}_{seat}.json"
    tape = out / f"{seed}_{seat}.json.gz"
    failure = out / f"{seed}_{seat}.failure.json"
    if failure.exists():
        raise ValueError("recorded error; adjudicate without silent retry")
    if receipt.exists():
        old = read(receipt)
        if (old["protocol_sha256"] != PLAN_SHA or old["pack_sha256"] != d["arms"][arm][1]
                or old["opponent_sha256"] != d["opponent"][1]):
            raise ValueError("unexpected receipt")
        check(out, old)
        return
    candidate = module(d["arms"][arm], "sapling_candidate")
    opponent = module(d["opponent"], "transfer_opponent")
    random.seed(seed)
    np.random.seed(seed)
    cfg = dict(exam.CFG)
    env = make("carbon", configuration={**cfg, "randomSeed": seed})
    board = list(env.states[0].observation.full_carbon)
    terrain = map_identity.terrain_sha256(board)
    if terrain != next(r["initial_terrain_u8_sha256"] for r in c["boards"] if r["seed"] == seed):
        raise ValueError("map identity")
    modules = [None, None]
    modules[seat], modules[1 - seat] = candidate, opponent
    actions, nonempty, cmds = [], [0, 0], [Counter(), Counter()]
    try:
        while not env.done:
            turn = len(actions)
            if turn >= cfg["episodeSteps"] - 1:
                raise ValueError("interpreter advanced beyond horizon")
            joint = []
            for s in range(2):
                obs = copy.deepcopy(public(env, s))
                if obs["step"] != turn or obs["player"] != s or "full_carbon" in obs:
                    raise ValueError("observation contract")
                act = normalize(modules[s].agent(obs, dict(cfg)), obs, s)
                nonempty[s] += bool(act)
                cmds[s].update(act.values())
                joint.append(act)
            actions.append(joint)
            env.step(joint)
        if min(nonempty) == 0 or opponent._policy.last_step != len(actions) - 1:
            raise ValueError("inactive policy or incomplete real calls")
        if [s.status for s in env.states] != ["DONE", "DONE"]:
            raise ValueError("not a completed game")
    except Exception as e:
        write(failure, {"protocol_sha256": PLAN_SHA, "seed": seed, "seat": seat, "arm": arm,
                        "error": repr(e), "turn": len(actions), "actions": actions})
        raise
    scores = [s.reward for s in env.states]
    margin = scores[seat] - scores[1 - seat]
    result = "win" if margin > 0 else "loss" if margin < 0 else "draw"
    raw = {"seed": seed, "candidate_seat": seat, "initial_full_carbon": board,
           "initial_terrain_u8_sha256": terrain, "game_configuration": cfg, "actions": actions,
           "steps": len(actions), "protocol_sha256": PLAN_SHA, "pack_sha256": d["arms"][arm][1],
           "opponent_sha256": d["opponent"][1], "official_score_by_seat": scores}
    write(tape, raw)
    row = {k: raw[k] for k in ("seed", "candidate_seat", "initial_terrain_u8_sha256", "protocol_sha256",
                               "pack_sha256", "opponent_sha256", "official_score_by_seat")}
    row.update({"raw_file": tape.name, "raw_sha256": sha(tape), "actual_transitions": len(actions),
                "result": result, "margin": margin, "U": utility(scores, seat),
                "public_agent_calls_by_seat": [len(actions)] * 2, "nonempty_steps_by_seat": nonempty,
                "commands_by_seat": [dict(c) for c in cmds], "actor_slot_counters": candidate._COUNTS,
                "opponent_last_observed_step": opponent._policy.last_step})
    verified = check(out, row)
    if verified["map_sha256"] != terrain:
        raise ValueError("fresh official replay disagrees on initial terrain")
    row["official_replay"] = verified
    write(receipt, row)
    print(arm, seed, seat, result, round(margin, 2), len(actions), flush=True)


def render():
    d = plan()
    by_arm = {}
    for arm in d["arms"]:
        rows = [read(HERE / arm / f"{seed}_{seat}.json") for seed in d["maps"] for seat in d["candidate_seats"]]
        for r in rows:
            if r["protocol_sha256"] != PLAN_SHA or check(HERE / arm, r) != r["official_replay"]:
                raise ValueError("replay mismatch")
        by_arm[arm] = rows
    maps = []
    for seed in d["maps"]:
        maprows = {}
        for arm, rows in by_arm.items():
            maprows[arm] = [r for r in rows if r["seed"] == seed]
        pairs = []
        for p, l in zip(maprows["parent"], maprows["learned"]):
            if (p["candidate_seat"], p["initial_terrain_u8_sha256"]) != (l["candidate_seat"], l["initial_terrain_u8_sha256"]):
                raise ValueError("pair mismatch")
            # Earliest divergent candidate action is directly comparable while both arms are on the same trajectory.
            ap = read(HERE / "parent" / p["raw_file"])["actions"]
            al = read(HERE / "learned" / l["raw_file"])["actions"]
            first = next((t for t, (x, y) in enumerate(zip(ap, al)) if x != y), None)
            if first is not None and ap[first][1 - p["candidate_seat"]] != al[first][1 - p["candidate_seat"]]:
                raise ValueError("opponent differed on a shared public prefix")
            pairs.append({"seat": p["candidate_seat"], "delta_margin": l["margin"] - p["margin"],
                          "delta_U": l["U"] - p["U"], "first_joint_divergence_turn": first})
        maps.append({"seed": seed, "pairs": pairs, "mean_delta_margin": sum(r["delta_margin"] for r in pairs) / 2,
                     "mean_delta_U": sum(r["delta_U"] for r in pairs) / 2})
    arms = {}
    for arm, rows in by_arm.items():
        arms[arm] = {"WDL": [sum(r["result"] == outcome for r in rows) for outcome in ("win","draw","loss")],
                     "mean_margin": sum(r["margin"] for r in rows) / 4, "mean_U": sum(r["U"] for r in rows) / 4}
    write(HERE / "results.json", {"protocol_sha256": PLAN_SHA, "runner_sha256": sha(Path(__file__)), "arms": arms,
                                  "map_clusters": maps, "games": by_arm})
    print(arms, maps)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("clearance")
    game = sub.add_parser("play")
    game.add_argument("arm", choices=("parent", "learned"))
    game.add_argument("seed", type=int)
    game.add_argument("seat", type=int, choices=(0, 1))
    sub.add_parser("report")
    a = p.parse_args()
    if a.command == "clearance":
        clearance()
    elif a.command == "play":
        play(a.arm, a.seed, a.seat)
    else:
        render()
