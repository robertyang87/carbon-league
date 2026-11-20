"""Differentially verify historical fast-forward against the untouched original search."""
import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "policy_window_returns"
sys.path.insert(0, str(RUN / "runner"))
import run as common
from recover import NativePrefix, SOURCE

H = common.original_state.h
CASES = [(621758540, 1, 4), (621758541, 64, 67)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_tape(seed, stop):
    manifest = json.loads((RUN / "hosted/manifest.json").read_bytes())
    entry = next(x for x in manifest["tickets"] if x["seed"] == seed)
    archive = RUN / "hosted" / f'{entry["artifact"]}.zip'
    require(sha(archive.read_bytes()) == entry["sha256"] and entry["role"] == "fit", "Training archive drift")
    with zipfile.ZipFile(archive) as z:
        member = z.read("source.json.gz")
        source = json.loads(gzip.decompress(member))
    seat = 1 - source["seat"]
    env = common.collect.begin(seed, source["configuration"])
    require(sha(bytes(env.states[0].observation.full_carbon)) == source["terrain"], "Wrong training map")
    tape = []
    for i, row in enumerate(source["rows"][:stop+1]):
        obs = common.public(env, seat, row["bank_before"])
        require(common.digest(obs) == row["enemy_public_sha256"] and common.board_id(env) == row["board_before"],
                "Native public observation or board identity mismatch")
        tape.append((obs, row["orders"][seat], row["clocks"][seat]["cost"]))
        if i < stop:
            env.step(row["orders"])
    return tape, {k:v for k,v in source["configuration"].items() if k != "randomSeed"}, {
        "zip_sha256": entry["sha256"], "source_sha256": sha(member), "artifact": entry["artifact"],
        "terrain": source["terrain"], "seed": seed, "opponent_seat": seat}


def state(native):
    root = native._ST
    return {
        "tracker": H(root["tr"]), "last_dest": H(root["tr"].last_dest), "prev": H(root["prev"]),
        "root_rng": H(root["rng"]), "cost": list(native._S_COST),
        "shape_table": H(native._SHIFT_T), "configured_rules": H(native._RX_CFG),
        "ambient_rng": H(common.original_state.rng(native))}


def repair(native, view, issued, cost):
    native._ST["tr"].set_actions(view["players"], issued)
    native._S_COST[:] = cost


def warm_geometry(native, first, cfg):
    # _turn populates the exact lazy geometry and configuration memos for this
    # configuration object. Its temporary board does not enter _ST or _S_COST.
    view, _, _ = first
    est = list(native._ST["tr"].bel.estimate())
    root = dict(view, carbon=est)
    native._turn(root, cfg, [{}, {}])


def root_probe(player, item, cfg):
    native = player.mod
    view, _, _ = item
    old_search = native._search
    captured = {}

    def observe(*args):
        _, _, est, command, workers, bases, prev, _, home = args
        # Keep references only while on the original search deadline.
        captured.update(estimate=est, offered=command, worker_logits=workers,
                        base_logits=bases, previous=prev, returning=home)
        return old_search(*args)

    old_recorder = native._S_REC
    native._S_REC = []
    native._search = observe
    try:
        action, clock = common.call(player, view, cfg, examiner=True)
        search_record = native._S_REC[-1] if native._S_REC else None
    finally:
        native._search = old_search
        native._S_REC = old_recorder
    return {
        "input_digests": {key:H(value) for key,value in captured.items()},
        "natural_action": action, "natural_clock": clock,
        "candidate_signatures": [common.digest(c) for c in search_record[2]] if search_record else [],
    }


def run_case(seed, first, root, original):
    tape, cfg, identity = fixed_tape(seed, root)
    player = common.collect.Player(original)
    native = player.mod
    common.original_state.prepare(native)
    init = tape[0]
    initial_action, _ = common.call(player, init[0], cfg, examiner=True)
    repair(native, *init)
    warm_geometry(native, init, cfg)
    seat = identity["opponent_seat"]
    early = NativePrefix(native, seat)
    for item in tape[1:first]:
        early.advance(*item)
    start = common.original_state.take(native)
    static = common.original_state.static(native)
    observations = []
    exact_time = 0.
    for obs, issued, cost in tape[first:root]:
        before = time.perf_counter()
        action, _ = common.call(player, obs, cfg, examiner=True)
        repair(native, obs, issued, cost)
        exact_time += time.perf_counter() - before
        observations.append({"step":obs["step"], "state":state(native),
                             "shadow_matches_issued": action == issued,
                             "public_sha256":common.digest(obs)})
    exact_root = root_probe(player, tape[root], cfg)
    common.original_state.restore(native, start)
    fast = NativePrefix(native, seat, first)
    fast_time = 0.
    for saved, item in zip(observations, tape[first:root], strict=True):
        before = time.perf_counter()
        count = fast.advance(*item)
        fast_time += time.perf_counter() - before
        require(saved["state"] == state(native), f'Decision state differs at {seed}/{saved["step"]}')
        saved["weighted_policy_draw_count"] = count
    require(common.original_state.static(native) == static, "Frozen search backends changed")
    fast_root = root_probe(player, tape[root], cfg)
    require(exact_root["input_digests"] == fast_root["input_digests"] and
            len(exact_root["input_digests"]) == 6, "Next-step native policy inputs diverged")
    a, b = (r["candidate_signatures"] for r in (exact_root, fast_root))
    common_n = min(len(a), len(b))
    return {
        **identity, "first_validated_step":first, "tested_root":root,
        "step0_already_natural":True, "step0_shadow_matches_logged":initial_action == init[1],
        "shared_snapshot":start["expected"], "uncompared_fast_steps_before_test":first-1,
        "native_prefix_seconds":exact_time, "fast_prefix_seconds":fast_time,
        "verified_prefix": observations, "full_root":exact_root, "fast_root":fast_root,
        "both_recorded_candidate_count":common_n,
        "actual_candidate_lists_equal":a == b if common_n else None,
        "same_ordered_candidate_prefix":a[:common_n] == b[:common_n] if common_n else None,
        "natural_actions_equal":exact_root["natural_action"] == fast_root["natural_action"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = common.REPO / "data/master3s2_source.py"
    require(sha(original.read_bytes()) == SOURCE == sha(subprocess.check_output(
        ["git", "show", "HEAD:sources/master3s2/master3s2.py"], cwd=common.REPO)), "Original identity mismatch")
    require(os.environ.get("CARBON_DEBUG") == "1", "Original must surface historical path errors")
    result = [run_case(*case, original) for case in CASES]
    report = {
        "schema": "sapling/native-prefix-differential/1", "source_sha256":SOURCE,
        "native_sampler": {"configured_seed":None, "temperature":1., "deadline_seconds":.58},
        "source_tree_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=common.REPO,text=True).strip(),
        "run_py_sha256":sha(Path(__file__).read_bytes()), "recover_py_sha256":sha((HERE/"recover.py").read_bytes()),
        "numpy": common.original_state.np.__version__, "platform":platform.platform(),
        "affinity":sorted(os.sched_getaffinity(0)),
        "full_public_calls":sum(3 + root - first for _, first, root in CASES),
        "new_games":0, "cases":result}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")

if __name__ == "__main__":
    main()
