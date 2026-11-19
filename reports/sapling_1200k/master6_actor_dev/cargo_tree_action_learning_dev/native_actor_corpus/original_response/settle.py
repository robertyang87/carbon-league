"""Offline verification of the two archived native branches. No opponent is executed."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())
CARGO = REPO / "reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev"
WIDTH = 32


def historical_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q = historical_module("trunk_accounting", CARGO / "territory_sequence/response/run.py")
K = historical_module("trunk_native_api", CARGO / "learned_carbon_observation/native_response/run.py")
R = K.R


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha_bytes(K.compact(value))


def clock_proof(clock, start, end):
    calls = clock["calls"]
    assert [c["step"] for c in calls] == list(range(start, end))
    for c in calls:
        assert c["kind"] in {"single", "searched", "depth_zero_return_base"}
        assert (c["kind"] == "searched") == (c["max_depth"] > 0)
        assert (c["kind"] == "single") == (c["candidates"] == 1)
        assert min(c["process_cpu_seconds"], c["public_wall_seconds"]) >= 0
    assert clock["summary"] == (R.native_clock.summarize(calls, .58, 8) if calls else None)
    wall = [c["public_wall_seconds"] for c in calls]
    return dict(summary=clock["summary"], calls=len(calls), wall_total=sum(wall), wall_max=max(wall, default=None),
                zero_depth_steps=[c["step"] for c in calls if c["kind"] == "depth_zero_return_base"])


def changed_units(a, b):
    return {u: [a.get(u, "NONE"), b.get(u, "NONE")] for u in sorted(a.keys() | b.keys())
            if a.get(u, "NONE") != b.get(u, "NONE")}


def shared_bank(case, folder):
    s, t0 = case["seat"], case["t0"]
    path = REPO / case["source"]
    assert R.sha(path) == case["source_sha256"]
    raw = R.load(path)
    assert raw["seed"] == case["seed"] and raw["candidate_seat"] == s and not raw["errors"]
    terrain = sha_bytes(bytes(map(int, raw["initial_full_carbon"])))
    assert terrain == raw["initial_terrain_u8_sha256"] == case["terrain"]
    shared = folder / "shared.json.gz"
    bank = R.load(shared)
    assert bank["case"] == case
    archive_receipt = REPO / case["receipt"]
    assert R.sha(archive_receipt) == case["receipt_sha256"]
    prefix = bank["prefix"]
    assert prefix == raw["actions"][:t0] and len(prefix) == t0 and digest(prefix) == bank["prefix_sha"]
    assert len(bank["trace"]) == t0 + 1 and bank["root"]["t"] == t0
    snap = gzip.decompress((folder / "original_root.pickle.gz").read_bytes())
    assert sha_bytes(snap) == bank["snapshot_sha"] and base64.b64decode(bank["snapshot_b64"], validate=True) == snap
    proof = clock_proof(bank["clock"], 0, t0 + 1)
    original_diff = []
    for t, row in enumerate(bank["trace"]):
        expected = R.live.clean(prefix[t][1-s]) if t < t0 else bank["root"]["enemy"]
        assert row["t"] == t and row["original_executed"] == expected
        if t < t0 and row["original_proposed"] != expected:
            original_diff.append(t)
    root = bank["root"]
    assert bank["trace"][-1]["original_proposed"] == root["enemy"]
    historical = R.live.clean(raw["actions"][t0][1-s])
    proof.update(archived_root_enemy=historical, live_root_enemy=root["enemy"], live_root_matches_archive=historical == root["enemy"],
                 prefix_original_proposal_mismatch_steps=original_diff, root_call=bank["clock"]["calls"][-1])
    return raw, bank, sha_bytes(gzip.decompress(shared.read_bytes())), proof


def replay(case, folder, raw, bank, shared_sha, arm, packages):
    s, t0 = case["seat"], case["t0"]
    path = folder / (arm + ".json.gz")
    receipt = R.load(path)
    assert receipt["arm"] == arm and receipt["identity"] == bank["identity"] and receipt["shared_sha"] == shared_sha
    for key in ("snapshot_sha", "state_sha"):
        assert receipt[key] == bank[key]
    tape = receipt["tape"]
    assert tape and digest(tape) == receipt["tape_sha"]
    controlled = min(len(tape), WIDTH)
    assert tape[0][1-s] == bank["root"]["enemy"]
    calls = clock_proof(receipt["clock"], t0 + 1, t0 + len(tape))
    assert (receipt["seed"], receipt["seat"], receipt["root_t"], receipt["root_joint"]) == (case["seed"], s, t0, tape[0])
    assert len(receipt["suffix_observations"]) == len(tape)
    for host in ("host_at_start", "host_at_end"):
        R.native_clock.validate_host(receipt[host], [bank["cpu"]])
        for key in ("hostname", "cpu_model", "kernel", "affinity", "cpu_max"):
            assert receipt[host][key] == bank["host"][key]
    env = K.checked_env(raw)
    models = {k: R.live.module(p, "settle_trunk_" + arm + k) for k, p in packages.items()}
    rows = receipt["comparisons"]
    assert [r["t"] for r in rows] == list(range(t0, t0 + controlled))
    official_trajectory, account_rows = [], []
    for t, joint in enumerate(bank["prefix"] + tape):
        assert not env.done
        own_obs = K.public(env, s)
        assert own_obs["step"] == t
        if t >= t0:
            assert receipt["suffix_observations"][t-t0] == [digest(own_obs), digest(K.public(env, 1-s))]
        if t <= t0:
            src = bank["trace"][t]
            assert src["own_sha"] == digest(own_obs) and src["enemy_sha"] == digest(K.public(env, 1-s))
            if t == t0:
                assert all(src[k] == bank["root"][k] for k in ("parent", "student", "own_sha", "enemy_sha"))
        names = ("parent", "student") if t < t0 + WIDTH else ("parent",)
        commands = {k: R.live.clean(models[k].agent(own_obs, raw["game_configuration"])) for k in names}
        if t <= t0:
            assert all(commands[k] == bank["trace"][t][k] for k in names)
        if t0 <= t < t0 + WIDTH:
            row = rows[t-t0]
            assert all(row[k] == commands[k] for k in commands)
            changed = changed_units(commands["parent"], commands["student"])
            assert row["changed"] is bool(changed)
            assert row["worker_changed"] == K.agreements(own_obs, commands["parent"], commands["student"])
            account_rows.append({"t": t, "changed": changed})
            assert row["executed"] == commands[arm] == joint[s]
        elif t >= t0 + WIDTH:
            assert commands["parent"] == joint[s]
        env.step(joint)
        if t >= t0:
            teams = env.states[0].observation.players
            official_trajectory.append(([float(p[0]) for p in teams], [len(p[3]) for p in teams]))
    scores, cash, statuses = K.score(env)
    margin = scores[s] - scores[1-s]
    assert (scores, cash, statuses, margin) == (receipt["official_scores"], receipt["physical_cash"],
                                                receipt["status"], receipt["official_margin"])
    old = Q.BEGIN, Q.END
    try:
        Q.BEGIN, Q.END = t0, t0 + WIDTH
        account = Q.accounting(raw, tape, cash, account_rows)
    finally:
        Q.BEGIN, Q.END = old
    for (official_cash, trees), row in zip(official_trajectory, account["steps"], strict=True):
        assert all(abs(a-b) <= .011 for a, b in zip(official_cash, row["cash"], strict=True))
        assert trees == row["live_trees"]
    changes = [{"t": r["t"], "units": r["changed"]} for r in account_rows if r["changed"]]
    wall = receipt["own_public_seconds"]
    assert len(wall) == len(tape) and all(math.isfinite(x) and x >= 0 for x in wall)
    return dict(arm=arm, receipt_file_sha=R.sha(path), tape_sha=receipt["tape_sha"], tape=tape,
                official_scores=scores, physical_cash=cash, margin=margin, status=statuses,
                verified_joint_steps=t0+len(tape), verified_controlled_steps=controlled, verified_8da4_after_window=len(tape)-controlled,
                window_proposal_differences=changes, financials=account, original_suffix_clock=calls,
                window_end_cash=official_trajectory[controlled-1][0], window_end_trees=official_trajectory[controlled-1][1],
                terminal_trees=official_trajectory[-1][1], root_cash=account["start"],
                window_ledger_by_physical_seat=account["window"], root_to_terminal_ledger_by_physical_seat=account["whole"],
                own_seconds=dict(count=len(wall), total=sum(wall), max=max(wall), includes_root=len(wall)==len(tape)))


def first_order_difference(parent, student, seat, t0):
    for i, (a, b) in enumerate(zip(parent, student)):
        if a[seat] != b[seat]:
            return dict(t=t0+i, parent=a[seat], student=b[seat], units=changed_units(a[seat], b[seat]))
    return None


def pair_economics(p, a, s):
    out = {}
    for key in ("whole", "window"):
        base, alt = p["financials"][key], a["financials"][key]
        by_category = {}
        for k in sorted(set().union(*(dict(x) for x in base+alt))):
            delta = [alt[who].get(k, 0) - base[who].get(k, 0) for who in (0, 1)]
            by_category[k] = dict(physical=delta, candidate_margin=delta[s]-delta[1-s])
        out[key] = by_category
    return out


def settle(plan, directory, parent_path=None, full_path=None, release_sha=None):
    assert __debug__ and len(plan["cases"]) == 2
    packages = {}
    for model, key, path in (("parent", "parent", parent_path), ("student", "full", full_path)):
        source = plan["packages"][key]
        packages[model] = Path(path) if path is not None else REPO / source["path"]
        assert R.sha(packages[model]) == source["sha256"]
    cases = plan["cases"]
    assert [(x["seat"], x["t0"]) for x in cases] == [(0,64),(1,160)]
    assert cases[0]["seed"] < cases[1]["seed"] and cases[0]["terrain"] != cases[1]["terrain"]
    assert [x["arm_order"] for x in cases] == [["parent","student"],["student","parent"]]
    reports, details, hardware = [], [], set()
    for case in cases:
        s, t0 = case["seat"], case["t0"]
        folder = Path(directory) / "native" / f"{case['seed']}_{s}"
        raw, bank, sha, opening = shared_bank(case, folder)
        identity = bank["identity"]
        assert identity["protocol"] == R.sha(HERE / "protocol.json")
        assert identity["source"] == case["source_sha256"] and identity["seat"] == s and identity["seed"] == case["seed"]
        assert identity["packages"] == {k:R.sha(path) for k,path in packages.items()}
        assert identity["runner"] == R.sha(HERE/"run.py") and identity["gate"] == R.sha(HERE/"freeze.py")
        assert identity["kernel"] == R.sha(CARGO / "learned_carbon_observation/native_response/run.py")
        assert identity["original"] == R.live.MASTER_SHA == R.sha(R.live.SOURCE)
        if release_sha is not None:
            assert identity["release"] == release_sha
        R.native_clock.validate_host(bank["host"], [bank["cpu"]])
        hardware.add((bank["host"]["hostname"], bank["host"]["cpu_model"], bank["host"]["kernel"]))
        arms = {arm: replay(case, folder, raw, bank, sha, arm, packages) for arm in case["arm_order"]}
        p, a = arms["parent"], arms["student"]
        own = first_order_difference(p["tape"], a["tape"], s, t0)
        enemy = first_order_difference(p["tape"], a["tape"], 1-s, t0)
        reports.append(dict(seed=case["seed"], seat=s, t0=t0, terrain=case["terrain"], source_sha=case["source_sha256"],
            shared_sha=sha, cpu=bank["cpu"], opening=opening, first_executed_own_difference=own,
            first_executed_enemy_difference=enemy, margin_delta=a["margin"]-p["margin"],
            first_enemy_diff_precedes_first_own=bool(enemy and (not own or enemy["t"] < own["t"])),
            common_length_compared=min(len(p["tape"]), len(a["tape"])),
            common_window_turns=min(WIDTH, len(p["tape"]), len(a["tape"])),
            different_executed_turns_in_common_window_by_physical_seat=[
                sum(x[i] != y[i] for x,y in zip(p["tape"][:WIDTH], a["tape"][:WIDTH])) for i in (0,1)],
            arms={k:{key:v for key,v in row.items() if key not in {"tape","financials"}} for k,row in arms.items()},
            economics_delta_by_physical_seat=pair_economics(p, a, s)))
        for arm, result in arms.items():
            details.append((folder / (arm+"_verified_ledger.json.gz"), result["financials"]))
    assert len(hardware) == 1
    for path, data in details:
        R.write(path, data)
    clocks = [r["opening"] for r in reports] + [arm["original_suffix_clock"] for r in reports for arm in r["arms"].values()]
    totals = dict(calls=sum(c["calls"] for c in clocks), depth_zero=sum(len(c["zero_depth_steps"]) for c in clocks),
                  max_wall=max(c["wall_max"] for c in clocks if c["wall_max"] is not None))
    report = dict(packages=plan["packages"], freeze_canonical_sha=digest(plan), release_file_sha=release_sha,
                  cases=reports, unique_original_calls=totals,
                  summed_candidate_margin_delta=sum(x["margin_delta"] for x in reports))
    R.write(Path(directory) / "settlement.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--full", type=Path)
    args = parser.parse_args()
    settle(R.load(args.freeze), args.out, args.parent, args.full, R.sha(args.freeze))
