"""Paired old-map interventions with live original responses. 'preflight' plays no branches."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ["CARBON_DEBUG"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
SEQUENCE = HERE.parent
ACTOR = HERE.parents[2]
sys.path.insert(0, str(ACTOR / "elite_role_mix_dev/conditional"))
import responses as R

PARENT = SEQUENCE.parent / "candidate.py"
STUDENT = SEQUENCE / "candidate.py"
PLAN = HERE / "protocol.json"
REGISTERED = "23c5815f5f2dab654d56dc67defec99b0f9ede17a7704bca7b6b7f83ba582efe"
BEGIN, END = 128, 160


def archive(case, plan):
    file = ACTOR / case["actor_relative_path"]
    receipt = file.with_name(file.name.replace("raw_", "receipt_").replace(".json.gz", ".json"))
    assert R.sha(file) == case["source_sha256"] and R.sha(receipt) == case["ticket_sha256"]
    ticket = R.load(receipt)
    raw = R.load(file)
    assert ticket["binding"]["original"] == plan["examiner_sha256"] and ticket["game"]["raw_sha256"] == R.sha(file)
    assert case["terrain"] == ticket["binding"]["terrain"] == raw["initial_terrain_u8_sha256"]
    assert case["seed"] == raw["seed"] and case["seat"] == raw["candidate_seat"]
    assert not raw["errors"] and raw["steps"] >= END and not ticket["game"]["errors"]
    return raw


def preflight():
    if not __debug__:
        raise RuntimeError("assertions must be enabled")
    plan = R.load(PLAN)
    assert R.sha(PLAN) == REGISTERED
    assert R.sha(PARENT) == plan["parent_sha256"]
    source = subprocess.check_output(["git", "-C", str(R.ROOT), "show", "HEAD:sources/master3s2/master3s2.py"])
    assert hashlib.sha256(source).hexdigest() == R.sha(R.live.SOURCE) == plan["examiner_sha256"]
    policy = R.load(SEQUENCE / "protocol.json")
    assert policy["parent"] == R.sha(PARENT) and all(c["seed"] in policy["excluded_seeds"] for c in plan["cases"])
    for case in plan["cases"]:
        raw = archive(case, plan)
        env = R.live.begin(raw)
        for both in raw["actions"][:BEGIN]:
            env.step(both)
        assert not env.done and all(R.live.public(env, i)["step"] == BEGIN for i in (0, 1))
    return plan


def accounting(raw, tape, physical_cash, comparisons=()):
    """Independent rule replay with closing balances and separate ground versus cargo rent."""
    sim = R.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    assert list(sim.carbon) == raw["initial_full_carbon"]
    for joint in raw["actions"][:BEGIN]:
        sim.step(*joint)
    sim.cash = R.CashEvents(sim.cash)
    start = list(sim.cash)
    ledger = [[Counter(), Counter()] for _ in range(3)]
    s = raw["candidate_seat"]
    by_turn = {r["t"]: r["changed"] for r in comparisons}
    changed_orders, purchased = [], {}
    trajectory = []
    for joint in tape:
        t = sim.step_count
        current = {}
        for uid, alternative in by_turn.get(t, {}).items():
            w = sim.workers.get(uid)
            if w:
                assert w.player == s
                obj = dict(kind="collector" if w.is_collector else "planter", pos=w.pos, cargo=w.carbon)
            else:
                obj = dict(kind="base", pos=dict(sim.bases[s])[uid], cargo=None)
            entry = dict(t=t, uid=uid, **obj, alternatives=alternative, executed=joint[s].get(uid, "NONE"),
                         immediate_cash_events=[], purchased_tree=None)
            current[uid] = entry
            changed_orders.append(entry)
        sim.cash.events.clear()
        sim.step(*joint)
        trees = {tree.pos: tree for tree in sim.trees.values()}
        for tree_id in list(purchased):
            if tree_id not in sim.trees or sim.trees[tree_id].player != s:
                holding = purchased.pop(tree_id)
                holding["first_tenure_ended_at_observation"] = t+1
        for kind, seat, amount, *fields in sim.cash.events:
            if len(fields) < 2 or fields[0] not in current or kind == "tree_absorption":
                continue
            uid, site = fields[:2]
            entry = current[uid]
            entry["immediate_cash_events"].append([kind, seat, amount, site])
            if kind in ("plant", "seize"):
                tree = trees[site]
                assert tree.player == s and tree.id not in purchased
                entry["purchased_tree"] = dict(id=tree.id, site=site, acquisition=kind, paid=-amount,
                                               ground_rent=0., cargo_tax=0.)
                purchased[tree.id] = entry["purchased_tree"]
        keys = [0] + ([1] if t < END else []) + ([2] if t < 200 else [])
        for kind, side, amount, *fields in sim.cash.events:
            parts = {kind: amount}
            if kind == "tree_absorption":
                parts = {"ground_rent": amount - fields[2], "cargo_tax": fields[2]}
                holding = purchased.get(fields[0])
                if holding is not None:
                    assert side == s
                    holding["ground_rent"] += parts["ground_rent"]
                    holding["cargo_tax"] += parts["cargo_tax"]
            for k in keys:
                ledger[k][side].update(parts)
        trajectory.append({"t": t, "cash": list(sim.cash), "events": list(sim.cash.events),
                           "live_trees": [sum(z.player == s for z in sim.trees.values()) for s in (0, 1)]})
    assert np.allclose(physical_cash, sim.cash, atol=1e-6, rtol=0)
    assert np.allclose([sum(l.values()) for l in ledger[0]], np.asarray(physical_cash)-start, atol=1e-5, rtol=0)
    return dict(start=start, end=list(sim.cash), whole=ledger[0], window=ledger[1], through_199=ledger[2], steps=trajectory, changed_orders=changed_orders)


def run_arm(job):
    case, name, cpu, warm, binding = job
    os.sched_setaffinity(0, {cpu})
    plan = R.load(PLAN)
    raw = archive(case, plan)
    cfg = raw["game_configuration"]
    dest = HERE / f"{case['seed']}_{case['seat']}_{name}.json.gz"
    identity = dict(binding=binding, archive=case["source_sha256"], cpu=cpu, name=name, warm=warm)
    if dest.exists():
        previous = R.load(dest)
        assert previous["identity"] == identity and previous["statuses"] == ["DONE", "DONE"]
        return previous
    assert R.sha(PARENT) == binding["parent"] and R.sha(STUDENT) == binding["student"]
    assert R.sha(R.live.SOURCE) == plan["examiner_sha256"]
    env = R.live.begin(raw)
    s = case["seat"]
    op = 1-s
    # Both arms evaluate both policies only during the intervention. This compares real
    # physical orders on each arm's own current state and equalizes diagnostic workload.
    models = {n: R.live.module(p, "response_" + n) for n, p in (("parent", PARENT), ("student", STUDENT))}
    master = R.live.module(R.live.SOURCE, "territory_response_original")
    assert master._SEED is None and master._TEMP == 1 and master._RX and master._SEARCH_OK and master._S_ABORT == .58
    tracker = master.BeliefTracker(master._Bel, op)
    previous = None
    for t, joint in enumerate(raw["actions"][:BEGIN]):
        ours, other = R.live.public(env, s), R.live.public(env, op)
        tracker.step(other["players"], t, other["carbon"])
        tracker.set_actions(other["players"], joint[op])
        previous = other["players"]
        for m in models.values():
            if t == 0:
                m.agent(ours, cfg)
            else:
                m._FEAT.observe(ours)
        env.step(joint)
    seed = int(hashlib.sha256(f"territory-sequence-v1|{case['terrain']}|{s}|{BEGIN}".encode()).hexdigest()[:8], 16)
    master._S_COST[:] = warm
    master._ST.clear()
    master._ST.update(tr=tracker, prev=previous, rng=np.random.RandomState(seed))
    recorder = R.native_clock.Recorder(master)
    original = recorder.wrap(master.agent)
    before = R.native_clock.host_snapshot(master)
    R.native_clock.validate_host(before, [cpu])
    assert all(p["num_threads"] == 1 for p in before["pools"]) and before["torch_threads"] == 1
    assert all(z["cpus"] == [cpu] for z in before["thread_affinity"])
    started = time.perf_counter()
    tape, proposals, own_seconds = [], [], []
    while not env.done:
        ours = R.live.public(env, s)
        t = ours["step"]
        begin_policy = time.perf_counter()
        parent = R.live.clean(models["parent"].agent(ours, cfg))
        psecs = time.perf_counter() - begin_policy
        own = parent
        if t < END:
            begin_policy = time.perf_counter()
            student = R.live.clean(models["student"].agent(ours, cfg))
            ssecs = time.perf_counter() - begin_policy
            if name == "student":
                own = student
                psecs = ssecs
            changed = {u: [parent.get(u, "NONE"), student.get(u, "NONE")] for u in sorted(parent.keys() | student.keys())
                       if parent.get(u) != student.get(u)}
            proposals.append(dict(t=t, changed=changed, parent=parent, student=student))
        own_seconds.append(psecs)
        joint = [None, None]
        joint[s] = own
        joint[op] = R.live.clean(original(R.live.public(env, op), cfg))
        tape.append(joint)
        env.step(joint)
    elapsed = time.perf_counter() - started
    after = R.native_clock.host_snapshot(master)
    R.native_clock.validate_host(after, [cpu])
    assert [state.status for state in env.states] == ["DONE", "DONE"] and not recorder.errors
    assert len(recorder.calls) == len(tape) and recorder.calls[0]["step"] == BEGIN
    scores = [state.reward for state in env.states]
    cash = [p[0] for p in env.states[0].observation.players]
    data = dict(identity=identity, rng=seed, physical_seat=s, official_scores=scores, margin=scores[s]-scores[op],
                statuses=[state.status for state in env.states], host_before=before, host_after=after,
                joint_tape=tape, clock=R.native_clock.summarize(recorder.calls, master._S_ABORT, master._S_D2),
                original_calls=recorder.calls, active_own_public_seconds=own_seconds, branch_wall_seconds=elapsed,
                comparisons=proposals, changed_turns=sum(bool(row["changed"]) for row in proposals),
                changed_units=sum(len(row["changed"]) for row in proposals), cash=accounting(raw,tape,cash,proposals))
    R.write(dest, data)
    return data


def run(candidate_sha, receipt_sha):
    plan = preflight()
    lock = R.load(HERE / "qualified.json")
    assert R.sha(STUDENT) == candidate_sha == lock["student"] and receipt_sha == lock["receipt"]
    assert R.sha(SEQUENCE / lock["receipt_file"]) == receipt_sha
    assert lock["protocol"] == REGISTERED and lock["runner"] == R.sha(Path(__file__))
    bind = dict(protocol=REGISTERED, runner=R.sha(Path(__file__)), parent=R.sha(PARENT), student=R.sha(STUDENT),
                qualification=R.sha(HERE/"qualified.json"), weights=R.sha(SEQUENCE/"candidate.npz"),
                preregistered_fit=R.sha(SEQUENCE/"protocol.json"))
    assert bind["weights"] == lock["weights"] and bind["preregistered_fit"] == lock["fit_protocol"]
    rows = []
    with R.LOCK.open("a") as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i, case in enumerate(plan["cases"]):
            warm_file = HERE / f"warm_{case['seed']}.json"
            if not warm_file.exists():
                with ProcessPoolExecutor(1) as pool:
                    cost = pool.submit(R.native_warm).result()
                R.write(warm_file, dict(binding=bind, cost=cost))
            w = R.load(warm_file)
            assert w["binding"] == bind
            with ProcessPoolExecutor(2) as pool:
                a, b = pool.map(run_arm, [(case, "parent", i, w["cost"], bind), (case, "student", 1-i, w["cost"], bind)])
            assert a["rng"] == b["rng"]
            s = case["seat"]
            def rent(z):
                led = z["cash"]["through_199"]
                return led[s].get("ground_rent", 0) - led[1-s].get("ground_rent", 0)
            rows.append(dict(seed=case["seed"], seat=s, parent_scores=a["official_scores"], student_scores=b["official_scores"],
                             margin_delta=b["margin"]-a["margin"], ground_margin_delta=rent(b)-rent(a),
                             actual_changed_turns=b["changed_turns"], actual_changed_units=b["changed_units"],
                             shared_root_original_order_equal=a["joint_tape"][0][1-s] == b["joint_tape"][0][1-s],
                             opponent_different_orders=sum(x[1-s]!=y[1-s] for x,y in zip(a["joint_tape"], b["joint_tape"]))))
    passed = all(z["margin_delta"]>0 and z["actual_changed_turns"]>=2 for z in rows) and sum(z["ground_margin_delta"] for z in rows)>0
    R.write(HERE/"results.json", dict(binding=bind, rows=rows, passed=passed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "run"])
    parser.add_argument("--candidate-sha")
    parser.add_argument("--receipt-sha")
    args = parser.parse_args()
    if args.stage == "preflight":
        print([(c["seed"],c["seat"]) for c in preflight()["cases"]])
    else:
        run(args.candidate_sha, args.receipt_sha)
