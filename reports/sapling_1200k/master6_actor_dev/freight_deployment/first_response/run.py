"""Replay the exact frozen first online intervention, then run paired native responses."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent
BASE = DEPLOY.parent.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE / "root_recovery_branches"))
import live
from branches import begin, clean, public

PLAN = HERE / "protocol.json"
PLAN_SHA = "c3b99184367d375425e78c13ce8d03077d049737d38fd11acfac79f79572b88e"


def read(path):
    content = path.read_bytes()
    return json.loads(gzip.decompress(content) if path.suffix == ".gz" else content)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(path, value):
    if path.exists():
        assert read(path) == value, ("frozen evidence differs", path)
    else:
        live.save(path, value)


def stream(seed):
    state = np.random.RandomState(seed).get_state()
    return dict(mt_algorithm=state[0], mt_keys=state[1].tolist(), mt_index=state[2],
                has_normal=state[3], stored_normal=state[4])


def prepare(plan, item):
    seat = item["seat"]
    wd = HERE / f"seat{seat}"
    wd.mkdir(exist_ok=True)
    row = item["selection"]
    rawpath = ROOT / item["source"]
    assert sha(rawpath) == item["source_sha256"]
    raw = read(rawpath)
    assert raw["seed"] == plan["map_seed"] and raw["candidate_seat"] == seat
    assert raw["pack_sha256"] == plan["parent_sha256"]
    env = begin(raw)
    pack = live.module(DEPLOY / "agent.py", f"frozen_prefix_{seat}")
    prev = []
    for t, joint in enumerate(raw["actions"][:item["step"] + 1]):
        obs = public(env, seat)
        assert obs["step"] == t
        chosen = clean(pack.agent(obs, raw["game_configuration"]))
        ref = clean(joint[seat])
        if t == item["step"]:
            assert pack._FREIGHT.last["changed"]
            assert list(chosen.items()) == list(row["joint_after"].items())
            assert list(ref.items()) == list(row["joint_before"].items())
            assert pack._FREIGHT.last["public_predictions"][row["target_worker"]] == row["target_forecast"]
            retain(wd / "root_input.json.gz", dict(public_observation=obs, public_history=prev[-5:],
                   fixed_simultaneous_opponent_commands=joint[1-seat], ordered_parent=list(ref.items()),
                   ordered_intervention=list(chosen.items()), selection=row))
        else:
            assert chosen == ref and not pack._FREIGHT.last["changed"], ("earlier commit", t, seat)
            prev.append(obs)
            env.step(joint)
    root = dict(t=item["step"], seat=seat, worker=row["worker"], old=row["baseline_command"],
                alternative=row["alternative"], tree="", enemy_worker=row["target_worker"],
                target_worker=row["target_worker"], enemy_load=row["target_forecast"]["cargo"])
    case = dict(seed=raw["seed"], seat=seat, family="actual_first_deployment_freight", root_t=root["t"],
                source_dir=str(rawpath.parent.relative_to(ROOT)),
                record=dict(seed=raw["seed"], seat=seat, source=item["source"],source_sha256=item["source_sha256"]),
                root=root, plan_sha256=PLAN_SHA, root_input_sha256=sha(wd/"root_input.json.gz"))
    retain(wd/"live_protocol.json", case)
    init = wd / "live_shared_init.json"
    expected = stream(item["new_mt19937_uint32"])
    if not init.exists():
        master = live.module(live.SOURCE, f"master_warm_{seat}")
        retain(init, dict(**expected, warm_cost=list(master._S_COST), source_sha256=live.MASTER_SHA))
    stored = read(init)
    assert all(stored[k] == v for k,v in expected.items()) and stored["source_sha256"] == live.MASTER_SHA
    assert len(stored["warm_cost"]) > 0


def main():
    assert os.sched_getaffinity(0) == {3} and sha(PLAN) == PLAN_SHA
    plan = read(PLAN)
    assert sha(DEPLOY/"real_roots.json") == plan["source_roots_sha256"]
    assert sha(DEPLOY/"agent.py") == plan["deployed_sha256"]
    assert sha(Path(live.__file__)) == plan["responder_sha256"]
    assert sha(live.SOURCE) == plan["opponent_sha256"] == live.MASTER_SHA
    native = subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:sources/master3s2/master3s2.py"])
    assert hashlib.sha256(native).hexdigest() == live.MASTER_SHA
    live.PACK = ROOT / plan["parent_path"]
    live.PACK_HASH = plan["parent_sha256"]
    assert sha(live.PACK) == live.PACK_HASH
    if len(sys.argv) == 3:
        seat, arm = int(sys.argv[1]), sys.argv[2]
        assert arm in plan["arms"]
        wd = HERE / f"seat{seat}"
        live.run(arm, wd, read(wd/"live_protocol.json"), resume=True)
        return
    with PLAN.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in plan["roots"]:
            prepare(plan,item)
        for item in plan["roots"]:
            for arm in plan["arms"]:
                subprocess.run([sys.executable,"-u",str(Path(__file__)),str(item["seat"]),arm],check=True)


if __name__ == "__main__":
    main()
