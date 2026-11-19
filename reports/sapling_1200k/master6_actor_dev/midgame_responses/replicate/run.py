"""Append one independent full-search stream per frozen root without changing the original experiment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
sys.path.insert(0,str(ORIGINAL))
from study import live, read, save, sha, verify

PLAN = HERE / "protocol.json"
PLAN_SHA = "689d1ebf7464a22b1a6972baa42f7888886248200948901e863beb2b838356a1"
RNG_FIELDS = ("mt_algorithm","mt_keys","mt_index","has_normal","stored_normal")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected):
    require(sha(path)==expected, f"source identity differs: {path.name}")


def retain(path, content):
    if path.exists():
        require(path.read_bytes()==content, f"frozen evidence changed: {path.name}")
    else:
        tmp=path.with_suffix(path.suffix+".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)


def rng_hash(state):
    payload=json.dumps([state[k] for k in RNG_FIELDS],separators=(",",":")).encode()
    return hashlib.sha256(payload).hexdigest()


def prepare():
    pinned(PLAN,PLAN_SHA)
    plan=read(PLAN)
    for name,key in (("roots.json","old_roots_sha256"),("study.py","old_study_sha256"),
                     ("protocol.json","old_protocol_sha256"),("action_pair_labels.json","old_published_labels_sha256")):
        pinned(ORIGINAL/name,plan[key])
    old, fresh = set(),set()
    receipt=[]
    for item in plan["families"]:
        family=item["family"];source=ORIGINAL/family;wd=HERE/family;wd.mkdir(exist_ok=True)
        for name,key in (("live_protocol.json","old_case_sha256"),("root_input.json.gz","old_public_pair_sha256"),
                         ("live_shared_init.json","old_init_sha256"),("verified.json.gz","old_verified_sha256")):
            pinned(source/name,item[key])
        for name in ("live_protocol.json","root_input.json.gz"):
            retain(wd/name,(source/name).read_bytes())
        original=read(source/"live_shared_init.json");expected=dict(original)
        state=np.random.RandomState(item["new_seed_uint32"]).get_state()
        expected.update(dict(zip(RNG_FIELDS,[state[0],state[1].tolist(),state[2],state[3],state[4]])))
        new_bytes=json.dumps(expected,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()+b"\n"
        retain(wd/"live_shared_init.json",new_bytes)
        actual=read(wd/"live_shared_init.json")
        require(actual["warm_cost"]==original["warm_cost"] and actual["source_sha256"]==plan["opponent_sha256"],"warm cost or examiner differs")
        old.add(rng_hash(original));fresh.add(rng_hash(actual))
        receipt.append({"family":family,"case_sha256":sha(wd/"live_protocol.json"),"input_sha256":sha(wd/"root_input.json.gz"),
                        "old_stream_state_sha256":rng_hash(original),"new_stream_state_sha256":rng_hash(actual),
                        "shared_init_sha256":sha(wd/"live_shared_init.json"),"warm_cost_both_rounds":actual["warm_cost"],"seed":item["new_seed_uint32"]})
    require(len(old)==len(fresh)==2 and not old.intersection(fresh),"repeated MT19937 state")
    record={"plan_sha256":PLAN_SHA,"runner_sha256":sha(Path(__file__)),"streams":receipt}
    retain(HERE/"streams.json",json.dumps(record,indent=2,allow_nan=False).encode()+b"\n")
    return plan


def run_pair(item):
    family=item["family"];wd=HERE/family
    for arm in ("parent","alternative"):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),"arm","--family",family,"--arm",arm],check=True)
    v=verify(wd,read(wd/"live_protocol.json"))
    first,second=[read(wd/f"live_{k}.json") for k in ("parent","alternative")]
    require(first["root_opponent_observation_sha256"]==second["root_opponent_observation_sha256"],"different root opponent view")
    old=read(ORIGINAL/family/"verified.json.gz")
    require(v["shared_init_sha256"]!=old["shared_init_sha256"] and v["root"]==old["root"],"repeat not independent or root changed")
    for arm in ("parent","alternative"):
        newinfo=read(wd/f"live_{arm}.json");oldinfo=read(ORIGINAL/family/f"live_{arm}.json")
        require(newinfo["root_actions"]==oldinfo["root_actions"],"frozen first joint move changed")
        require(newinfo["original_trajectory_sha256"]==oldinfo["original_trajectory_sha256"],"source trajectory changed")
    print(f"audited {family} {v['delta_margin']:+.2f} ({v['delta_u']:+.6f})",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","arm"))
    parser.add_argument("--family",choices=("seize_enemy_tree","hold_own_tree"))
    parser.add_argument("--arm",choices=("parent","alternative"))
    args=parser.parse_args()
    plan=prepare()
    if args.command=="prepare":
        return
    if args.command=="arm":
        wd=HERE/args.family;cfg=read(wd/"live_protocol.json")
        pinned(Path(live.__file__),cfg["full_search_runner_sha256"])
        live.run(args.arm,wd,cfg,resume=True)
        return
    key=hashlib.sha256(str(HERE).encode()).hexdigest()[:20]
    with (Path(tempfile.gettempdir())/f"sapling-tree-repeat-{key}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for item in plan["families"]:
            run_pair(item)


if __name__=="__main__":
    main()
