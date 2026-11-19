"""Clear four new nominal seeds and physical terrains before any directions are played."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
FIT=HERE.parent
ACTOR=FIT.parent
BASE=ACTOR.parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
import map_identity
from tree_win_gate import RESERVED_RANGE

PLAN_PATH=HERE/"protocol.json"
PLAN_SHA="d245dc63d95f359706582f752925afa6c8a1104dae474c4af5bee17d211ae4dc"
PLAN=json.loads(PLAN_PATH.read_bytes())
MAPS=PLAN["train_maps"]+PLAN["development_maps"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def checked(raw,seed):
    board=raw["candidate_initial_terrain_u8_sha256"]
    assert raw["verdict"]=="eligible_in_checked_scope", (seed,raw["conflicts"],raw["same_terrain_matches"])
    assert raw["candidate_seed"]==seed and raw["frozen_git_revision"]==PLAN["source_revision"]
    assert raw["current_declaration"]["sha256"]==PLAN_SHA and not raw["reserved"]
    assert not raw["conflicts"] and not raw["same_terrain_matches"] and not raw["ladder_seed_or_terrain_match"]
    assert not RESERVED_RANGE[0]<=seed<RESERVED_RANGE[1]
    assert board==raw["official_opening_sha256"]==map_identity.local_board(seed)
    return board


def fetch(seed,staging):
    dest=staging/f"{seed}.json"
    if not dest.exists():
        result=subprocess.run([sys.executable,str(ACTOR/"freight_deployment/map_preflight.py"),
                           "--seed",str(seed),"--revision",PLAN["source_revision"],
                           "--declaration",str(PLAN_PATH.relative_to(ROOT)),"--out",str(dest),
                           "--registry",str(ACTOR/"freight_deployment/first_response/map_preflight.json")],
                         env={**os.environ,"GIT_CONFIG_COUNT":"1","GIT_CONFIG_KEY_0":"grep.threads","GIT_CONFIG_VALUE_0":"1"},check=False)
        if result.returncode!=0:raise ValueError(f"workspace inventory failed for map {seed}")
    return dest,checked(load(dest),seed)


def additional():
    seeds=set();boards=set();unknown=[];witness={}
    for p in sorted((BASE/"all_top5").glob("intake_*.json")):
        cards=load(p).get("games",[])
        for g in cards:
            if not isinstance(g,dict) or "id" not in g:continue
            gid=g["id"]
            if g.get("seed") is not None:seeds.add(g["seed"])
            replay=ROOT.parent/"sapling-all-top5-20261119/replays"/f"{gid}.json.gz"
            if not replay.exists():
                unknown.append(gid);continue
            raw=load(replay)
            ob=raw["steps"][0][0]["observation"]
            assert ob["step"]==0
            seed=raw["configuration"]["randomSeed"];board=map_identity.terrain_sha256(ob["full_carbon"])
            seeds.add(seed);boards.add(board);witness[gid]={"seed":seed,"terrain":board}
    return seeds,boards,sorted(set(unknown)),witness


def verify_historical_registries(current):
    paths=sorted(FIT.glob("map_audit_*.json.gz"))+sorted((FIT/"replication").glob("map_clearance_*.json.gz"))
    assert len(paths)==4
    historical_seeds=set();historical_terrains=set();sources={}
    for p in paths:
        audit=load(p)
        for row in audit["local_native_or_training_maps"]:
            if row["seed"] is not None:historical_seeds.add(row["seed"])
            historical_terrains.add(row["terrain"])
        sources[str(p.relative_to(FIT))]=sha(p)
    assert not set(MAPS)&historical_seeds and not set(current["boards"].values())&historical_terrains
    witness=dict(protocol_sha256=PLAN_SHA,nominal_seed_count=len(historical_seeds),terrain_count=len(historical_terrains),
                 older_inputs=sources,matches=0)
    path=HERE/"older_registry_crosscheck.json"
    if path.exists():assert load(path)==witness
    else:atomic(path,(json.dumps(witness,indent=2)+"\n").encode())


def main():
    os.sched_setaffinity(0,{0})
    os.setpriority(os.PRIO_PROCESS,0,19)
    assert sha(PLAN_PATH)==PLAN_SHA and len(set(MAPS))==4
    if (HERE/"clearance.json").exists():
        receipt=load(HERE/"clearance.json")
        assert receipt["protocol_sha256"]==PLAN_SHA and len(receipt["boards"])==4
        for seed in MAPS:
            path=HERE/f"map_audit_{seed}.json.gz"
            assert sha(path)==receipt["per_map_audit_sha256"][str(seed)]
            assert checked(load(path),seed)==receipt["boards"][str(seed)]
        verify_historical_registries(receipt)
        return
    staging=Path(tempfile.gettempdir())/("sapling-map-audit-"+PLAN_SHA[:16])
    staging.mkdir(exist_ok=True)
    with (staging/"check.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda seed:fetch(seed,staging),MAPS))
        registry_id,ladder_seeds,ladder_boards=map_identity.ladder_boards()
        extra_seeds,extra_boards,unknown,witness=additional()
        prior=load(FIT/"map_clearance.json")["boards"]
        older_maps=load(FIT/"replication/protocol.json")["maps"]
        retired_seeds=set(map(int,prior))|set(older_maps)
        retired_boards=set(prior.values())|{map_identity.local_board(s) for s in older_maps}
        boards={str(s):b for s,(_,b) in zip(MAPS,results)}
        assert len(set(boards.values()))==4
        assert not set(MAPS)&(ladder_seeds|extra_seeds|retired_seeds)
        assert not set(boards.values())&(ladder_boards|extra_boards|retired_boards)
        hashes={}
        for s,(path,_) in zip(MAPS,results):
            target=HERE/f"map_audit_{s}.json.gz"
            atomic(target,gzip.compress(path.read_bytes(),mtime=0))
            hashes[str(s)]=sha(target)
        record={"protocol_sha256":PLAN_SHA,"verdict":"eligible_in_checked_scope","boards":boards,
                "per_map_audit_sha256":hashes,"retired_earlier_stage_maps":sorted(retired_seeds),
                "ladder_registry_sha256":registry_id,"ladder_seed_count":len(ladder_seeds),
                "ladder_board_count":len(ladder_boards),"additional_intake_records":witness,
                "intake_without_available_opening":unknown}
        atomic(HERE/"clearance.json",(json.dumps(record,indent=2,ensure_ascii=False)+"\n").encode())
        verify_historical_registries(record)
        print("all four independent terrain identities cleared",boards,flush=True)


if __name__=="__main__":
    main()
