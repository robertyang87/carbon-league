"""Three-arm economic representation comparison with an immutable arena and serialized native work."""
import argparse
import fcntl
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
import tempfile
import time

# source_games imports an older file with the unqualified name 'build'.
HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
T=ACTOR/"tree_response_training"
spec=importlib.util.spec_from_file_location("sapling_prior_frozen_arena",ACTOR/"symmetry_dev/run.py")
arena=importlib.util.module_from_spec(spec)
spec.loader.exec_module(arena)
sys.path[:0]=[str(T/"supplement_01"),str(T)]
from source_games import ENV,ROOT,gate,trial,sha
from select_roots import load,retain

ARMS=("parent","control","candidate")
PLAN="553f899c47df00482d3fba1f1e41d09ef3860026034deff3db2005c425a084fc"


def require(value,message):
    if not value:
        raise ValueError(message)


def approved():
    plan=load(HERE/"protocol.json")
    release=load(HERE/"release.json")
    clearance=load(HERE/"clearance.json")
    mechanism=load(HERE/"mechanism.json")
    require(sha(HERE/"protocol.json")==PLAN==release["protocol_sha256"]==clearance["declaration_sha256"]
            ==mechanism["protocol_sha256"],"registration changed")
    require(clearance["verdict"]=="eligible_in_checked_scope" and set(clearance["boards"])==set(map(str,plan["maps"]))
            and mechanism["passed"] and mechanism["release_sha256"]==sha(HERE/"release.json"),"mechanism or clearance failed")
    for arm in ARMS[1:]:
        require(sha(HERE/(arm+".py"))==release[arm+"_pack_sha256"] and
                sha(HERE.parent/(arm+".npz"))==plan["student_actor_sha256"][arm],"changed pack or student")
    require(load(HERE.parent/"result.json")["eligible_for_native_comparison"] and
            sha(HERE.parent/"result.json")==release["training_result_sha256"],"wrong training result")
    return plan,release,clearance


def verify_arm(arm, binding):
    plan,rel,clear=approved()
    d=HERE/arm
    result=load(d/"results.json")
    official=load(d/"official_replay.json")
    h=result["header"]; independent=official["runs"][arm]
    require(official["engine_source_sha256"]==binding["official_engine_sources"] and
            h["source_sha256"]==binding["arena_sources"],"arena source mismatch")
    pkg=rel[arm+"_pack_sha256"]; weights=rel["parent_weight_manifest_sha256"] if arm=="parent" else sha(HERE/"release.json")
    require(h["pack_sha256"]==pkg==independent["pack_sha256"] and h["weight_sha256"]==weights
            and h["cpus"]==[0,1] and h["seeds"]==plan["maps"] and h["examiner_md5"]==plan["examiner_md5"]
            and result["status"]=="complete","wrong input pack, maps or examiner")
    index={(r["seed"],r["candidate_seat"]):r for r in result["games"]}
    proof={(r["seed"],r["seat"]):r for r in independent["games"]}
    expected={(seed,seat) for seed in plan["maps"] for seat in plan["physical_seats"]}
    require(index.keys()==proof.keys()==expected and len(result["games"])==len(independent["games"])==4,"missing seat or duplicate")
    rows={}
    for key,r in sorted(index.items()):
        seed,seat=key; o=proof[key]; terrain=clear["boards"][str(seed)]
        gate.validate_game(r); gate.verify_raw(d,r,pkg,weights,expected_terrain=terrain)
        gate.validate_examiner_configuration(load(d/r["raw_file"])["configuration"])
        scores=o["official_reward_by_seat"]; m=scores[seat]-scores[1-seat]
        require(m==r["margin"] and o["statuses"]==["DONE","DONE"] and scores==r["official_score_by_seat"]
                and o["map_sha256"]==terrain and o["steps"]==r["actual_transitions"]
                and o["transcript_sha256"]==r["raw_sha256"]==sha(d/r["raw_file"]),"official result or terrain differs")
        wdl=1 if m>0 else .5 if m==0 else 0
        u=.5*wdl+.5/(1+math.exp(-min(40.,max(-40.,m/1000))))
        rows[key]={"official_scores":scores,"margin":m,"U":u,"outcome":r["result"],"transitions":o["steps"],
                   "raw_sha256":r["raw_sha256"],"raw_file":r["raw_file"],"search":r["search_return_types"],
                   "wall_seconds":r["wall_seconds"]}
    return rows


def compare(groups,plan):
    rows=[{"seed":s,"seat":p,**{arm:groups[arm][s,p] for arm in ARMS}} for s in plan["maps"] for p in (0,1)]
    rollup={arm:{k:sum(r[arm]["outcome"]==k for r in rows) for k in ("win","draw","loss")} for arm in ARMS}
    candidates={}
    for baseline in ("parent","control"):
        dm=[r["candidate"]["margin"]-r[baseline]["margin"] for r in rows]
        du=[r["candidate"]["U"]-r[baseline]["U"] for r in rows]
        per_map={str(s):mean(v for r,v in zip(rows,dm) if r["seed"]==s) for s in plan["maps"]}
        center=mean(per_map.values()); se=stdev(per_map.values())/math.sqrt(len(per_map))
        criteria={"both_maps_positive":min(per_map.values())>0,"three_of_four_U":sum(v>0 for v in du)>=3,
                  "no_fewer_wins":rollup["candidate"]["win"]>=rollup[baseline]["win"]}
        candidates[baseline]={"per_map_mean_delta_margin":per_map,"avg_margin_change":center,
                              "illustrative_two_map_t95":[center-12.706204736*se,center+12.706204736*se],
                              "positive_U_pairs":sum(v>0 for v in du),"criteria":criteria,
                              "per_seat_delta_margin":{str(p):mean(v for r,v in zip(rows,dm) if r["seat"]==p) for p in (0,1)}}
    ok=all(all(v["criteria"].values()) for v in candidates.values())
    return {"paired_map_seats":rows,"rollup":rollup,"candidate_vs":candidates,
            "verdict":"exploratory_followup_signal" if ok else "retain_parent","formal":False}


def certified():
    plan,rel,_=approved();binding=load(HERE/"arena_binding.json")
    result=compare({arm:verify_arm(arm,binding) for arm in ARMS},plan)
    return {"schema":"sapling-economy-aux-three-arm-native-v1","protocol_sha256":PLAN,
            "release_sha256":sha(HERE/"release.json"),"mechanism_sha256":sha(HERE/"mechanism.json"),
            "arena_binding_sha256":sha(HERE/"arena_binding.json"),
            "competitors":{a:{"results_sha256":sha(HERE/a/"results.json"),
                              "official_sha256":sha(HERE/a/"official_replay.json")} for a in ARMS},**result}


def compete(lease):
    plan,rel,clear=approved()
    with lease.native_session(), arena.arena_snapshot(plan) as (base,sources), tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp);parent,weights,digest=trial.recipe("learned",tmp)
        require(digest==sha(parent)==plan["parent_pack_sha256"] and sha(weights)==rel["parent_weight_manifest_sha256"],"parent differs")
        expected=load(T/"source/official_replay.json")["engine_source_sha256"]
        engine=base.parents[1]/"code/env_src/carbon_challenge-master"
        require(all(sha(engine/k)==v for k,v in expected.items()),"official interpreter differs")
        binding={"revision":plan["source_revision"],"protocol_sha256":PLAN,"release_sha256":sha(HERE/"release.json"),
                 "map_clearance_sha256":sha(HERE/"clearance.json"),"arena_sources":sources,"official_engine_sources":expected}
        retain(HERE/"arena_binding.json",binding)
        for arm in ARMS:
            d=HERE/arm
            package=parent if arm=="parent" else HERE/(arm+".py")
            manifest=weights if arm=="parent" else HERE/"release.json"
            subprocess.run([sys.executable,"-u",str(base/"tree_pack_arena.py"),"--pack",str(package),
                            "--weights",str(manifest),"--pack-sha256",rel[arm+"_pack_sha256"],
                            "--run",str(d),"--cpus","0","1","--maps",*map(str,plan["maps"])],env=ENV,check=True)
            paper=tmp/(arm+".json")
            subprocess.run([sys.executable,str(base/"official_game_replay.py"),str(d),"--out",str(paper)],env=ENV,check=True)
            retain(d/"official_replay.json",load(paper));verify_arm(arm,binding)
        result=certified();retain(HERE/"comparison.json",result)
    lease.release()
    print(json.dumps(result["rollup"]),result["verdict"],flush=True)


def oversee():
    with (HERE/".owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("already supervised");return
        while not all((HERE/p).exists() for p in ("release.json","mechanism.json","clearance.json")) or not (T/"supplement_01/economy_lease.py").exists():
            time.sleep(5)
        import economy_lease
        if (HERE/"mechanism_skip.json").exists():
            economy_lease.skip()
        elif (HERE/"comparison.json").exists():
            require(load(HERE/"comparison.json")==certified(),"sealed result changed")
            economy_lease.release()
        else:
            compete(economy_lease)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("run","check","capsule"))
    arg=ap.parse_args()
    if arg.mode=="run":oversee()
    elif arg.mode=="check":print(json.dumps(certified()))
    else:
        with arena.arena_snapshot(load(HERE/"protocol.json")) as (base,src):
            print("frozen arena",len(src))
