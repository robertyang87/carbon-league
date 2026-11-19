"""Run only the authorized two new maps. Restore an interrupted match from the last committed joint action."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
import hashlib
import importlib.util
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
BASE=ACTOR.parent
ROOT=BASE.parents[1]
PLAN=HERE/"protocol.json"
OUT=HERE/"native_games"
PARENT=ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py"
CANDIDATE=HERE/"candidate.py"
CAND_SHA="dd1690c1732ca829a6ebc200fa9ec88e21d5bdd1f8f7cf8cc164674e1fe8cb59"
LOCK=ACTOR/"tree_response_training/.continuation.lock"
sys.path.insert(0,str(BASE))
from tree_pack_arena import arena
import official_game_replay as official
from tree_win_gate import validate_examiner_configuration,validate_game
import numpy as np


def sha(v):
    return hashlib.sha256(v if isinstance(v,bytes) else v.read_bytes()).hexdigest()


def pack(v):
    return json.dumps(v,allow_nan=False,separators=(",",":"),sort_keys=True).encode()


def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)


def checked(v,msg):
    if not v:raise ValueError(msg)


def save(p,v):
    data=pack(v);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".{os.getpid()}.tmp")
    tmp.write_bytes(gzip.compress(data,compresslevel=1,mtime=0) if p.suffix==".gz" else data+b"\n")
    os.replace(tmp,p)


def agreed(p,v):
    if p.exists():checked(read(p)==v,"existing receipt changed: "+str(p))
    else:save(p,v)


def binding():
    plan=read(PLAN);p=read(HERE/"package.json");op=read(HERE/"opportunities.json.gz");cases=read(HERE/"precedents.json")
    checked(plan["new_dev"]["max_games"]==8 and len(set(plan["new_dev"]["seeds"]))==2 and plan["new_dev"]["seats"]==[0,1]
            and plan["new_dev"]["arms"]==["parent","candidate"],"wrong fixed native schedule")
    checked(sha(CANDIDATE)==CAND_SHA==p["candidate_sha256"]==op["candidate_sha256"] and
            sha(PARENT)==plan["parent_sha256"]==p["parent_sha256"] and sha(PLAN)==p["protocol_sha256"]==op["protocol_sha256"],"candidate identity")
    checked(op["opportunity_gate"]=="PASS" and op["n_games"]==32 and op["n_maps"]==16 and op["n_windows"]==64
            and op["changed"]>=8 and op["changed_maps"]>=4 and op["changed_actual_1step_payments"]>=2,"opportunity gate")
    checked(cases["result"].startswith("No exact previous complete original") and len(cases["prior_experiments"])>=10,"precedents not cleared")
    checked(sha(arena.prior.SOURCE)==plan["original_sha256"]==sha(subprocess.check_output(
        ["git","show","HEAD:sources/master3s2/master3s2.py"],cwd=ROOT)),"canonical full original differs")
    gatefile=ACTOR/"loaded_cargo_defense_dev/verdict.json"
    checked(gatefile.exists(),"prior cargo court not released")
    return {"protocol_sha256":sha(PLAN),"opportunities_sha256":sha(HERE/"opportunities.json.gz"),
            "parent_sha256":sha(PARENT),"candidate_sha256":CAND_SHA,"precedents_sha256":sha(HERE/"precedents.json"),
            "original_sha256":sha(arena.prior.SOURCE),"engine_sha256":{x:sha(official.ENGINE/x) for x in
            ("zerosum_env/core.py","zerosum_env/envs/carbon/carbon.py","zerosum_env/envs/carbon/helpers.py","zerosum_env/envs/carbon/idgen.py")},
            "arena_sha256":sha(Path(arena.__file__)),"checkpoint_code_sha256":sha(Path(__file__)),
            "status":"development_only","formal_games":0}


def preflight(identity):
    libfile=ACTOR/"stratified_action_value_dev/source/pilot.py"
    spec=importlib.util.spec_from_file_location("shared_full_world_map_checker",libfile)
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    lib.HERE,lib.PLAN,lib.OUTPUT=HERE,PLAN,OUT
    lb=lib.read
    def protocol_view(p):
        d=lb(p)
        if p==PLAN:
            with zipfile.ZipFile(HERE/"native_preflight_0.zip") as z:rev=json.loads(z.read("identity.json"))["frozen_git_revision"]
            d={**d,"maps":d["new_dev"]["seeds"],"source_revision":rev}
        return d
    lib.read=protocol_view
    lib.gate=lambda:("PASS",identity)
    import map_identity as mi
    from tree_win_gate import RESERVED_RANGE
    p=protocol_view(PLAN);fp,ls,lt=mi.ladder_boards()
    _,es,et=mi.exposed_boards([])
    boards={s:mi.local_board(s) for s in p["maps"]};own=lib.known_ownership(read(BASE/mi.LOCAL_EXPOSURE),boards)
    catalog=ACTOR/"freight_deployment/first_response/map_preflight.json"
    for i,(s,b) in enumerate(boards.items()):
        with zipfile.ZipFile(HERE/f"native_preflight_{i}.zip") as z:v=json.loads(z.read("identity.json"))
        checked(v["candidate_seed"]==s and v["candidate_initial_terrain_u8_sha256"]==v["official_opening_sha256"]==b
                and v["current_declaration"]["sha256"]==sha(PLAN) and v["frozen_git_revision"]==p["source_revision"]
                and not v["conflicts"] and not v["same_terrain_matches"] and not v["reserved"] and not v["ladder_seed_or_terrain_match"]
                and v["historical_catalog_loaded_sha256"]==sha(catalog) and v["verdict"]=="eligible_in_checked_scope",
                "initial source-wide map scan")
        checked(s not in ls and b not in lt and (s in own or (s not in es and b not in et)) and not RESERVED_RANGE[0]<=s<RESERVED_RANGE[1],
                "current registered seed or opening overlap")
    checked(len(set(boards.values()))==2,"map alias")
    return boards,{"current_paths":lib.newer_exposures(boards),"ladder_sources":len(mi.SOURCES),"ladder_terrains":len(lt),"ladder_sha256":fp}


def save_progress(path, identity, sim, actions,times,ev,opp,side,seed,arm):
    rng=opp._ST["rng"].get_state()
    save(path,{"binding":identity,"seat":side,"seed":seed,"arm":arm,"step":sim.step_count,"state_sha256":sha(pack(sim.get_state())),
               "actions":actions,"search":ev,"times":times,"stats":opp._S_STATS,"cost":opp._S_COST,
               "rng":[rng[0],rng[1].tolist(),int(rng[2]),int(rng[3]),float(rng[4])]})


def replay_decision(record,opp,agent,obs,cfg,side,player,sim,events):
    t=sim.step_count;expected=record["actions"][t][player]
    if player==side:
        checked(agent(obs,cfg)==expected,"candidate prefix differs from committed run")
    else:
        st=opp._ST
        if t==0:
            st["tr"]=opp.BeliefTracker(opp._Bel,player);st["prev"]=None
        st["tr"].step(obs["players"],t,obs["carbon"])
        st["tr"].set_actions(obs["players"],expected)
        st["prev"]=obs["players"]
        events.append(record["search"][t])
    return expected


def restore(record,opp,sim):
    checked(sha(pack(sim.get_state()))==record["state_sha256"],"replayed map/cash/worker state differs")
    alg,keys,ix,gauss,cached=record["rng"]
    rng=np.random.RandomState();rng.set_state((alg,np.array(keys,dtype=np.uint32),ix,gauss,cached));opp._ST["rng"]=rng
    opp._net()
    opp._S_COST[:]=record["cost"]
    stats=record["stats"]
    for k in ("depth","rolls","cands"):
        if isinstance(stats.get(k),dict):stats[k]={int(i):v for i,v in stats[k].items()}
    opp._S_STATS.clear();opp._S_STATS.update(stats)


def played(job,record,checkpoint,identity,arm):
    """Instrument the authoritative play loop at its action and commit boundaries; no alternate result formula."""
    source=inspect.getsource(arena.play)
    def insert(before,after):
        nonlocal source
        checked(source.count(before)==1,"arena checkpoint hook changed")
        source=source.replace(before,after)
    insert("                action = agent(sim.obs(player), sim.config)",
           "                if record and sim.step_count < record['step']:\n"
           "                    action = replay_decision(record,opp,agent,sim.obs(player),sim.config,side,player,sim,search_events)\n"
           "                else:\n                    action = agent(sim.obs(player), sim.config)")
    insert("                times[player].append(time.monotonic()-start)",
           "                times[player].append(record['times'][player][sim.step_count] if record and sim.step_count<record['step'] else time.monotonic()-start)")
    insert("        if any(extinct):\n            break",
           "        if record and sim.step_count==record['step']:\n            restore(record,opp,sim)\n"
           "        if not record or sim.step_count>record['step']:\n"
           "            save_progress(checkpoint,identity,sim,actions_log,times,search_events,opp,side,seed,arm)\n"
           "        if any(extinct):\n            break")
    ns={**vars(arena),"restore":restore,"replay_decision":replay_decision,"save_progress":save_progress,
        "checkpoint":checkpoint,"identity":identity,"record":record,"arm":arm}
    exec(compile(source,str(Path(arena.__file__))+"::saved-turns","exec"),ns)
    return ns["play"](job)


def one(task):
    arm,seed,seat,cpu,board,identity=task;dest=OUT/arm;dest.mkdir(exist_ok=True,parents=True)
    receipt=dest/f"receipt_{seed}_{seat}.json";raw=dest/f"raw_{seed}_{seat}.json.gz";rowfile=dest/f"row_{seed}_{seat}.json"
    ck=dest/f"progress_{seed}_{seat}.json.gz";ticket=dest/f"ticket_{seed}_{seat}.json"
    packfile=CANDIDATE if arm=="candidate" else PARENT;ph=sha(packfile)
    agreed(ticket,{"binding":identity,"seed":seed,"seat":seat,"terrain":board,"arm":arm})
    if not raw.exists():
        previous=read(ck) if ck.exists() else None
        if previous:
            checked(previous["binding"]==identity and (previous["seed"],previous["seat"],previous["arm"])==(seed,seat,arm)
                    and len(previous["actions"])==len(previous["search"])==previous["step"],"wrong saved-state identity")
        row=played((str(packfile),ph,seed,seat,cpu,str(dest),ph),previous,ck,identity,arm)
        save(rowfile,row)
    d=read(raw)
    checked(d["seed"]==seed and d["candidate_seat"]==seat and d["weight_identity"]==ph==d["pack_sha256"]
            and d["initial_terrain_u8_sha256"]==board and d["examiner_md5"]=="57f9a594cce4f2ec9e3573d8d9d26ca1"
            and not d["errors"] and not d["audit_messages"] and len(d["actions"])==d["steps"]
            and [v[0] for v in d["search_calls"]]==list(range(d["steps"]))
            and all(v[1] in {"searched","single","depth_zero_return_base"} for v in d["search_calls"])
            and any(v[1]=="searched" for v in d["search_calls"]),"bad native run")
    validate_examiner_configuration(d["configuration"])
    if rowfile.exists():row=read(rowfile);validate_game(row)
    else:
        row=dict(seed=seed,candidate_seat=seat,raw_sha256=sha(raw),raw_file=raw.name,actual_transitions=d["steps"],
                 result=d["result"],margin=d["margin"],official_score_by_seat=d["official_score_by_seat"],recovered_from_full_raw=True)
    v=official.check(dest,row);checked(v["map_sha256"]==board,"official terrain mismatch")
    result={"binding":identity,"official":v,"game":row}
    agreed(receipt,result)
    if ck.exists():ck.unlink()
    return result


def supervise():
    identity=binding();agreed(OUT/"binding.json",identity)
    while True:
        with LOCK.open("a") as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(6);continue
            boards,exposure=preflight(identity)
            save(OUT/"coordination.json",{"registered":read(PLAN)["new_dev"],"exposure":exposure,
                  "covered_terrain":boards,"global_lock":str(LOCK.relative_to(ROOT)),
                  "cargo_release_sha256":sha(ACTOR/"loaded_cargo_defense_dev/verdict.json")})
            pending=[];receipts=[]
            for j,(seed,b) in enumerate(boards.items()):
                for seat in (0,1):
                    wave=[]
                    for i,arm in enumerate(("parent","candidate") if (j+seat)%2==0 else ("candidate","parent")):
                        p=OUT/arm/f"receipt_{seed}_{seat}.json"
                        if p.exists():
                            result=read(p);checked(result["binding"]==identity and official.check(p.parent,result["game"])==result["official"],"changed completed game")
                            receipts.append((arm,result["game"]))
                        else:wave.append((arm,seed,seat,i,b,identity))
                    if wave and not pending:pending=wave
            if not pending:
                checked(len(receipts)==8 and len({(a,r["seed"],r["candidate_seat"]) for a,r in receipts})==8,"not all eight receipts")
                output={"status":"complete","binding":identity,"formal_games":0,"maps":[],"games":[]}
                for seed in boards:
                    entries=[(a,r) for a,r in receipts if r["seed"]==seed]
                    means={a:sum(v["margin"] for x,v in entries if x==a)/2 for a in ("parent","candidate")}
                    output["maps"].append({"seed":seed,"mean_official_margin":means,"delta":means["candidate"]-means["parent"]})
                output["mean_paired_official_margin_delta"]=sum(r["delta"] for r in output["maps"])/2
                for a,r in receipts:output["games"].append(dict(arm=a,seed=r["seed"],seat=r["candidate_seat"],result=r["result"],scores=r["official_score_by_seat"],margin=r["margin"],raw_sha256=r["raw_sha256"]))
                agreed(OUT/"results.json",output);print("completed 8 official games",flush=True);return
            print("wave",[(q[0],q[1],q[2]) for q in pending],flush=True)
            with ProcessPoolExecutor(max_workers=len(pending),mp_context=multiprocessing.get_context("spawn")) as pool:
                for result in pool.map(one,pending):
                    print("official",result["official"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--go",action="store_true")
    args=ap.parse_args()
    if args.go:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/".owner.lock").open("a") as owner:
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);supervise()
    else:print(preflight(binding()))
