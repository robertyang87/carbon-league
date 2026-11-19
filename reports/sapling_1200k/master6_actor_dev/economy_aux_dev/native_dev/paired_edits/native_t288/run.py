"""One registered joint edit, with two fresh fully responsive original-examiner continuations."""
import argparse
import fcntl
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
EDIT=HERE.parent
DEV=EDIT.parent
ACTOR=DEV.parents[1]
BASE=ACTOR.parent
ROOT=BASE.parents[1]
TRAIN=ACTOR/"tree_response_training"
HIRING=ACTOR/"late_workforce_dev/native"
SOURCE=ROOT/"data/master3s2_source.py"
SEED,SEAT,T=621497120,1,288
ARMS=("auxiliary_root","control_root")
sys.path.insert(0,str(BASE/"root_recovery_branches"))
import live
sys.path.insert(0,str(ACTOR))
import native_progress
sys.path.insert(0,str(DEV/"game_analysis"))
import analyze as analysis
read,sha,check=analysis.read,analysis.sha,analysis.check
ENV={**os.environ,**{k:"1" for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS","CARBON_DEBUG")}}
SOURCE_SHA="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"


def freeze(path,data):
    if path.exists():
        check(read(path)==data,f"frozen result differs: {path.name}")
    else:
        live.save(path,data)


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(",",":"),sort_keys=True,allow_nan=False).encode()).hexdigest()


def inputs():
    doc=read(EDIT/"one_turn.json.gz")
    game,=[g for g in doc["games"] if g["seed"]==SEED and g["seat"]==SEAT]
    root,=[r for r in game["edit_roots"] if r["pre"]==T]
    tape_file=DEV/"candidate"/f"raw_{SEED}_{SEAT}.json.gz"
    tape=read(tape_file)
    check(sha(tape_file)==game["raw_sha256"]["candidate"] and len(root["edited"])==2,"wrong intervention")
    official=read(DEV/"candidate/official_replay.json")
    cert,=[g for g in official["runs"]["candidate"]["games"] if g["seed"]==SEED and g["seat"]==SEAT]
    check(cert["transcript_sha256"]==sha(tape_file) and cert["steps"]==299 and
          cert["map_sha256"]==tape["initial_terrain_u8_sha256"] and cert["official_reward_by_seat"]==tape["official_score_by_seat"],"wrong certified source")
    check(all(sha(analysis.official.ENGINE/k)==v for k,v in official["engine_source_sha256"].items()),"official engine differs")
    check(tape["actions"][T]==[root["observed_opponent_order"],root["complete_own_actions"]["candidate"]],"recorded root differs")
    git_sha=hashlib.sha256(subprocess.check_output(["git","show","HEAD:sources/master3s2/master3s2.py"],cwd=ROOT)).hexdigest()
    check(git_sha==sha(SOURCE)==SOURCE_SHA and tape["examiner_md5"]=="57f9a594cce4f2ec9e3573d8d9d26ca1","full original examiner differs")
    return tape,root,game,cert


def prepare():
    tape,root,game,cert=inputs()
    release=read(DEV/"release.json")
    selected={arm:root["complete_own_actions"]["candidate" if arm=="auxiliary_root" else "control"] for arm in ARMS}
    frozen={"schema":"two-full-original-endgame-responses-one-known-t288-root-v1","seed":SEED,"seat":SEAT,"turn":T,
            "development_only":True,"new_games":0,"formal_games":0,"max_original_search_calls":20,
            "arms":selected,"fixed_simultaneous_opponent_order":root["observed_opponent_order"],"edited":root["edited"],
            "conditions":"Only the complete friendly root command varies. Both arms use the frozen auxiliary candidate thereafter. Each original opponent observes its own seat history and responds freely. One fresh paired conditional stream; wall-clock budgets run normally.",
            "post_root_rng_seed":int(game["raw_sha256"]["candidate"][:8],16),"original_warm_cost":"Shared original import calibration from the exclusive court, then each arm adapts naturally.",
            "max_followup_turns_per_arm":10,"physical_cpus":[0],"criterion":"Report complete official U, margin, cash channels and the two edited carriers in both arms. A single known position measures one conditional difference.",
            "input_sha256":{"original_full_source":SOURCE_SHA,"candidate":sha(DEV/"candidate.py"),"control":sha(DEV/"control.py"),
                 "both_actual_edits":sha(EDIT/"one_turn.json.gz"),"raw":game["raw_sha256"]["candidate"],"terrain":cert["map_sha256"],
                 "official_certificate":sha(DEV/"candidate/official_replay.json"),"release":sha(DEV/"release.json"),
                 "original_live_helpers":sha(Path(live.__file__)),"public_engine_helpers":sha(BASE/"root_recovery_branches/branches.py"),
                 "freeze_runner":sha(Path(__file__)),"downstream_verifier":sha(HERE/"verify.py")}}
    check(frozen["input_sha256"]["candidate"]==release["candidate_pack_sha256"] and frozen["input_sha256"]["control"]==release["control_pack_sha256"],"wrong source pack")
    freeze(HERE/"protocol.json",frozen)
    return frozen


def court():
    flag,status=native_progress.hiring_progress()
    check(flag=="complete" and status["native_arms"]==4,"four independent hiring responses not certified")
    check(not competing(),"different native agent active")
    return {"hiring_final_sha256":sha(HIRING/"results.json"),"hiring_grant_sha256":sha(HIRING/"granted.json"),
            "last_turn_sha256":sha(ACTOR/"endgame_queue_dev/native_result.json"),
            "tree_acceptance_sha256":sha(TRAIN/"response_head/independent_acceptance.json"),
            "tree_08_aggregate_sha256":sha(TRAIN/"supplement_01/batches/intake_08/aggregate.json"),
            "cpu":0,"protocol_sha256":sha(HERE/"protocol.json")}


def competing():
    workers=[]
    for p in Path("/proc").glob("[0-9]*"):
        try:
            parts=(p/"cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        if not parts or b"python" not in Path(os.fsdecode(parts[0])).name.encode():continue
        if any(Path(os.fsdecode(x)).name=="tree_pack_arena.py" for x in parts if x.endswith(b".py")):
            workers.append(p.name)
    return workers


def snapshot(rng):
    s=rng.get_state()
    return [s[0],s[1].tolist(),s[2],s[3],s[4]]


def resumestate(rng,r):
    rng.set_state((r[0],np.array(r[1],dtype=np.uint32),r[2],r[3],r[4]))


def facts(master):
    return {"_TEMP":master._TEMP,"_SEED":master._SEED,"_S_ABORT":master._S_ABORT,"_SEARCH_OK":bool(master._SEARCH_OK),"_RX":bool(master._RX),
            "_S_K":master._S_K,"_S_M":master._S_M,"_S_MMAX":master._S_MMAX,"_S_HOMEFIX":bool(master._S_HOMEFIX)}


def worker(arm):
    check(os.sched_getaffinity(0)=={0},"worker assigned wrong core")
    protocol=prepare()
    check(read(HERE/"grant.json")==court(),"different predecessor receipts")
    tape,root,_,_=inputs()
    env=live.begin(tape); cfg=tape["game_configuration"]
    own=live.module(DEV/"candidate.py","common_auxiliary_continuation")
    master=live.module(SOURCE,"full_original_examiner")
    config=facts(master)
    check(config["_TEMP"]==1. and config["_SEED"] is None and config["_SEARCH_OK"] and config["_RX"] and config["_S_ABORT"]==.58,"original opponent unavailable")
    shared=HERE/"shared.json"
    if not shared.exists():
        freeze(shared,{"protocol_sha256":sha(HERE/"protocol.json"),"original":config,"warm_cost":list(master._S_COST),"registered_rng_seed":protocol["post_root_rng_seed"]})
    initial=read(shared)
    check(initial["original"]==config and initial["protocol_sha256"]==sha(HERE/"protocol.json"),"wrong shared definition")
    master._S_COST[:]=initial["warm_cost"]
    rng=np.random.RandomState(initial["registered_rng_seed"])
    tracker=master.BeliefTracker(master._Bel,1-SEAT)
    folder=HERE/arm;folder.mkdir(exist_ok=True)
    started=time.monotonic()
    searched=0
    original=master._search
    def count(*args,**kwargs):
        nonlocal searched
        searched+=1
        return original(*args,**kwargs)
    master._search=count
    for t in range(T+1):
        j=tape["actions"][t];op=live.public(env,1-SEAT);s=live.public(env,SEAT)
        tracker.step(op["players"],t,op["carbon"]);tracker.set_actions(op["players"],j[1-SEAT])
        if t==0 or t==T:
            check(own.agent(s,cfg)==j[SEAT],"actual candidate history differs")
        else:own._FEAT.observe(s)
        if t<T:
            env.step(j)
        else:
            joint=[j[0],protocol["arms"][arm]]
            master._ST.clear();master._ST.update(tr=tracker,prev=op["players"],rng=rng)
            env.step(joint)
            assets=env.states[0].observation.players
            projected=root["outcomes"]["candidate" if arm=="auxiliary_root" else "control"]["post"]
            check([{"cash":p[0],"workers":p[2],"trees":p[3]} for p in assets]==projected,"wrong physical root")
            freeze(folder/"root.json",{"protocol_sha256":sha(HERE/"protocol.json"),"shared_sha256":sha(shared),
                                       "own_pre_observation_sha256":digest(s),"opponent_pre_observation_sha256":digest(op),
                                       "joint":joint,"post_assets":projected})
    while not env.done:
        t=env.states[0].observation.step
        check(T<t<=298 and t-T<=10,"remaining-call budget exhausted")
        o=live.public(env,0);s=live.public(env,1)
        expected=own.agent(s,cfg)
        path=folder/f"turn_{t}.json.gz";attempt=folder/f"search_{t}.json"
        pre={"turn":t,"arm":arm,"protocol_sha256":sha(HERE/"protocol.json"),"own_public_sha256":digest(s),"enemy_public_sha256":digest(o)}
        if path.exists():
            rec=read(path)
            check(read(attempt)==pre and rec["pre"]==pre and rec["joint"][SEAT]==expected,"resume differs from saved joint")
            tracker.step(o["players"],t,o["carbon"]);tracker.set_actions(o["players"],rec["joint"][0])
            master._ST["prev"]=o["players"]
            resumestate(rng,rec["rng_after"]);master._S_COST[:]=rec["cost_after"]
            env.step(rec["joint"])
            check([p[0] for p in env.states[0].observation.players]==rec["post_cash"],"saved continuation diverged")
            continue
        # Mark before entering the opponent. A crash mid-call cannot silently spend the same allowance twice.
        with attempt.open("x") as reservation:
            json.dump(pre,reservation,separators=(",",":"),allow_nan=False)
        before=searched; call_started=time.monotonic()
        enemy_action=master.agent(o,cfg)
        elapsed=time.monotonic()-call_started
        check(searched==before+1,"original failed to enter search exactly once")
        joint=[enemy_action,expected]
        for seat in (0,1):
            assets=env.states[0].observation.players[seat]
            check(not(set(joint[seat])-set(assets[1])-set(assets[2])),"command addresses a foreign entity")
        env.step(joint)
        stats=master._S_STATS
        freeze(path,{"pre":pre,"joint":joint,"post_cash":[p[0] for p in env.states[0].observation.players],
                     "original_search_calls":1,"wall_seconds":elapsed,"rng_after":snapshot(rng),"cost_after":list(master._S_COST),
                     "search_stats_session_to_date":{k:stats[k] for k in ("searched","changed","aborted","single")}})
    log=[read(p) for p in sorted(folder.glob("turn_*.json.gz"))]
    check(len(log)==len(list(folder.glob("search_*.json")))==env.states[0].observation.step-T-1<=10 and
          all(r["pre"]["turn"]==T+i+1 for i,r in enumerate(log)) and [p.status for p in env.states]==["DONE","DONE"],"wrong number of outcomes")
    scores=[p.reward for p in env.states]
    freeze(folder/"outcome.json",{"arm":arm,"root_sha256":sha(folder/"root.json"),"protocol_sha256":sha(HERE/"protocol.json"),
          "shared_sha256":sha(shared),"status":[p.status for p in env.states],"official_scores":scores,
          "final_t":env.states[0].observation.step,"margin":scores[1]-scores[0],"search_calls":len(log),
          "issued":[sha(p) for p in sorted(folder.glob("search_*.json"))],
          "completed":[sha(p) for p in sorted(folder.glob("turn_*.json.gz"))]})


def schedule():
    with (HERE/".owner.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        prepare()
        while not (HIRING/"results.json").exists():time.sleep(8)
        with (TRAIN/".continuation.lock").open("a") as machine:
            fcntl.flock(machine,fcntl.LOCK_EX)
            freeze(HERE/"grant.json",court())
            for arm in ARMS:
                if not (HERE/arm/"outcome.json").exists():
                    subprocess.run(["taskset","-c","0",sys.executable,"-u",str(Path(__file__).resolve()),arm],env=ENV,check=True)
            subprocess.run(["nice","-n19","taskset","-c","2",sys.executable,"-u",str(HERE/"verify.py")],env=ENV,check=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("prepare","schedule",*ARMS));mode=ap.parse_args().mode
    if mode=="prepare":prepare()
    elif mode=="schedule":schedule()
    else:worker(mode)
