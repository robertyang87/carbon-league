"""Replan both sides after the recorded joint root with a reconstructed Master3s2 history."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from branches import OUT,HERE,ROOT,DIRS,PACK,PACK_HASH,begin,clean,digest,public,utility

SOURCE=ROOT/"data/master3s2_source.py"
MASTER_SHA="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def save(path,obj):
    payload=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    content=gzip.compress(payload,mtime=0) if path.suffix==".gz" else payload+b"\n"
    tmp=path.with_name(path.name+".tmp");tmp.write_bytes(content);tmp.replace(path)


def prepare(where=OUT):
    if digest(SOURCE.read_bytes())!=MASTER_SHA or digest(subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:sources/master3s2/master3s2.py"]))!=MASTER_SHA:
        raise ValueError("opponent source identity")
    path=where/"live_shared_init.json"
    if path.exists():return
    opp=module(SOURCE,"recovery_warm_master")
    rng=np.random.RandomState(None).get_state()
    save(path,dict(mt_algorithm=rng[0],mt_keys=rng[1].tolist(),mt_index=rng[2],has_normal=rng[3],stored_normal=rng[4],
                   warm_cost=list(opp._S_COST),source_sha256=MASTER_SHA))


def run(arm,where=OUT,case=None,resume=False,stop_after=None):
    cfg=json.loads((where/"live_protocol.json").read_bytes())
    shared=json.loads((where/"live_shared_init.json").read_bytes())
    seed,seat,k=cfg["seed"],cfg["seat"],cfg["family"]; op=1-seat
    if case is None:
        source=DIRS[seed]
        baseline=json.loads((OUT/"results.json").read_bytes())["games"]
        available=[(g,v["root"]) for g in baseline for family,v in g["roots"].items() if family==k]
        if len(available)!=1:raise ValueError("root family is not unique")
        g,r=available[0]
    else:
        source=ROOT/case["source_dir"];g,r=case["record"],case["root"]
    t0=r["t"]
    if (g["seed"],g["seat"],t0)!=(seed,seat,cfg["root_t"]):raise ValueError("wrong root")
    binding=dict(protocol_sha256=digest((where/"live_protocol.json").read_bytes()),shared_init_sha256=digest((where/"live_shared_init.json").read_bytes()),original_trajectory_sha256=g["source_sha256"],arm=arm,source_sha256=MASTER_SHA,pack_sha256=PACK_HASH)
    done=where/f"live_{arm}.json";checkpoint=where/f"live_{arm}_checkpoint.json.gz"
    if resume and done.exists():
        rec=json.loads(done.read_bytes())
        if (any(rec.get(k)!=v for k,v in binding.items()) or digest((where/rec["action_tape"]).read_bytes())!=rec["action_sha256"]
                or rec["statuses"]!=["DONE","DONE"]):raise ValueError("completed arm seal mismatch")
        checkpoint.unlink(missing_ok=True)
        print("verified completed arm",arm,seed,seat,flush=True);return
    raw_b=(ROOT/g["source"]).read_bytes()
    if digest(raw_b)!=g["source_sha256"] or digest(SOURCE.read_bytes())!=MASTER_SHA or digest(PACK.read_bytes())!=PACK_HASH:raise ValueError("input identity")
    raw=json.loads(gzip.decompress(raw_b));cert=json.loads((source/"official_replay.json").read_bytes())["runs"][source.name]["games"]
    raw["initial_terrain_u8_sha256"]=next(c["map_sha256"] for c in cert if (c["seed"],c["seat"])==(seed,seat))
    env=begin(raw);own=module(PACK,"recovery_live_student");master=module(SOURCE,"recovery_live_master")
    assert master._SEED is None and master._TEMP==1 and master._SEARCH_OK and master._RX
    assert list(master._S_COST)!=[] and shared["source_sha256"]==MASTER_SHA
    master._S_COST[:]=shared["warm_cost"]
    tracker=master.BeliefTracker(master._Bel,op);hist=None;feat=None;first=None;public_at_root=None
    for t,commands in enumerate(raw["actions"][:t0+1]):
        visible=public(env,op); own_obs=public(env,seat)
        tracker.step(visible["players"],t,visible["carbon"]);tracker.set_actions(visible["players"],commands[op]);hist=visible["players"]
        if t==0:
            z=own.agent(own_obs,raw["game_configuration"]);feat=own._FEAT
            if clean(z)!=clean(commands[seat]):raise ValueError("step zero differs")
        elif t<t0:feat.observe(own_obs)
        if t<t0:env.step(commands);continue
        z=clean(own.agent(own_obs,raw["game_configuration"]))
        if z!=clean(commands[seat]):raise ValueError("original root differs")
        z0=dict(z)
        if r["worker"] not in own_obs["players"][seat][2] or z.get(r["worker"],"NONE")!=r["old"]:
            raise ValueError("focal unit or baseline command differs")
        if arm!="parent":
            if r["alternative"]=="NONE":z.pop(r["worker"],None)
            elif r["alternative"] in ("UP","RIGHT","DOWN","LEFT"):z[r["worker"]]=r["alternative"]
            else:raise ValueError("invalid worker action")
        changed={uid for uid in z0.keys()|z.keys() if z0.get(uid)!=z.get(uid)}
        if changed!=({r["worker"]} if arm!="parent" else set()):raise ValueError("unexpected joint action changes")
        a=commands.copy();a[seat]=z;first=a
        public_at_root=digest(json.dumps(visible,sort_keys=True,separators=(",",":")).encode())
        env.step(a)
    rng=np.random.RandomState()
    rng.set_state((shared["mt_algorithm"],np.array(shared["mt_keys"],dtype=np.uint32),shared["mt_index"],shared["has_normal"],shared["stored_normal"]))
    master._ST.clear();master._ST.update(tr=tracker,prev=hist,rng=rng)
    nsearch=0; real_search=master._search
    def count(*args,**kw):
        nonlocal nsearch
        nsearch+=1
        return real_search(*args,**kw)
    master._search=count
    actions=[first];trace=[];start=time.monotonic();prev_elapsed=0.;resumed=0
    if resume and checkpoint.exists():
        cp=json.loads(gzip.decompress(checkpoint.read_bytes()))
        if cp["identity"]!=binding or cp["actions"][0]!=first:raise ValueError("checkpoint identity")
        for a in cp["actions"][1:]:
            visible=public(env,op);own._FEAT.observe(public(env,seat))
            tracker.step(visible["players"],visible["step"],visible["carbon"])
            tracker.set_actions(visible["players"],a[op]);master._ST["prev"]=visible["players"]
            env.step(a)
        if cp["cash"]!=[p[0] for p in env.states[0].observation.players] or cp["t"]!=env.states[0].observation.step:
            raise ValueError("checkpoint replay mismatch")
        actions=cp["actions"];trace=cp["trace"];nsearch=cp["nsearch"];prev_elapsed=cp["elapsed"]
        sr=cp["rng"];rng.set_state((sr[0],np.array(sr[1],dtype=np.uint32),sr[2],sr[3],sr[4]))
        master._S_COST[:]=cp["cost"];ms=cp["stats"]
        for key in ("depth","rolls","cands"):ms[key]={int(k):v for k,v in ms[key].items()}
        master._S_STATS.update(ms);resumed=cp.get("resumes",0)+1
        print("resume",seed,seat,arm,cp["t"],flush=True)
    baseline_calls=nsearch
    while not env.done:
        if env.states[0].observation.step%30==0:
            save(where/f"live_{arm}_status.json",dict(t=env.states[0].observation.step,calls=nsearch,elapsed=round(prev_elapsed+time.monotonic()-start,1)))
        t=env.states[0].observation.step
        a=[None,None]
        a[seat]=own.agent(public(env,seat),raw["game_configuration"])
        a[op]=master.agent(public(env,op),raw["game_configuration"])
        for s,cmd in enumerate(a):
            p=env.states[0].observation.players[s]
            if set(cmd)-set(p[1])-set(p[2]):raise ValueError("foreign entity")
        env.step(a);actions.append(a)
        if t<t0+10 or t%25==0:
            after=env.states[0].observation
            trace.append(dict(t=t+1,cash=[p[0] for p in after.players],target_owner=next((s for s in (0,1) if r["tree"] in after.players[s][3]),None)))
        force_checkpoint=stop_after is not None and nsearch-baseline_calls >=stop_after
        if resume and not env.done and (nsearch%25==0 or force_checkpoint):
            sr=rng.get_state()
            cp=dict(identity=binding,t=env.states[0].observation.step,cash=[p[0] for p in env.states[0].observation.players],
                    actions=actions,trace=trace,nsearch=nsearch,elapsed=prev_elapsed+time.monotonic()-start,
                    rng=[sr[0],sr[1].tolist(),sr[2],sr[3],sr[4]],cost=list(master._S_COST),stats=master._S_STATS,resumes=resumed)
            save(checkpoint,cp)
            if force_checkpoint:
                print("checkpoint",seed,seat,arm,cp["t"],flush=True);return
    statuses=[p.status for p in env.states];scores=[p.reward for p in env.states]
    stats=master._S_STATS
    if statuses!=["DONE","DONE"] or nsearch!=len(actions)-1:raise ValueError("live search coverage")
    log=where/f"live_{arm}_actions.json.gz";save(log,actions)
    result=dict(arm=arm,seed=seed,seat=seat,root=r,root_opponent_observation_sha256=public_at_root,root_actions=first,
                score=scores,margin=scores[seat]-scores[op],u=utility(scores,seat),statuses=statuses,terminal_t=env.states[0].observation.step,
                search_calls=nsearch,search_completed=stats["searched"],search_changed=stats["changed"],search_aborted=stats["aborted"],search_single=stats["single"],
                elapsed=round(prev_elapsed+time.monotonic()-start,2),source_sha256=MASTER_SHA,pack_sha256=PACK_HASH,
                original_trajectory_sha256=g["source_sha256"],shared_init_sha256=digest((where/"live_shared_init.json").read_bytes()),
                protocol_sha256=digest((where/"live_protocol.json").read_bytes()),trace=trace,action_tape=log.name,action_sha256=digest(log.read_bytes()))
    if resume:result["resumed_checkpoints"]=resumed
    save(done,result)
    checkpoint.unlink(missing_ok=True)
    (where/f"live_{arm}_status.json").unlink(missing_ok=True)
    print(arm,scores,result["margin"],result["u"],nsearch,stats["searched"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm",choices=["prepare","capture","parent"]);a=p.parse_args()
    if a.arm=="prepare":prepare()
    else:run(a.arm)
