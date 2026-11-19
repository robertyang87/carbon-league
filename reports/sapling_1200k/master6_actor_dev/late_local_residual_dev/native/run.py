"""Recorded-prefix paired interventions with the authentic wall-clock-limited opponent."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
ACTOR = DEV.parent
BASE = ACTOR.parent
ROOT = BASE.parents[1]
TRAIN = ACTOR / "tree_response_training"
PARENT = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
SOURCE = ROOT / "data/master3s2_source.py"
SHARED = ACTOR / "economy_aux_dev/native_dev/paired_edits/native_t288/shared.json"
sys.path[:0] = [str(DEV), str(BASE/"root_recovery_branches")]
import live
import runtime

def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dig(data):
    return hashlib.sha256(json.dumps(data, allow_nan=False, sort_keys=True, separators=(",",":")).encode()).hexdigest()

def check(condition, detail):
    if not condition:
        raise ValueError(detail)

def freeze(path, value):
    if path.exists():
        check(read(path) == value, f"frozen {path} differs")
    else:
        live.save(path, value)

def identities():
    canonical = hashlib.sha256(subprocess.check_output(["git","show","HEAD:sources/master3s2/master3s2.py"], cwd=ROOT)).hexdigest()
    check(canonical == live.MASTER_SHA == sha(SOURCE), "examiner identity")
    check(sha(PARENT) == "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0", "parent identity")
    return {"parent":sha(PARENT),"original":canonical,"runtime":sha(DEV/"runtime.py"),"validator":sha(DEV/"deploy.py"),
            "shared_warm_cost":sha(SHARED),"protocol":sha(DEV/"protocol.json"),"runner":sha(Path(__file__))}

def source(row):
    raw = ROOT/row["raw"]
    cert = raw.parent/"official_replay.json"
    check(sha(raw)==row["raw_sha256"] and sha(cert)==row["official_sha256"],"certified raw bytes changed")
    proof = read(cert)
    for path, h in proof["engine_source_sha256"].items():
        check(sha(ROOT/"code/env_src/carbon_challenge-master"/path)==h,"environment source changed")
    official = proof["runs"]["source"]
    tape = read(raw)
    entry, = [g for g in official["games"] if (g["seed"],g["seat"])==(row["seed"],row["seat"])]
    check(entry["transcript_sha256"]==row["raw_sha256"] and tape["initial_terrain_u8_sha256"]==entry["map_sha256"]==row["map"]
          and row["seat"]==tape["candidate_seat"] and row["seed"]==tape["seed"] and not tape["audit_messages"]
          and tape["official_score_by_seat"]==entry["official_reward_by_seat"] and entry["statuses"]==["DONE","DONE"]
          and official["pack_sha256"]==tape["pack_sha256"]==sha(PARENT) and tape["examiner_md5"]=="57f9a594cce4f2ec9e3573d8d9d26ca1","wrong game")
    return tape

def fresh_actor():
    actor = live.module(PARENT,"conditional_residual_common_v1")
    # Original v1's sole persistent playing state is Feat. The counters accumulated
    # in the decoder have no readers on the policy path. Save this root's exact inputs.
    decode = actor._mixed_actions
    ctx = []
    def capture(x, aux):
        ctx[:] = [x, aux]
        return decode(x,aux)
    actor._mixed_actions = capture
    scope = dict(vars(actor))
    exec(compile((DEV/"deploy.py").read_bytes(),"deploy.py","exec"),scope)
    return actor, ctx, scope["_local_edit"]

def selected(row):
    tape=source(row); s=row["seat"]; cfg=tape["game_configuration"]; env=live.begin(tape)
    actor, context, validate=fresh_actor()
    reasons={}; observed={}
    for t,j in enumerate(tape["actions"][:297]):
        obs=live.public(env,s)
        if t==0 or t>=286:
            base=actor.agent(obs,cfg)
            check(base==j[s],f"recorded v1 joint differs {row['seed']}/{s}/{t}")
        else:
            actor._FEAT.observe(obs)
        if 286<=t<=296:
            view={k:obs[k] for k in ("step","player","players","trees","carbon")}
            choices=runtime.variants(view,cfg,base)
            if choices:
                p,=choices
                edited,why=validate(view,cfg,base,{p["uid"]:p["action"]},*context,actor._FEAT)
                reasons[why]=reasons.get(why,0)+1
                if why=="accepted":
                    check(edited!=base and p["original"]==base.get(p["uid"],"NONE"),"local edit identity")
                    observed={"t":t,"proposal":p,"own_public_sha256":dig(obs),"op_public_sha256":dig(live.public(env,1-s)),
                              "base":base,"alt":edited,"root_enemy":j[1-s],
                              "paired_seed":int(dig([row["raw_sha256"],t,"local_response_1"])[:8],16)}
                    break
            else:
                reasons["no_raw_choice"]=reasons.get("no_raw_choice",0)+1
        env.step(j)
    return {"source":row,"screen":reasons,"root":observed or None}

def panel():
    plan=read(DEV/"protocol.json")
    rows=plan["source_groups"]
    check(len(rows)==16 and len({(r["map"],r["seat"]) for r in rows})==16 and
          not ({r["map"] for r in rows if r["split"]=="train"} & {r["map"] for r in rows if r["split"]=="held"}),"source overlap")
    sig=identities()
    for row in rows:
        p=HERE/f'selection_{row["seed"]}_{row["seat"]}.json'
        if not p.exists():
            freeze(p,{"identities":sig,**selected(row)})
            print("selected",row["seed"],row["seat"],(read(p)["root"] or {}).get("t"),flush=True)
        else:
            check(read(p)["identities"]==sig,"frozen selector differs")

def resume_rng(rng,z):
    rng.set_state((z[0],np.asarray(z[1],dtype="uint32"),z[2],z[3],z[4]))

def run_arm(row,root,arm,folder):
    out=folder/f"{arm}.json.gz"
    if out.exists():
        z=read(out);check(z["source"]==row and z["root_digest"]==dig(root) and z["arm"]==arm,"stale result");return z
    tape=source(row); cfg=tape["game_configuration"]; s=row["seat"]; e=1-s; T=root["t"]
    own,_,_=fresh_actor(); env=live.begin(tape); master=live.module(SOURCE,"conditional_residual_full_master")
    check(master._TEMP==1. and master._SEED is None and master._SEARCH_OK and master._RX and master._S_ABORT==.58,"original settings")
    master._S_COST[:]=read(SHARED)["warm_cost"]
    rng=np.random.RandomState(root["paired_seed"]); tracker=master.BeliefTracker(master._Bel,e)
    calls=0
    original=master._search
    def counted(*args,**kwargs):
        nonlocal calls
        calls+=1
        return original(*args,**kwargs)
    master._search=counted
    for t,j in enumerate(tape["actions"][:T+1]):
        op=live.public(env,e); obs=live.public(env,s)
        tracker.step(op["players"],t,op["carbon"]);tracker.set_actions(op["players"],j[e])
        if t in (0,T):
            check(own.agent(obs,cfg)==j[s],"continuation actor root differs")
        else:
            own._FEAT.observe(obs)
        if t<T:
            env.step(j)
    check(dig(obs)==root["own_public_sha256"] and dig(op)==root["op_public_sha256"] and j[s]==root["base"] and j[e]==root["root_enemy"],"different common root")
    joint=j.copy();joint[s]=root[arm]
    master._ST.clear();master._ST.update(tr=tracker,prev=op["players"],rng=rng)
    env.step(joint);tail=[joint]; times=[]
    while not env.done:
        t=env.states[0].observation.step
        check(T<t<=298,"terminal turn boundary")
        obs=live.public(env,s);op=live.public(env,e)
        us=own.agent(obs,cfg); before=calls; t0=time.perf_counter()
        them=master.agent(op,cfg);elapsed=time.perf_counter()-t0
        check(calls==before+1,"native search did not run")
        orders=[None,None];orders[s]=us;orders[e]=them;tail.append(orders)
        for side in (0,1):
            assets=obs["players"][side]
            check(not(set(orders[side])-set(assets[1])-set(assets[2])), "unknown entity")
        times.append({"step":t,"wall":elapsed,"own_public_sha256":dig(obs),"enemy_public_sha256":dig(op)})
        env.step(orders)
    check([z.status for z in env.states]==["DONE","DONE"] and calls==len(tail)-1,"official completion")
    scores=[p.reward for p in env.states]
    result={"source":row,"root_digest":dig(root),"arm":arm,"orders":tail,"search":times,"original_calls":calls,
            "original_config":read(SHARED)["original"],"own_public_sha256":root["own_public_sha256"],"enemy_public_sha256":root["op_public_sha256"],
            "rng_seed":root["paired_seed"],"cost_start":read(SHARED)["warm_cost"],"cost_end":list(master._S_COST),
            "official_scores":scores,"official_margin":scores[s]-scores[e],"final_step":env.states[0].observation.step,
            "final_status":[z.status for z in env.states],"final_cash":[p[0] for p in env.states[0].observation.players],"U":live.utility(scores,s)}
    freeze(out,result);return result

def run(split):
    check(os.sched_getaffinity(0)=={1},"stipulated original worker on a single reserved physical CPU")
    expected=identities()
    weights=read(DEV/"weights.json") if split=="held" else None
    inputs=[r for r in read(DEV/"protocol.json")["source_groups"] if r["split"]==split]
    if split=="held":
        freeze(HERE/"held_weight_sha256.json",{"sha256":sha(DEV/"weights.json"),"protocol":expected["protocol"]})
    with (TRAIN/".continuation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for row in inputs:
            p=HERE/f'selection_{row["seed"]}_{row["seat"]}.json'
            if not p.exists():freeze(p,{"identities":expected,**selected(row)})
            rec=read(p);check(rec["source"]==row and rec["identities"]==expected,"stale selected root")
            root=rec["root"]
            if not root: continue
            if split=="held":
                v=runtime.predict(root["proposal"],weights) if weights.get("enabled") else {}
                if not (v and v["margin"]>10 and v["principal"]>=-5 and v["own_delivery"]>=-5):continue
            folder=HERE/f'{split}_{row["seed"]}_{row["seat"]}'
            folder.mkdir(exist_ok=True)
            for arm in ("base","alt"):
                z=run_arm(row,root,arm,folder)
                print(split,row["seed"],row["seat"],root["t"],arm,z["official_margin"],z["original_calls"],flush=True)
        freeze(HERE/f"{split}_complete.json",{"identities":expected,"weights":sha(DEV/"weights.json") if weights is not None else None,"registered_map_seats":len(inputs)})

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("select","train","held"));opt=ap.parse_args()
    if opt.mode=="select": panel()
    else:run(opt.mode)
