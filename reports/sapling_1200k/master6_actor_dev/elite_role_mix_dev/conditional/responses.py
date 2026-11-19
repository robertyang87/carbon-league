"""Fixed-map registered spot responses from actual pre-change game tapes under the shared original-examiner clock."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
BASE=ACTOR.parent
ROOT=ACTOR.parents[2]
sys.path[:0]=[str(BASE/"root_recovery_branches"),str(BASE),str(ROOT/"code")]
import live
import native_clock
from fastsim import FastCarbon
from turning_point_ledger import CashEvents

PLAN=HERE/"match_protocol.json"
P=json.loads(PLAN.read_bytes())
LOCK=ACTOR/"tree_response_training/.continuation.lock"
PACK={"parent":ACTOR/"actor_partner_dev/candidate.py","candidate":HERE/"candidate.py"}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)


def write(p,data):
    raw=json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    raw=gzip.compress(raw,mtime=0) if p.suffix==".gz" else raw+b"\n"
    if p.exists():
        if p.read_bytes()!=raw:raise ValueError(f"existing evidence differs: {p}")
        return
    temp=p.with_suffix(".part")
    temp.write_bytes(raw); os.replace(temp,p)


def inputs(case):
    rawpath=ROOT/case["source"]
    assert sha(rawpath)==case["source_sha256"] and sha(PACK["parent"])==P["parent"] and sha(PACK["candidate"])==P["candidate"]
    assert sha(live.SOURCE)==live.MASTER_SHA==P["original"]
    ticket=load(rawpath.with_name(rawpath.name.replace("raw_","receipt_").replace(".json.gz",".json")))
    assert ticket["game"]["raw_sha256"]==sha(rawpath)
    raw=load(rawpath)
    assert raw["pack_sha256"]==P["parent"] and raw["seed"]==case["seed"] and raw["candidate_seat"]==case["seat"] and not raw["errors"]
    return raw


def select(case):
    raw=inputs(case)
    seat=case["seat"];models=[live.module(path,"selection_"+name) for name,path in PACK.items()]
    sim=FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    if list(sim.carbon)!=raw["initial_full_carbon"]:raise ValueError("map mismatch")
    for t,joint in enumerate(raw["actions"][:281]):
        view=sim.obs(seat)
        if t==0 or t>=265:
            acts=[live.clean(m.agent(view,raw["game_configuration"])) for m in models]
            if acts[0]!=live.clean(joint[seat]):raise ValueError("parent action not reproduced")
            if t>=265 and acts[0]!=acts[1]:
                return dict(**case,t=t,parent=acts[0],candidate=acts[1],opponent=joint[1-seat])
        else:
            for m in models:m._FEAT.observe(view)
        sim.step(*joint)
    return dict(**case,t=None)


def cash_summary(raw,seat,t0,tape,score):
    sim=FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    for pair in raw["actions"][:t0]:sim.step(*pair)
    sim.cash=CashEvents(sim.cash);before=list(sim.cash)
    ledger=[Counter(),Counter()]
    for pair in tape:
        sim.cash.events.clear();sim.step(*pair)
        for kind,side,delta,*_ in sim.cash.events:ledger[side][kind]+=delta
    if not np.allclose(sim.cash,score,atol=1e-6,rtol=0):raise ValueError("official to cash re-simulation differs")
    for s in (0,1):
        if abs(sum(ledger[s].values())-(score[s]-before[s]))>1e-5:raise ValueError("cash ledger doesn't close")
    return dict(pre=before,post=score,events=[dict(z) for z in ledger],
                tree_income_margin=ledger[seat]["tree_absorption"]-ledger[1-seat]["tree_absorption"],
                delivery_margin=ledger[seat]["own_delivery"]-ledger[1-seat]["own_delivery"])


def native_warm():
    os.sched_setaffinity(0,{0})
    return list(live.module(live.SOURCE,"neutral_response_warm")._S_COST)


def arm(job):
    case,name,cpu,shared,*replication=job;os.sched_setaffinity(0,{cpu})
    raw=inputs(case);t0=case["t"];seat=case["seat"];op=1-seat;cfg=raw["game_configuration"]
    dest=HERE/f"responses_{raw['seed']}_{seat}_{name}.json.gz"
    binding=dict(protocol=sha(PLAN),source=case["source_sha256"],pack=sha(PACK[name]),original=sha(live.SOURCE),cpu=cpu)
    if replication:
        dest=replication[0]/dest.name
        binding["replication_protocol"]=sha(replication[0]/"protocol.json")
    if dest.exists():
        r=load(dest)
        if r["binding"]!=binding or r["statuses"]!=["DONE","DONE"]:raise ValueError("incomplete or changed arm")
        return r
    own=live.module(PACK[name],"response_"+name); master=live.module(live.SOURCE,"response_master")
    if not (master._SEED is None and master._TEMP==1 and master._SEARCH_OK and master._RX and master._S_ABORT==.58):
        raise ValueError("original examiner configuration drift")
    env=live.begin(raw); tr=master.BeliefTracker(master._Bel,op)
    for t,joint in enumerate(raw["actions"][:t0+1]):
        ownv,opv=live.public(env,seat),live.public(env,op)
        tr.step(opv["players"],t,opv["carbon"]);tr.set_actions(opv["players"],joint[op])
        if t in (0,t0):
            a=live.clean(own.agent(ownv,cfg))
            if t==0 and a!=live.clean(joint[seat]):raise ValueError("opening mismatch")
        else:own._FEAT.observe(ownv)
        if t<t0:env.step(joint)
    if a!=case[name] or joint[op]!=case["opponent"]:raise ValueError("registered actual orders changed")
    first=list(joint);first[seat]=a;tape=[first]; env.step(first)
    seed=int(hashlib.sha256(f"role-response|{raw['seed']}|{seat}|{t0}".encode()).hexdigest()[:8],16)
    if shared is None:raise ValueError("new arms require a paired measured initialization")
    master._S_COST[:]=shared
    default_cost=list(master._S_COST)
    master._ST.clear();master._ST.update(tr=tr,prev=opv["players"],rng=np.random.RandomState(seed))
    recorder=native_clock.Recorder(master)
    opponent=recorder.wrap(master.agent)
    while not env.done:
        pair=[None,None];pair[seat]=own.agent(live.public(env,seat),cfg);pair[op]=opponent(live.public(env,op),cfg)
        env.step(pair);tape.append(pair)
    statuses=[s.status for s in env.states];scores=[s.reward for s in env.states]
    if statuses!=["DONE","DONE"] or recorder.errors or len(recorder.calls)!=len(tape)-1 or not len(recorder.calls):
        raise ValueError("incomplete original examiner response")
    result=dict(case=case,arm=name,binding=binding,initial_rng=seed,default_warm_cost=default_cost,
                statuses=statuses,score=scores,margin=scores[seat]-scores[op],tape=tape,
                calls=recorder.calls,cash=cash_summary(raw,seat,t0,tape,scores),
                clock=native_clock.summarize(recorder.calls,master._S_ABORT,master._S_D2))
    write(dest,result);return result


def run():
    numeric=load(HERE/"serving.json")
    if sha(HERE/"serving.json")!=P["offline_gate_sha256"] or not numeric["eligible_for_fresh_research_match"]:
        raise ValueError("offline gate missing")
    cases=[select(c) for c in P["exposures"]]
    write(HERE/"selected_roots.json",dict(protocol=sha(PLAN),cases=cases))
    if any(c["t"] is None for c in cases):
        write(HERE/"response_gate.json",dict(protocol=sha(PLAN),eligible=False,reason="no changed action in frozen window"));return
    results=[]
    with LOCK.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for c in cases:
            seat=c["seat"]
            saved=[HERE/f"responses_{c['seed']}_{seat}_{name}.json.gz" for name in ("parent","candidate")]
            path=HERE/f"shared_warm_{c['seed']}_{seat}.json"
            shared=None
            if path.exists():
                entry=load(path)
                if entry["protocol"]!=sha(PLAN) or entry["examiner"]!=P["original"]:raise ValueError("invalid shared warmup identity")
                shared=entry["cost"]
            elif not any(f.exists() for f in saved):
                with ProcessPoolExecutor(1) as pool:shared=pool.submit(native_warm).result()
                write(path,dict(protocol=sha(PLAN),examiner=P["original"],cost=shared))
            with ProcessPoolExecutor(2) as pool:
                old,new=pool.map(arm,[(c,"parent",seat,shared),(c,"candidate",1-seat,shared)])
            paired=old["initial_rng"]==new["initial_rng"] and old["default_warm_cost"]==new["default_warm_cost"]
            diff=[i+c["t"] for i,(x,y) in enumerate(zip(old["tape"],new["tape"])) if x[1-seat]!=y[1-seat]]
            results.append(dict(matched_initial_rng_and_cost=paired,seed=c["seed"],seat=seat,t=c["t"],parent_margin=old["margin"],candidate_margin=new["margin"],
                delta=new["margin"]-old["margin"],response_changed_turns=diff,parent_calls=len(old["calls"]),candidate_calls=len(new["calls"]),
                delta_tree=new["cash"]["tree_income_margin"]-old["cash"]["tree_income_margin"],
                delta_deposits=new["cash"]["delivery_margin"]-old["cash"]["delivery_margin"]))
            print(c["seed"],results[-1],flush=True)
    write(HERE/"response_gate.json",dict(protocol=sha(PLAN),eligible=all(r["matched_initial_rng_and_cost"] and r["delta"]>0 and r["response_changed_turns"] for r in results),results=results))


if __name__=="__main__":run()
