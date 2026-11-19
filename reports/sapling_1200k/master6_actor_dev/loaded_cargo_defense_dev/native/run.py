"""Custody of the loaded-cargo same-root native court; shared frozen original-response implementation."""
import argparse
import hashlib
from collections import defaultdict
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import time

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
ACTOR=DEV.parent
PREVIOUS=ACTOR/"stratified_action_value_dev/native"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod);return mod
prior=load("shared_stratified_native",PREVIOUS/"run.py")
# Each CLI process owns this imported module instance. All writes of its generic arm worker now target this experiment.
prior.HERE=HERE
prior.DEV=DEV
read,sha,dig,freeze,check,live=prior.read,prior.sha,prior.dig,prior.freeze,prior.check,prior.live
ROOT=prior.ROOT
budget=load("defense_native_budget",HERE/"budget.py")
def frozen_hash():return sha(HERE/"frozen.json")
def paired_seed(card):
    return int(dig(["official_cash_defense_crn_v1",card["raw_sha256"],card["t"],card.get("primary_id") or card["id"],card.get("stream",0)])[:8],16)
prior.random_seed=paired_seed

def panel():
    inp=read(DEV/"native_input.json");path=DEV/inp["cards"]
    check(sha(path)==inp["cards_sha256"] and inp["parent_pack_sha256"]==sha(prior.prior.PARENT),"panel/parent mismatch")
    if inp.get("protocol_sha256"):check(sha(DEV/"protocol.json")==inp["protocol_sha256"],"unregistered panel")
    cards=read(path);cards=cards if isinstance(cards,list) else cards["cases"]
    check(0<inp["wall_seconds"]<=2100 and 0<=inp["max_followup_turns"]<=298,"invalid registered budget")
    names={c["id"]:c for c in cards}
    check(len(names)==len(cards) and all(re.fullmatch(r"[a-zA-Z0-9_.-]+",x) and x not in (".","..") for x in names),"defense card identity")
    maps={split:{c["map"] for c in cards if c["split"]==split} for split in ("train","held")}
    check(not (maps["train"]&maps["held"]),"map contamination")
    for c in cards:
        check(c["split"] in maps and c["seat"] in (0,1) and c["base"]!=c["alt"],"invalid card")
        if c.get("diagnostic") or c.get("stream",0):
            p=names[c["primary_id"]]
            check(c["split"]=="train" and c.get("diagnostic") and c["stream"]>0 and not p.get("stream",0) and
                  all(c[k]==p[k] for k in ("raw_sha256","map","seat","t","base","alt","root_enemy")),"diagnostic depends on changes other than random stream")
    return inp,cards

def signature(raw_sha,seat,t,base,alt,enemy):
    return dig([raw_sha,seat,t,base,alt,enemy])

def archived():
    index=defaultdict(list)
    def add(k,kind,folder,files):
        index[k].append({"kind":kind,"folder":str(folder.relative_to(ROOT)),
                         "hashes":{p.name:sha(p) for p in files}})
    # Newer signed stratified original response records, including the two planned repeat streams.
    d=PREVIOUS;cpath=d.parent/"cards.json"
    if cpath.exists():
        old_seal=read(d/"frozen.json");old_seal_sha=sha(d/"frozen.json")
        check(old_seal["inputs"][str(prior.prior.SOURCE.relative_to(ROOT))]==live.MASTER_SHA and
              old_seal["inputs"][str(prior.prior.PARENT.relative_to(ROOT))]==sha(prior.prior.PARENT),"old stratified opponent or policy changed")
        for c in read(cpath)["cases"]:
            f=d/c["id"]
            if not (f/"pair.json").exists():continue
            pair=read(f/"pair.json")
            check(pair["frozen_sha256"]==old_seal_sha and pair["card_sha256"]==dig(c) and all(pair["files"][a]==sha(f/f"{a}.json.gz") for a in ("base","alt")),"stale stratified archive")
            add(signature(c["raw_sha256"],c["seat"],c["t"],c["base"],c["alt"],c["root_enemy"]),"stratified",f,
                [f/"pair.json",f/"base.json.gz",f/"alt.json.gz"])
    d=ACTOR/"late_local_residual_dev/native"
    for p in d.glob("selection_*.json"):
        z=read(p);r=z["root"];s=z["source"]
        if not r:continue
        f=d/f'{s["split"]}_{s["seed"]}_{s["seat"]}'
        if not (f/"alt_account.json.gz").exists():continue
        check(z["identities"]["original"]==live.MASTER_SHA and z["identities"]["parent"]==sha(prior.prior.PARENT),"wrong old late actor or examiner")
        ob,oa=(read(f/f"{k}.json.gz") for k in ("base","alt"))
        check(ob["rng_seed"]==oa["rng_seed"]==r["paired_seed"] and ob["root_digest"]==oa["root_digest"]==dig(r)
              and ob["original_config"]==oa["original_config"]==read(prior.prior.SHARED)["original"],"late pair used different root or responder")
        add(signature(s["raw_sha256"],s["seat"],r["t"],r["base"],r["alt"],r["root_enemy"]),"late_local",f,
            [f/"base.json.gz",f/"alt.json.gz",f/"base_account.json.gz",f/"alt_account.json.gz"])
    # Both ordinary and repeated tree/early-lane records are discoverable by signed paired root/physical joints.
    for pp in ACTOR.rglob("live_protocol.json"):
        f=pp.parent; a,b=(f/f"live_{n}.json" for n in ("parent","alternative"))
        if not a.exists() or not b.exists():continue
        m,n=map(read,(a,b));s=m["seat"]
        if m["pack_sha256"]!=sha(prior.prior.PARENT) or m["pack_sha256"]!=n["pack_sha256"]:continue
        if (m["source_sha256"]!=live.MASTER_SHA or m["source_sha256"]!=n["source_sha256"] or m["original_trajectory_sha256"]!=n["original_trajectory_sha256"] or
              m["shared_init_sha256"]!=n["shared_init_sha256"] or m["statuses"]!=n["statuses"] or m["statuses"]!=["DONE","DONE"]):continue
        j,k=m["root_actions"],n["root_actions"]
        check(j[1-s]==k[1-s] and m["root"]["t"]==n["root"]["t"] and m["protocol_sha256"]==n["protocol_sha256"]==sha(pp),"invalid shared source root")
        add(signature(m["original_trajectory_sha256"],s,m["root"]["t"],j[s],k[s],j[1-s]),"legacy_live",f,
            [pp,a,b,f/m["action_tape"],f/n["action_tape"],f/"live_shared_init.json"])
    return index

def prepare():
    inp,cards=panel()
    inputs=[Path(__file__),HERE/"budget.py",HERE/"labels.py",PREVIOUS/"run.py",PREVIOUS/"labels.py",prior.OLD/"native/run.py",
            prior.OLD/"evidence/account.py",prior.OLD/"deploy.py",ACTOR/"lot_settlement_dev/lots.py",ACTOR/"late_advantage_actor_dev/lots.py",
            prior.prior.PARENT,prior.prior.SOURCE,prior.prior.SHARED,DEV/"native_input.json",DEV/inp["cards"],Path(live.__file__),
            prior.prior.BASE/"root_recovery_branches/branches.py",ROOT/"code/fastsim.py"]
    for path in ("protocol.json","package.json","candidate.py","coverage.json","source/inventory.json","all_opportunities.json.gz"):
        if (DEV/path).exists():inputs.append(DEV/path)
    canonical=hashlib.sha256(subprocess.check_output(["git","show","HEAD:sources/master3s2/master3s2.py"],cwd=ROOT)).hexdigest()
    check(sha(prior.prior.SOURCE)==live.MASTER_SHA==canonical,"wrong full original")
    if not (HERE/"cache_match.json").is_file():
        origins=archived()
        matches={}
        for c in cards:
            k=signature(c["raw_sha256"],c["seat"],c["t"],c["base"],c["alt"],c["root_enemy"])
            hits=sorted(origins.get(k,[]),key=lambda d:d["folder"])
            if hits:
                matches[c["id"]]=hits
        freeze(HERE/"cache_match.json",{"cards_sha256":inp["cards_sha256"],"signed_prior_pairs":sum(map(len,origins.values())),
               "matches":matches,"selection":"First prior verified exact pair by artifact path for train primary only, registered repeats are fresh. Previously exposed exact held pair is ineligible."})
    reuse=read(HERE/"cache_match.json")
    check(reuse["cards_sha256"]==inp["cards_sha256"],"stale cache index")
    for c in cards:
        check(c["split"]!="held" or c["id"] not in reuse["matches"],"previously exposed exact pair included as held")
    inputs.append(HERE/"cache_match.json")
    freeze(HERE/"frozen.json",{"files":{str(p.relative_to(ROOT)):sha(p) for p in inputs},"count":len(cards),
          "rng_rule":"hash(raw tape, turn, primary card, independently numbered stream); previously certified training pair may be reused with its actual archived stream"})

def selected(inp,cards,split):
    rows=[c for c in cards if c["split"]==split]
    if split=="train":return rows,None
    path=DEV/"held_selection.json";h=read(path)
    identities={}
    if "candidate_sha256" in h:
        identities["candidate_sha256"]=sha(DEV/"candidate.py")
        check(identities["candidate_sha256"]==h["candidate_sha256"]==read(DEV/"package.json")["sha256"],"held active pack changed")
    if "weights_sha256" in h:
        identities["weights_sha256"]=sha(DEV/"weights.json")
        check(h["weights_sha256"]==identities["weights_sha256"],"held separate weights changed")
    check(identities and h["cards_sha256"]==inp["cards_sha256"],"held strategy identity or cards changed")
    by_id={c["id"]:c for c in rows};ids=h["active_ids"]
    check(len(set(ids))==len(ids) and set(ids)<=by_id.keys(),"unknown or duplicated deployed root")
    declared=read(DEV/"coverage.json")["all_held_seats"]
    def record(x):return (x["seed"],x["seat"],x["map"],x.get("first_changed_card"),bool(x.get("excluded_first")))
    check(sorted(map(record,declared))==sorted(map(record,h["all_registered"])),"held registered seats disagree with public first real dispatch")
    signed={(r["seed"],r["seat"]):None if r.get("excluded_first") else r.get("first_changed_card") for r in h["all_registered"]}
    check(len(signed)==len(h["all_registered"]) and {(by_id[q]["seed"],by_id[q]["seat"]):q for q in ids}==
          {k:v for k,v in signed.items() if v is not None},"held roots differ from first deployed changes")
    check(len({(by_id[q]["map"],by_id[q]["seat"]) for q in ids})==len(ids),"multiple choices for the same held map/seat")
    if "triggered_ids" in h:
        first={}
        for i in sorted(h["triggered_ids"],key=lambda k:(by_id[k]["t"],by_id[k].get("priority_rank",0),k)):
            c=by_id[i];first.setdefault((c["seed"],c["seat"]),c["id"])
        check(first=={k:v for k,v in signed.items() if v is not None},"an earlier eligible deployment would act first")
    freeze(HERE/"held_freeze.json",{"held_selection_sha256":sha(path),**identities,"frozen_sha256":frozen_hash()})
    return [by_id[i] for i in ids],sha(path)

def related_job(inp):
    if prior.competing():return True
    # Other tracks can freeze a conditional claim with a planned path and its signed end marker.
    return any((ROOT/p["start"]).exists() and not (ROOT/p["finish"]).is_file() for p in inp.get("priority_queues",[]))

def shared_arm(c,a,d,deadline,limit):
    return prior.run_arm(c,a,d,deadline,limit)

def materialize(c,a,d,old):
    f=ROOT/old["folder"]
    check(all(sha(f/k)==v for k,v in old["hashes"].items()),"cached arm changed since registration")
    if old["kind"]=="legacy_live":
        key="parent" if a=="base" else "alternative";z=read(f/f"live_{key}.json")
        records=read(f/z["action_tape"]);scores=z["score"];statuses=z["statuses"];term=z["terminal_t"]
        stream_origin=z["shared_init_sha256"];n=z["search_calls"]
        check(sha(f/z["action_tape"])==z["action_sha256"] and z["original_trajectory_sha256"]==c["raw_sha256"],"unverified old joint tape")
    else:
        z=read(f/f"{a}.json.gz")
        if old["kind"]=="stratified":
            records=[z["joint0"]]+[r["joint"] for r in z["history"]];statuses=z["statuses"];term=z["terminal_step"]
            stream_origin=z["binding"]["paired_seed"];n=z["completed_original_calls"]
        else:
            records=z["orders"];statuses=z["final_status"];term=z["final_step"];stream_origin=z["rng_seed"];n=z["original_calls"]
            check(z["source"]["raw_sha256"]==c["raw_sha256"],"different parent replay")
        scores=z["official_scores"]
    env,own,_,us,op,tape=prior.root_state(c);s=c["seat"];e=1-s;cfg=tape["game_configuration"]
    check(records[0][s]==c[a] and records[0][e]==c["root_enemy"] and len(records)-1==n,"cached root or search count")
    env.step(records[0]);history=[]
    for j in records[1:]:
        x=live.public(env,s);y=live.public(env,e)
        check(not env.done and own.agent(x,cfg)==j[s],"old continuing policy differs")
        history.append({"t":x["step"],"joint":j,"own_public_sha256":dig(x),"op_public_sha256":dig(y),
                        "wall_seconds":None,"original":"verified historical response"})
        env.step(j)
    check(env.done and [p.status for p in env.states]==statuses==["DONE","DONE"] and scores==[p.reward for p in env.states]
          and term==env.states[0].observation.step,"old formal score not reproducible")
    binding={"id":c["id"],"card_sha256":dig(c),"arm":a,"frozen_sha256":frozen_hash(),"paired_seed":stream_origin}
    freeze(d/f"{a}.json.gz",{"binding":binding,"joint0":records[0],"history":history,"abandoned_attempts":[],
           "initial_public":[dig(us),dig(op)],"official_scores":scores,"official_margin":scores[s]-scores[e],"U":live.utility(scores,s),
           "cash":[p[0] for p in env.states[0].observation.players],"terminal_step":term,"statuses":statuses,"completed_original_calls":n,
           "new_original_calls":0,"archived_source":old})

def schedule(split):
    inp,cards=panel();check((HERE/"frozen.json").exists(),"entry requires a signed native freeze");prepare()
    check(os.sched_getaffinity(0)=={1} and os.environ.get("CARBON_DEBUG")=="1" and all(os.environ.get(k)=="1" for k in
          ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")),"one native core, BLAS and debug are required")
    pending,sel=selected(inp,cards,split)
    cap=inp["wall_seconds"];hint=float(os.environ.get("STRATIFIED_REMAINING_SECONDS",str(cap)))
    check(hint>=0 and __import__("math").isfinite(hint),"invalid external wall budget")
    deadline=time.monotonic()+min(cap,hint)
    cache=read(HERE/"cache_match.json")["matches"]
    for c in pending:
        d=HERE/c["id"];d.mkdir(exist_ok=True);pair=d/"pair.json"
        if pair.exists():
            z=read(pair);check(z["frozen_sha256"]==frozen_hash() and z["card_sha256"]==dig(c) and z["files"]=={a:sha(d/f"{a}.json.gz") for a in ("base","alt")},"stale finished pair")
            continue
        previously=cache.get(c["id"]) if not c.get("diagnostic") and split=="train" else None
        if previously:
            for a in ("base","alt"):materialize(c,a,d,previously[0])
        else:
            if related_job(inp):return 75
            with (prior.TRAIN/".continuation.lock").open("a") as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return 75
                if related_job(inp) or time.monotonic()+2>=deadline:return 75
                with budget.reserve(cap) as remain:
                    if remain<2:return 75
                    end=min(deadline,time.monotonic()+remain)
                    for a in ("base","alt"):
                        if not shared_arm(c,a,d,end,inp["max_followup_turns"]):return 75
        freeze(pair,{"id":c["id"],"card_sha256":dig(c),"frozen_sha256":frozen_hash(),"held_selection_sha256":sel,"cache":bool(previously),
               "files":{a:sha(d/f"{a}.json.gz") for a in ("base","alt")}})
        print("certified native pair",c["id"],"cache" if previously else "new",flush=True)
        time.sleep(.2)
    freeze(HERE/f"{split}_complete.json",{"active":[r["id"] for r in pending],"frozen_sha256":frozen_hash(),"held_selection_sha256":sel})
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("prepare","train","held","labels"));ap.add_argument("split",nargs="?",choices=("train","held"));args=ap.parse_args()
    if args.mode=="prepare":prepare()
    elif args.mode=="labels":load("loaded_defense_lot_labels",HERE/"labels.py").main(sys.modules[__name__],args.split)
    else:sys.exit(schedule(args.mode))
