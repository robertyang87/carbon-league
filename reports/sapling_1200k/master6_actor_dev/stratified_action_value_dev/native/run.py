"""Signed stratified native action interventions. Each pair takes the shared arena lock separately."""
import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
ACTOR=DEV.parent
OLD=ACTOR/"late_local_residual_dev"
PRIORITY_FINISH=ACTOR/"early_lane_dev/repeatability/case_621474120_0/finished.json"
PRIORITY_PLAN=PRIORITY_FINISH.parents[1]/"protocol.json"

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

prior=load("frozen_collector_native",OLD/"native/run.py")
account=load("frozen_collector_accounting",OLD/"evidence/account.py")
live,read,sha,check,freeze,dig=prior.live,prior.read,prior.sha,prior.check,prior.freeze,prior.dig
ROOT,TRAIN=prior.ROOT,prior.TRAIN

def load_inputs():
    path=DEV/"native_input.json"; inp=read(path)
    datafile=DEV/inp["cards"]
    check(sha(datafile)==inp["cards_sha256"] and inp["parent_pack_sha256"]==sha(prior.PARENT),"card or frozen parent changed")
    cards=read(datafile)
    cards=cards if isinstance(cards,list) else cards["cases"]
    check(cards and len(set(c["id"] for c in cards))==len(cards),"missing or duplicate card identity")
    check(all(re.fullmatch(r"[A-Za-z0-9_.-]+",c["id"]) and c["id"] not in (".","..") for c in cards),"unsafe case name")
    split_maps={split:{c["map"] for c in cards if c["split"]==split} for split in ("train","held")}
    check(all(c["split"] in split_maps for c in cards) and not (split_maps["train"] & split_maps["held"]),"train/held map overlap")
    check(isinstance(inp["max_followup_turns"],int) and 0<=inp["max_followup_turns"]<=298 and inp["wall_seconds"]>0,"invalid quota")
    if inp.get("protocol_sha256"):
        check(sha(DEV/"protocol.json")==inp["protocol_sha256"],"study protocol changed")
    register=read(DEV/"source/inventory.json")
    source_by_sha={r["raw_sha256"]:r for r in register["games"]}
    for c in cards:
        r=source_by_sha[c["raw_sha256"]]
        check(all(r[k]==c[k] for k in ("split","seed","seat","raw","map","official_sha256")) and
              c["t"] not in r["excluded_exact_physical_steps"],"prior or unsigned root, refused duplicate response")
    ids={c["id"]:c for c in cards}
    for c in cards:
        if c.get("diagnostic"):
            original=ids[c["primary_id"]]
            check(c["split"]=="train" and c.get("stream")==1 and not original.get("diagnostic") and
                  all(c[k]==original[k] for k in ("raw_sha256","map","seat","t","base","alt","root_enemy","proposal")),
                  "diagnostic pair differs by more than stream")
        else: check(c.get("stream",0)==0,"unexpected stream")
    return inp,cards

def prepare():
    inp,cards=load_inputs()
    root_source=prior.SOURCE
    check(sha(root_source)==live.MASTER_SHA and __import__("hashlib").sha256(subprocess.check_output(
        ["git","show","HEAD:sources/master3s2/master3s2.py"],cwd=ROOT)).hexdigest()==live.MASTER_SHA,"original examiner source mismatch")
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),HERE/"labels.py",OLD/"native/run.py",OLD/"evidence/account.py",
        ACTOR/"lot_settlement_dev/lots.py",ACTOR/"late_advantage_actor_dev/lots.py",prior.PARENT,root_source,prior.SHARED,
        DEV/"core.py",DEV/"budget.py",DEV/"policy.py",OLD/"deploy.py",DEV/"source/inventory.json",DEV/"support.json",DEV/"protocol.json",Path(live.__file__),prior.BASE/"root_recovery_branches/branches.py",ROOT/"code/fastsim.py",DEV/"native_input.json",DEV/inp["cards"])}
    receipt={"inputs":hashes,"registered_cards":len(cards),"rng_rule":"sha256(canonical-json([raw_sha256, t, id, stratified_action_value_crn_1]))[:8]",
             "original_configuration":read(prior.SHARED)["original"],"warm_cost":read(prior.SHARED)["warm_cost"]}
    freeze(HERE/"frozen.json",receipt)
    return receipt

def source(card):
    raw=ROOT/card["raw"]
    cert=ROOT/card["official"] if card.get("official") else raw.parent/"official_replay.json"
    check(sha(raw)==card["raw_sha256"] and sha(cert)==card["official_sha256"],"unsigned source game")
    tape=read(raw); doc=read(cert)
    for rel,h in doc["engine_source_sha256"].items():
        check(h==sha(ROOT/"code/env_src/carbon_challenge-master"/rel),"official engine mismatch")
    matches=[g for v in doc["runs"].values() if v["pack_sha256"]==sha(prior.PARENT) for g in v["games"]
             if (g["seed"],g["seat"],g["transcript_sha256"])==(card["seed"],card["seat"],card["raw_sha256"])]
    g,=matches
    check(tape["pack_sha256"]==sha(prior.PARENT) and tape["examiner_md5"]=="57f9a594cce4f2ec9e3573d8d9d26ca1" and not tape["audit_messages"] and
          tape["candidate_seat"]==card["seat"] and tape["seed"]==card["seed"] and len(tape["actions"])==g["steps"] and
          tape["initial_terrain_u8_sha256"]==card["map"]==g["map_sha256"] and
          tape["official_score_by_seat"]==g["official_reward_by_seat"] and g["statuses"]==["DONE","DONE"],"source identity or official outcome mismatch")
    return tape

def edit_validation(obs,cfg,own,ctx,card):
    s=obs["player"]; base,alt=card["base"],card["alt"]
    listed=card.get("edits") or [card["proposal"]]
    changed={u for u in set(base)|set(alt) if base.get(u,"NONE")!=alt.get(u,"NONE")}
    check(changed=={p["uid"] for p in listed} and len(listed)==len(changed)>0,"wrong edit set")
    check(all(p["original"]==base.get(p["uid"],"NONE") and p["action"]==alt.get(p["uid"],"NONE") for p in listed),"edit doesn't explain joint")
    actor=own._FEAT; f=sys.modules["cnet2_feat"]; x,aux=ctx
    for order in (base,alt):
        labels=actor.labels(order)
        check(actor.to_actions(labels)==order,"redecoded joint differs")
        prefix=f.Prefix(x,aux)
        for i in np.flatnonzero(x["tok_mask"][:f.OWN]):
            _,_,allowed=prefix.feats(i); sel=int(labels[i])
            check(0<=sel<len(allowed) and allowed[sel],"illegal action under real joint prefix")
            prefix.advance(i,sel)
    check(not (own._return_conflicts(x,actor.labels(alt)) - own._return_conflicts(x,actor.labels(base))),"new own collision or blocked recruit")
    p=obs["players"][s]
    check(not(set(alt)-set(p[1])-set(p[2])),"unknown own actor")
    roster=[w for tm in obs["players"] for w in tm[2].values()]
    nc=sum(w[2]=="COLLECTOR" for w in roster); count=len(p[2]);cash=p[0]
    cost={"RECCOLLECTOR":cfg["recCollectorCost"]+cfg["recCollectorIncreaseCost"]*nc,
          "RECPLANTER":cfg["recPlanterCost"]+cfg["recPlanterIncreaseCost"]*(len(roster)-nc)}
    for bid in p[1]:
        if bid not in alt:continue
        a=alt[bid];check(a in cost and cash>=cost[a] and count<cfg["workerLimit"],"unfunded recruit")
        count+=1; cash-=cost[a]

def root_state(c,master=None):
    tape=source(c); s=c["seat"]; e=1-s;t0=c["t"];cfg=tape["game_configuration"]
    own,ctx,_=prior.fresh_actor(); env=live.begin(tape)
    check(0<=t0<len(tape["actions"]),"source terminal or absent step")
    tr=master.BeliefTracker(master._Bel,e) if master else None
    for t,j in enumerate(tape["actions"][:t0+1]):
        us=live.public(env,s);op=live.public(env,e)
        if tr:
            tr.step(op["players"],t,op["carbon"]);tr.set_actions(op["players"],j[e])
        if t in (0,t0):
            check(own.agent(us,cfg)==j[s],"recorded parent joint differs after same history")
        else:own._FEAT.observe(us)
        if t<t0:env.step(j)
    check(j[s]==c["base"] and j[e]==c["root_enemy"],"simultaneous root differs")
    edit_validation(us,cfg,own,ctx,c)
    for k,v in (("own_public_sha256",us),("op_public_sha256",op),("root_own_public_sha256",us),("root_public_enemy_sha256",op)):
        if k in c:check(c[k]==dig(v),"public view differs")
    return env,own,tr,us,op,tape

def random_seed(c):
    return int(dig([c["raw_sha256"],c["t"],c["id"],"stratified_action_value_crn_1"])[:8],16)

def rng_state(rng):
    algo,keys,idx,norm,stored=rng.get_state()
    return [algo,keys.tolist(),idx,norm,stored]

def frozen_hash():
    return sha(HERE/"frozen.json")

def run_arm(c,arm,d,deadline,limit):
    out=d/f"{arm}.json.gz"; chk=d/f"{arm}.partial.json.gz"; issue=d/f"{arm}.issued.json"
    binding={"id":c["id"],"card_sha256":dig(c),"arm":arm,"frozen_sha256":frozen_hash(),"paired_seed":random_seed(c)}
    if out.exists():
        complete=read(out);check(complete["binding"]==binding and not issue.exists(),"invalid completed arm")
        return True
    master=live.module(prior.SOURCE,"fresh_native_master"); auth=read(prior.SHARED)
    check(master._TEMP==1. and master._SEED is None and master._SEARCH_OK and master._RX and master._S_ABORT==.58,"opponent settings")
    env,own,tr,us,op,tape=root_state(c,master)
    cfg=tape["game_configuration"];s=c["seat"];e=1-s;T=c["t"]
    check(cfg["episodeSteps"]-2-T<=limit,"registered tail exceeds call limit")
    rng=np.random.RandomState(random_seed(c))
    master._S_COST[:]=auth["warm_cost"];master._ST.clear();master._ST.update(tr=tr,prev=op["players"],rng=rng)
    j0=tape["actions"][T].copy();j0[s]=c[arm]
    env.step(j0); history=[]; recovered=[]; calls=0
    real=master._search
    def counted(*a,**kw):
        nonlocal calls
        calls+=1
        return real(*a,**kw)
    master._search=counted
    if chk.exists():
        z=read(chk);check(z["binding"]==binding and z["joint0"]==j0,"stale restart record")
        history=z["history"];recovered=z["abandoned_attempts"]
        for r in history:
            t=env.states[0].observation.step; obs=live.public(env,s);v=live.public(env,e)
            check(not env.done and r["t"]==t and r["op_public_sha256"]==dig(v) and r["own_public_sha256"]==dig(obs) and
                  r["joint"][s]==own.agent(obs,cfg),"restart replay diverges")
            tr.step(v["players"],t,v["carbon"]);tr.set_actions(v["players"],r["joint"][e])
            master._ST["prev"]=v["players"];env.step(r["joint"])
        prior.resume_rng(rng,z["rng_after"]);master._S_COST[:]=z["cost_after"]
    if issue.exists():
        ticket=read(issue)
        if history and history[-1]["issue"]==ticket:issue.unlink()
        else:
            check(ticket["binding"]==binding and ticket["t"]==env.states[0].observation.step and
                  ticket not in recovered and len([r for r in recovered if r["t"]==ticket["t"]])<1,"ambiguous repeated unfinished call")
            recovered.append(ticket);issue.unlink()
    while not env.done:
        if time.monotonic()+1.1>=deadline:
            live.save(chk,{"binding":binding,"joint0":j0,"history":history,"abandoned_attempts":recovered,
                           "rng_after":rng_state(rng),"cost_after":list(master._S_COST)})
            return False
        t=env.states[0].observation.step;v=live.public(env,e);obs=live.public(env,s)
        check(T<t<cfg["episodeSteps"]-1 and len(history)<limit,"native call quota")
        me=own.agent(obs,cfg); before=calls; ticket={"binding":binding,"t":t,"pid":os.getpid(),"issued_ns":time.time_ns()}
        with issue.open("x") as f:json.dump(ticket,f)
        start=time.monotonic();other=master.agent(v,cfg);wall=time.monotonic()-start
        check(calls==before+1,"no original search")
        joint=[None,None];joint[s]=me;joint[e]=other
        for p in (0,1):
            assets=obs["players"][p]
            check(not(set(joint[p])-set(assets[1])-set(assets[2])),"unknown worker or base")
        env.step(joint)
        history.append({"t":t,"joint":joint,"own_public_sha256":dig(obs),"op_public_sha256":dig(v),"wall_seconds":wall,"issue":ticket})
        live.save(chk,{"binding":binding,"joint0":j0,"history":history,"rng_after":rng_state(rng),"cost_after":list(master._S_COST),
                       "abandoned_attempts":recovered})
        issue.unlink()
    scores=[a.reward for a in env.states]
    check([a.status for a in env.states]==["DONE","DONE"] and env.states[0].observation.step==T+len(history)+1,"wrong official termination")
    freeze(out,{"binding":binding,"joint0":j0,"history":history,"abandoned_attempts":recovered,
        "initial_public":[dig(us),dig(op)],"original_config":auth["original"],"warm_cost":auth["warm_cost"],
        "final_cost":list(master._S_COST),"official_scores":scores,"official_margin":scores[s]-scores[e],"U":live.utility(scores,s),
        "terminal_step":env.states[0].observation.step,"statuses":[p.status for p in env.states],
        "cash":[p[0] for p in env.states[0].observation.players],"completed_original_calls":len(history)})
    chk.unlink(missing_ok=True)
    return True

def choose(inp,cards,split):
    declared=[c for c in cards if c["split"]==split]
    if split=="train":
        return declared,None
    p=DEV/"held_selection.json";h=read(p);check(h["weights_sha256"]==sha(DEV/"weights.json"),"held weights changed")
    declared_by_id={c["id"]:c for c in declared};active=h["active_ids"]
    check(len(set(active))==len(active) and set(active)<=set(declared_by_id),"invalid held selection")
    check(h["cards_sha256"]==inp["cards_sha256"],"held cards changed")
    picker=load("frozen_action_residual_picker",DEV/"policy.py")
    weights=read(DEV/"weights.json"); groups={}
    for c in declared:
        groups.setdefault((c["map"],c["seat"]),{}).setdefault(c["t"],[]).append(c)
    expected=[]
    for terrain in groups.values():
        for _,options in sorted(terrain.items()):
            options=sorted(options,key=lambda c:c["priority_rank"])
            proposal=picker.choose([c["proposal"] for c in options],weights)
            if proposal is not None:
                expected.append(next(c["id"] for c in options if proposal==c["proposal"]))
                break
    check(set(active)==set(expected),"held arm is not first real policy intervention")
    pairs=[(declared_by_id[i]["map"],declared_by_id[i]["seat"]) for i in active]
    check(len(pairs)==len(set(pairs)),"two selected roots in one held map/seat")
    freeze(HERE/"held_freeze.json",{"selection_sha256":sha(p),"weights_sha256":h["weights_sha256"],"frozen_sha256":frozen_hash()})
    return [declared_by_id[i] for i in active],sha(p)

def competing():
    for p in Path("/proc").glob("[0-9]*"):
        with contextlib.suppress(OSError):
            args=(p/"cmdline").read_bytes().split(b"\0")
            if args and b"python" in os.path.basename(args[0]) and any(os.path.basename(v)==b"tree_pack_arena.py" for v in args[1:]):
                return True
    return False

def schedule(split):
    inp,cards=load_inputs();check((HERE/"frozen.json").is_file(),"call prepare only after authorization")
    prepare()
    check(read(DEV/"support.json")["eligible"],"pre-registered support gate failed")
    check(os.sched_getaffinity(0)=={int(inp.get("cpu",1))} and os.environ.get("CARBON_DEBUG")=="1" and
          all(os.environ.get(k)=="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")),"native CPU, debug or BLAS environment differs")
    active,held_sig=choose(inp,cards,split)
    remaining=float(os.environ["STRATIFIED_REMAINING_SECONDS"])
    check(np.isfinite(remaining) and remaining>=0,"invalid persisted remaining wall allowance")
    if remaining<2:return 75
    sys.path.insert(0,str(DEV))
    budget=load("stratified_active_budget",DEV/"budget.py")
    deadline=time.monotonic()+min(float(inp["wall_seconds"]),remaining)
    for c in active:
        d=HERE/c["id"];d.mkdir(exist_ok=True)
        if (d/"pair.json").exists():
            r=read(d/"pair.json")
            check(r["files"]=={a:sha(d/f"{a}.json.gz") for a in ("base","alt")} and r["card_sha256"]==dig(c),"changed finished pair")
            continue
        if (PRIORITY_PLAN.exists() and not PRIORITY_FINISH.exists()) or competing():
            print("other arena first",flush=True);return 75
        with (TRAIN/".continuation.lock").open("a") as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print("native arena busy",flush=True);return 75
            if competing() or (PRIORITY_PLAN.exists() and not PRIORITY_FINISH.exists()):return 75
            if time.monotonic()+2>=deadline:return 75
            with budget.reserve(inp["wall_seconds"]) as active_seconds:
                if active_seconds<2:return 75
                pair_deadline=min(deadline,time.monotonic()+active_seconds)
                for a in ("base","alt"):
                    if not run_arm(c,a,d,pair_deadline,inp["max_followup_turns"]):return 75
                freeze(d/"pair.json",{"id":c["id"],"card_sha256":dig(c),"frozen_sha256":frozen_hash(),
                     "held_selection_sha256":held_sig,"files":{a:sha(d/f"{a}.json.gz") for a in ("base","alt")}})
        print("completed paired native root",c["id"],flush=True)
        time.sleep(.2)
    freeze(HERE/f"{split}_complete.json",{"split":split,"active":[c["id"] for c in active],"frozen_sha256":frozen_hash(),"held_selection_sha256":held_sig})
    return 0

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("mode",choices=("prepare","train","held","labels"));cli.add_argument("split",nargs="?",choices=("train","held"))
    opt=cli.parse_args()
    if opt.mode=="prepare":prepare()
    elif opt.mode=="labels":
        check(opt.split is not None,"labels needs split")
        load("stratified_labels",HERE/"labels.py").main(sys.modules[__name__],opt.split)
    else:sys.exit(schedule(opt.mode))
