"""Obtain public Apex proposals; certify complete native outcomes independently of any network label."""
import ast
from copy import deepcopy
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

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
ROOT=ACTOR.parents[2]
TREE=ACTOR/"tree_response_training"
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ACTOR.parent/"root_recovery_branches"))
import cnet2_feat as F
from fastsim import FastCarbon
from branches import clean
from cnet2_train import FIELDS
PARENT="9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
NATIVE="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
APEX="f2882881e71b036421c9236b4fa0f1deef1919ed8045822191ed1632323bd54a"


def read(p): return json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())
def hashb(b): return hashlib.sha256(b).hexdigest()
def sha(p):return hashb(p.read_bytes())
def check(ok,why):
    if not ok:raise ValueError(why)
def dump(p,v):
    b=json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    p.write_bytes(gzip.compress(b,mtime=0) if p.suffix==".gz" else b+b"\n")


def identity():
    src=ROOT/"code/apexcarbon_v4tta_search.py";check(sha(src)==APEX,"wrong Apex")
    exam=subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:sources/master3s2/master3s2.py"])
    check(hashb(exam)==NATIVE,"wrong original opponent")
    def embedded(raw):
        return {a.targets[0].id:hashb(a.value.value.encode()) for a in ast.parse(raw).body if isinstance(a,ast.Assign) and
                isinstance(a.targets[0],ast.Name) and a.targets[0].id in ("_WEIGHTS","_VALUE")}
    theirs=embedded(exam);teacher=embedded(src.read_bytes())
    check(teacher==theirs and set(teacher)=={"_WEIGHTS","_VALUE"},"different lineage")
    return {"file":str(src.relative_to(ROOT)),"sha256":APEX,"original_examiner_sha256":NATIVE,
            "identical_embedded_base64_sha256":teacher,"frozen_prereg_sha256":sha(HERE/"protocol.json")}


def registration():
    groups=[TREE,*[TREE/f"batches/intake_{i:02}" for i in range(2,6)],
            *[TREE/f"supplement_01/batches/intake_{i:02}" for i in range(6,9)]]
    reg=[]
    for index,group in enumerate(groups,1):
        files=sorted(group.glob("case_*/training_label.json"))
        check(len(files)==8,"incomplete documented native batch")
        for f in files:
            wd=f.parent;lp=read(wd/"live_protocol.json")
            reg.append({"case":str(wd.relative_to(ROOT)),"split":"train" if index<=6 else "hold",
                        "batch":index,"seed":lp["seed"],"seat":lp["seat"],"t":lp["root_t"],
                        "source":lp["record"]["source"],"source_sha256":lp["record"]["source_sha256"]})
    return reg


def commands(f,x,order):
    y=f.labels(order)
    if clean(f.to_actions(y))!=clean(order):return y,None,"unknown_id_or_order"
    pref=F.Prefix(x,f.aux);uf=np.zeros((F.OWN,F.UF),np.float32);mf=np.zeros((F.OWN,5,F.MF),np.float32);legal=np.zeros((F.OWN,5),bool)
    for s in range(F.OWN):
        if not x["tok_mask"][s]:continue
        uf[s],mf[s],legal[s]=pref.feats(s)
        if not legal[s,y[s]]:return y,None,"illegal_sequential_command"
        pref.advance(s,int(y[s]))
    return y,dict(uf=uf,mf=mf,legal=legal),"legal"


def native(row):
    wd=ROOT/row["case"];root=read(wd/"live_protocol.json")["root"];s=row["seat"];t=row["t"]
    arms={}
    for arm in ("parent","alternative"):
        a=read(wd/("live_"+arm+".json"));p=wd/a["action_tape"];tail=read(p)
        check(a["action_sha256"]==sha(p) and a["source_sha256"]==NATIVE and a["pack_sha256"]==PARENT and
              a["protocol_sha256"]==sha(wd/"live_protocol.json") and a["original_trajectory_sha256"]==row["source_sha256"] and
              a["shared_init_sha256"]==sha(wd/"live_shared_init.json") and a["statuses"]==["DONE","DONE"] and
              a["terminal_t"]==len(tail)+t and a["search_calls"]==len(tail)-1 and
              abs((a["score"][s]-a["score"][1-s])-a["margin"])<=.011,"unbound native response")
        arms[arm]={"own":clean(tail[0][s]),"enemy":clean(tail[0][1-s]),"margin":a["margin"],"score":a["score"],
                   "source_sha256":sha(wd/("live_"+arm+".json")),"tape_sha256":sha(p)}
    p,a=arms["parent"],arms["alternative"];uid=root["worker"]
    check(p["enemy"]==a["enemy"] and {k for k in p["own"].keys()|a["own"].keys() if p["own"].get(k)!=a["own"].get(k)}=={uid} and
          p["own"].get(uid,"NONE")==root["old"] and a["own"].get(uid,"NONE")==root["alternative"],"registered pair mismatch")
    return uid,arms


def main():
    ident=identity();reg=registration()
    spec=importlib.util.spec_from_file_location("frozen_apex_v1_public_proposer",ROOT/ident["file"])
    mod=importlib.util.module_from_spec(spec);start=time.perf_counter();spec.loader.exec_module(mod)
    init=time.perf_counter()-start;default_cost=deepcopy(mod._S_COST)
    os.environ["CARBON_DEBUG"]="1"  # errors must not be silently converted into stationary labels
    cards=[];features=[];exposure={k:set() for k in ("train","hold")}
    from collections import defaultdict
    sources=defaultdict(list)
    for row in reg:sources[row["source"]].append(row)
    for src,items in sources.items():
        path=ROOT/src;raw=read(path)
        check(sha(path)==items[0]["source_sha256"] and raw["pack_sha256"]==PARENT and not raw["errors"],"incorrect development source")
        seed,s=raw["seed"],raw["candidate_seat"];cfg=raw["game_configuration"]
        sim=FastCarbon(cfg).reset(seed=seed); terrain=hashb(bytes(int(c) for c in sim.carbon))
        check(terrain==raw["initial_terrain_u8_sha256"],"initial layout")
        f=F.Feat(s);tracker=mod.BeliefTracker(s);prev=None
        when={v["t"]:v for v in items};check(len(when)==len(items),"duplicate source root")
        for t in range(max(when)+1):
            obs=sim.obs(s);x=f.observe(obs)
            current=when.get(t)
            if current is not None:
                uid,arms=native(current);p,a=arms["parent"],arms["alternative"]
                check(p["own"]==clean(raw["actions"][t][s]) and p["enemy"]==clean(raw["actions"][t][1-s]),"source simultaneous commands")
                salt=int(hashb(("apex-sapling-prior-1|"+current["case"]).encode())[:8],16)
                # Model an observational, teacher-forced episode: only past own executed orders and legally visible world.
                mod._ST.clear();mod._ST.update(tr=deepcopy(tracker),prev=deepcopy(prev),rng=np.random.RandomState(salt))
                mod._S_COST[:]=default_cost
                tm=time.perf_counter();cpu=time.process_time()
                decision=clean(mod.agent(deepcopy(obs),{k:v for k,v in cfg.items() if k!="randomSeed"}))
                elapsed,used=time.perf_counter()-tm,time.process_time()-cpu
                yy,df,leg=commands(f,x,decision);yp,vp,lp=commands(f,x,p["own"]);ya,va,la=commands(f,x,a["own"])
                slot=f.own_uid.index(uid);dmove=decision.get(uid,"NONE")
                field=dict(**current,source_terrain_sha256=terrain,public_before_sha256=hashb(json.dumps(obs,sort_keys=True).encode()),
                    uid=uid,slot=slot,parent_action=p["own"].get(uid,"NONE"),alternative_action=a["own"].get(uid,"NONE"),
                    proposal_action=dmove,full_proposal=decision,proposal_legality=leg,parent_legality=lp,alternative_legality=la,
                    proposal_versus_v1_joint=decision!=p["own"],exact_complete_alternative=decision==a["own"],wall=elapsed,cpu=used,
                    native={k:{z:v[z] for z in ("score","margin","source_sha256","tape_sha256")} for k,v in arms.items()},
                    margin_delta=a["margin"]-p["margin"],rng_seed=salt)
                field["nominates_registered_alternative"]=leg=="legal" and dmove==field["alternative_action"]
                if vp is not None and va is not None and abs(field["margin_delta"])>.01:
                    check(np.array_equal(yp[:slot],ya[:slot]) and np.array_equal(vp["legal"][:slot+1],va["legal"][:slot+1]),"unequal focal prefix")
                    field["feature_index"]=len(features)
                    # y is the independent parent prefix. There is no teacher action target or future truth among input features.
                    item={k:x[k] for k in FIELDS if k in x}
                    item.update(vp,y=yp,carbon=np.zeros(225,np.uint8),teacher=2,gid=len(cards),step=t,value=0.)
                    features.append({k:np.asarray(item[k],dtype=FIELDS[k][0]) for k in FIELDS})
                    field["pair_ids"]=[int(yp[slot]),int(ya[slot])]
                exposure[current["split"]].add((seed,terrain));cards.append(field)
            tracker.step(obs["players"],t,obs["carbon"]);tracker.set_actions(obs["players"],raw["actions"][t][s]);prev=deepcopy(obs["players"])
            sim.step(*raw["actions"][t])
        print(seed,s,len(cards),flush=True)
    for i in (0,1):check({x[i] for x in exposure["train"]}.isdisjoint(x[i] for x in exposure["hold"]),"held map overlap")
    z=HERE/"public_features.npz";np.savez_compressed(z,**{k:np.stack([v[k] for v in features]) for k in features[0]})
    dump(HERE/"proposals.json.gz",{"identity":ident,"initialization_seconds":init,"initial_cost":default_cost,
            "source_sha256":sha(Path(__file__)),"feature_sha256":sha(z),"episodes":reg,"rows":cards})


if __name__=="__main__":main()
