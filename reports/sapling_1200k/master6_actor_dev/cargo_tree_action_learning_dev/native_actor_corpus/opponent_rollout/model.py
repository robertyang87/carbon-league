"""Small action scorer shared by cache construction, fitting and sampled free decoding."""
from collections import defaultdict
import os
from pathlib import Path
import sys
import time

import numpy as np
from register import ACTOR, CARGO, HERE, ROOT, SEED, load, rel, save, sha
sys.path.insert(0,str(ACTOR/"own_search_bridge_dev"))
import bridge
F=bridge.F
ROLE=("collector","planter","base")
CONTRACT=HERE/"protocol.json"
SOURCES=HERE/"sources.json.gz"
REFERENCE=ACTOR/"own_search_bridge_dev/opponent_response_dev"
BASELINES={"published":REFERENCE/"opponent.npz","calibrated":REFERENCE/"active_calibration/opponent.npz"}

def validate():
    plan=load(CONTRACT)
    for p,value in plan["input_sha256"].items():
        if sha(ROOT/p)!=value:raise ValueError("input drift: "+p)
    if os.sched_getaffinity(0)!={3}:raise ValueError("run on CPU 3")
    return plan

def ticks_for(group):
    by_key=defaultdict(list)
    for card in group:by_key[(card["physical_sha256"],card["expert_seat"])].append(card)
    result={}
    for cards in by_key.values():
        ordered=sorted(cards,key=lambda c:c["raw_sha256"])
        for i,c in enumerate(ordered):
            n=64//len(cards)+int(i<64%len(cards))
            if n>c["transitions"]:raise ValueError("fewer unique turns than planned")
            rng=np.random.default_rng(int(c["raw_sha256"][:16],16)^SEED)
            result[c["raw_sha256"]]=[int(rng.choice(bucket)) for bucket in np.array_split(np.arange(c["transitions"]),n)]
    return result

def frames(card,ticks):
    path=ROOT/card["path"]
    if sha(path)!=card["raw_sha256"]:raise ValueError("source tape changed")
    r=load(path);seat=card["expert_seat"]
    assert seat==1-r["candidate_seat"] and len(r["actions"])==card["transitions"]
    sim=bridge.FastCarbon(r["game_configuration"]).reset(seed=r["seed"])
    feat=F.Feat(seat);wanted=set(ticks)
    for t,pair in enumerate(r["actions"]):
        obs=sim.obs(seat)
        # Full world is retained inside the engine, and never passed to this feature encoder.
        assert "full_carbon" not in obs and obs["player"]==seat and obs["step"]==t
        x=feat.observe(obs)
        if t in wanted:yield t,obs,feat,x,pair[seat]
        sim.step(*pair)
    if list(sim.cash)!=r["last_post_cash"]:raise ValueError("reference terminal cash differs")

def compact_fit_card(card,ticks):
    mats=[];masks=[];ys=[];roles=[];stats=defaultdict(int)
    for t,obs,f,x,commands in frames(card,ticks):
        stats["roots"]+=1
        targets=f.labels(commands)
        if f.to_actions(targets)!={i:a for i,a in commands.items() if a and a!="NONE"}:
            raise ValueError("target does not round-trip to physical commands")
        prefix=F.Prefix(x,f.aux);slots=np.flatnonzero(x["tok_mask"][:F.OWN])
        for i,s in enumerate(slots):
            mat,legal=bridge.features(x,prefix,s);a=int(targets[s]);k=int(x["tok_kind"][s])
            if not legal[a]:
                stats["first_unrepresentable_roots"]+=1;stats["first_and_blocked_slots"]+=len(slots)-i;break
            if legal.sum()>1:
                mats.append(mat.astype(np.float16));masks.append(legal);ys.append(a);roles.append(k)
            else:stats["forced_slots"]+=1
            prefix.advance(s,a)
    return dict(mats=np.asarray(mats),legal=np.asarray(masks),y=np.asarray(ys,np.int8),role=np.asarray(roles,np.int8)),stats

def free_prediction(f,x,weights):
    pref=F.Prefix(x,f.aux);out=np.full(F.OWN,-1,dtype=np.int64);details={}
    for s in np.flatnonzero(x["tok_mask"][:F.OWN]):
        mat,legal=bridge.features(x,pref,s);scores=bridge.predict(weights,mat)
        a=int(np.argmax(np.where(legal,scores,-np.inf)));out[s]=a
        details[int(s)]=(legal.copy(),scores,a);pref.advance(s,a)
    return f.to_actions(out),details
