"""Rank real paired interventions with frozen observable H10/H30 distribution heads."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
DIST=HERE.parent
ACTOR=DIST.parent
ROOT=ACTOR.parents[2]
sys.path.insert(0,str(DIST))
from score import load_model, old, base
import cnet2_feat as feat
sys.path.insert(0,str(ACTOR.parent/"root_recovery_branches"))
from branches import begin, public, clean
from all_top5.economy_extract import CashEvents, FastCarbon, POS
TYPES=("own_delivery","own_tree","rival_delivery","rival_tree")


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes())


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def require(v,why):
    if not v: raise ValueError(why)


def scored_batch(x):
    cols=("maps","scal","tok_pos","tok_kind","tok_cargo","tok_mask","hist_cell","hist_cargo","hist_move")
    # Dummy label/auxiliary target are needed only by the legacy normalization routine.
    b={k:np.expand_dims(x[k],0) for k in cols}
    b.update(teacher=np.array([2]), y=np.array([[-1]]),carbon=np.zeros((1,225),np.uint8))
    t=base.trainer.to_dev(b,False)
    return {k:v for k,v in t.items() if k not in ("y","carbon")}


@torch.no_grad()
def predict(m,head,arr,means,valid):
    ctx=m.encode(scored_batch(arr))[1]
    lg=torch.nn.functional.linear(ctx,head["weight"],head["bias"]).reshape(8,9).masked_fill(~valid,-1e9)
    return (lg.softmax(-1)*means).sum(-1).reshape(2,4).tolist()


def usable_public(obs):
    return {k:deepcopy(obs[k]) for k in ("step","player","carbon","players","trees")}


def estimated_enemy(prev,obs,s):
    ans={}
    p0=prev["players"][1-s][2]
    p1=obs["players"][1-s][2]
    for uid,worker in p1.items():
        if uid not in p0:continue
        d=feat.move_between(int(p0[uid][0]),int(worker[0]))
        if d in (1,2,3,4): ans[uid]=feat.MOVES[d]
    return clean(ans)


def estimate_world(history,obs,cfg,f,s):
    # All counters are rebuilt from public rosters, including departed IDs.
    wmax=[0,0];tmax=[0,0]
    for state in history:
        for seat in (0,1):
            for slot,acc in ((2,wmax),(3,tmax)):
                for uid in state["players"][seat][slot]:
                    acc[seat]=max(acc[seat],int(uid.rsplit("-",1)[-1])+1)
    content={k:deepcopy(obs[k]) for k in ("step","players","trees")}
    content.update(player=0,carbon=np.asarray(f.est,dtype=float).tolist(),
                   tid_counter=tmax,wid_counter=wmax)
    return FastCarbon(cfg).load_observation(content)


def sums(arr):
    a=np.array(arr,dtype=float)
    return float(a[0]+a[1]-a[2]-a[3])


def money(sim, joint, seat):
    sim.cash.events.clear()
    before=float(sim.cash[seat]-sim.cash[1-seat])
    sim.step(*joint)
    amounts=np.zeros((2,8))
    for what,s,v,*_ in sim.cash.events:
        amounts[s,POS[what]]+=v
    after=float(sim.cash[seat]-sim.cash[1-seat])
    return amounts[[seat,1-seat]][:,[0,2]].reshape(-1),after-before


def score_case(wd,models,means,valid):
    file=wd/"live_protocol.json";plan=read(file);r=plan["root"]
    s,t=plan["seat"],plan["root_t"]
    src=ROOT/plan["record"]["source"]
    sha=old.sha(src)
    require(sha==plan["record"]["source_sha256"],"invalid original source")
    raw=read(src);cfg=raw["game_configuration"]
    require(s==raw["candidate_seat"] and t==r["t"] and raw["seed"]==plan["seed"],"registered identity")
    msha=old.sha(ROOT/"data/master3s2_source.py")
    require(msha==hashlib.sha256(subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:sources/master3s2/master3s2.py"])).hexdigest(),"official source alias")
    future={};summaries={};binding={}
    for arm in ("parent","alternative"):
        a=read(wd/("live_"+arm+".json"))
        tape=wd/a["action_tape"];tail=read(tape)
        require(a["original_trajectory_sha256"]==sha and a["source_sha256"]==msha and
                a["protocol_sha256"]==old.sha(file) and a["shared_init_sha256"]==old.sha(wd/"live_shared_init.json") and
                a["action_sha256"]==old.sha(tape) and a["search_calls"]==len(tail)-1 and
                a["statuses"]==["DONE","DONE"] and len(tail)+t==a["terminal_t"],"uncertified actual responder")
        require(clean(tail[0][1-s])==clean(raw["actions"][t][1-s]),"inconsistent common enemy")
        require(list(map(clean,tail[0]))==list(map(clean,a["root_actions"])),"root manifest action mismatch")
        future[arm],summaries[arm]=tail,a
        binding[arm]={"tape":a["action_sha256"],"source_seal":old.sha(wd/("live_"+arm+".json")),"native_calls":a["search_calls"]}
    a,b=(clean(future[k][0][s]) for k in ("parent","alternative"))
    require(a==clean(raw["actions"][t][s]) and
            {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}=={r["worker"]} and
            a.get(r["worker"],"NONE")==r["old"] and b.get(r["worker"],"NONE")==r["alternative"],"not the single signed intervention")
    require(summaries["parent"]["root_opponent_observation_sha256"]==summaries["alternative"]["root_opponent_observation_sha256"] and
            summaries["parent"]["pack_sha256"]==summaries["alternative"]["pack_sha256"],"continuation/evidence binding mismatch")
    # Authority advances recorded actions only; it never calls either strategy.
    env=begin(raw); sim=FastCarbon(cfg).reset(seed=raw["seed"])
    f=feat.Feat(s);history=[]
    for i in range(t+1):
        obs=usable_public(public(env,s))
        require(obs==usable_public(sim.obs(s)),"official and replay root features differ")
        arr=f.observe(obs)
        history.append(obs)
        if i<t:
            joint=raw["actions"][i];env.step(joint);sim.step(*joint)
    estimate=estimate_world(history,obs,cfg,f,s)
    guessed=estimated_enemy(history[-2],obs,s)
    result={"seed":raw["seed"],"seat":s,"step":t,"unit":r["worker"],"old":r["old"],"new":r["alternative"],
             "source":str(wd.relative_to(ROOT)),"original_sha256":sha,"signed_protocol":old.sha(file),
             "examiner_sha256":msha,"arms":binding,"income_head_context":"focal seat, same full public prefix through root",
             "focal_public_root_sha256":digest(obs),"estimated_shared_enemy":guessed,"actual_shared_enemy":clean(future["parent"][0][1-s])}
    forecasts={};actual={}
    for arm in ("parent","alternative"):
        tail=future[arm];seal=summaries[arm]
        # Replay only to obtain on-policy observations and official numerical labels.
        authority=begin(raw)
        for j in raw["actions"][:t]:authority.step(j)
        exact=FastCarbon(cfg);exact.set_state(sim.get_state());exact.cash=CashEvents(exact.cash)
        gross=[];cash=[];forecast={}
        for j,joint in enumerate(tail):
            v,n=money(exact,joint,s);gross.append(v);cash.append(n)
            authority.step(joint)
            require(all(abs(exact.cash[k]-authority.states[0].observation.players[k][0])<.011 for k in (0,1)),"not official cash")
            if j==0:
                successor=usable_public(public(authority,s))
                require(usable_public(exact.obs(s))==successor,"wrong private successor observation")
                forecast["actual_successor"]=(successor,n)
        reward=[x.reward for x in authority.states]
        require([x.status for x in authority.states]==["DONE","DONE"] and reward==seal["score"] and
                abs(reward[s]-reward[1-s]-seal["margin"])<.011,"wrong official outcome")
        require(len(cash)>=31,"H30 incomplete after root")
        g=np.asarray(gross); net=np.asarray(cash)
        actual[arm]={"official_margin":seal["margin"],"terminal_t":seal["terminal_t"],
            "root_net_treasury":cash[0],"root_gross":g[0].tolist(),"windows":{}}
        for h in (10,30):
            actual[arm]["windows"][str(h)]={}
            for tag,interval in (("successor",slice(1,h+1)),("from_root",slice(0,h)),("root_plus_successor",slice(0,h+1))):
                totals=g[interval].sum(0)
                actual[arm]["windows"][str(h)][tag]={"channels":totals.tolist(),"net_gross_receipts":sums(totals),
                    "net_treasury_change":float(net[interval].sum())}
        public_sim=FastCarbon(cfg);public_sim.set_state(estimate.get_state())
        joint=[None,None];joint[s]=clean(tail[0][s]);joint[1-s]=guessed
        pre=public_sim.cash[s]-public_sim.cash[1-s]
        public_sim.step(*joint)
        forecast["public_estimate"]=(usable_public(public_sim.obs(s)),float(public_sim.cash[s]-public_sim.cash[1-s]-pre))
        forecasts[arm]={}
        for scenario,(after,first) in forecast.items():
            next_x=deepcopy(f).observe(after)
            forecasts[arm][scenario]={"public_successor_sha":digest(after),"root_net_treasury":first,"readouts":{}}
            for model,(enc,_,head) in models.items():
                value=predict(enc,head,next_x,means,valid)
                forecasts[arm][scenario]["readouts"][model]={str(h):{
                    "channels":v,"net_gross_receipts":sums(v),"with_root_net_treasury":sums(v)+first} for h,v in zip((10,30),value)}
    result.update(actual=actual,forecast=forecasts)
    return result


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    fit=read(DIST/"fit_receipt.json"); sup=np.load(DIST/"support.npz")
    require(old.sha(DIST/"support.npz")==fit["support_sha256"],"model support mismatch")
    means=torch.tensor(sup["means"],dtype=torch.float32);valid=torch.from_numpy(sup["counts"]>0)
    models={n:load_model(DIST,fit,n) for n in ("control","candidate")}
    lane=ACTOR/"early_lane_dev/native_response"
    paid=ACTOR/"tree_response_training/supplement_01/batches"
    cases=sorted(lane.glob("case_*/live_protocol.json")) + [x for tag in ("intake_06","intake_07") for x in sorted((paid/tag).glob("case_*/live_protocol.json"))]
    require(len(cases)==20,"root census changed")
    out=[];reject=[]
    for path in cases:
        try:
            row=score_case(path.parent,models,means,valid);out.append(row)
            print("verified",path.parent.name,flush=True)
        except ValueError as exc:
            reject.append({"source":str(path.parent.relative_to(ROOT)),"reason":str(exc)})
            print("rejected",path.parent.name,str(exc),flush=True)
    packet={"protocol_sha256":old.sha(HERE/"protocol.json"),"source_sha256":old.sha(Path(__file__)),
            "actor_sha256":{k:v["actor_sha256"] for k,v in fit["weights"].items()},"rows":out,"excluded":reject}
    base.write_json(HERE/"results.json",packet)


if __name__=="__main__":
    main()
