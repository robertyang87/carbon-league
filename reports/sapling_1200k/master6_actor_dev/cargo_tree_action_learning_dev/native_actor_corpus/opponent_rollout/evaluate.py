"""Choose a scalar temperature on validation, then evaluate independently free-decoded physical orders."""
from collections import defaultdict
import math
import os
import time

import numpy as np
from model import (BASELINES, CONTRACT, F, HERE, ROLE, SOURCES, frames, free_prediction, load, save, sha,
                   ticks_for, validate)

TEMPERATURES=(.70,.85,1.,1.2,1.5)
ARMS=("published","calibrated","new")

def probability(logits, legal, temperature):
    logits=logits.astype(np.float64)/temperature
    x=np.where(legal,np.exp(np.clip(logits-np.max(logits[legal]),-700,0)),0.)
    return x/x.sum()

def prep(cards,weights):
    chosen=ticks_for(cards);turns=[]
    for c in cards:
        for t,obs,f,x,cmd in frames(c,chosen[c["raw_sha256"]]):
            labels=f.labels(cmd)
            assert f.to_actions(labels)=={k:v for k,v in cmd.items() if v and v!="NONE"}
            pred={};slotinfo={}
            for key,wt in weights.items():
                commands,details=free_prediction(f,x,wt)
                pred[key]=commands
                slotinfo[key]=details
            slots=[]
            for s in sorted(slotinfo["new"]):
                role=ROLE[int(x["tok_kind"][s])];uid=f.own_uid[s]
                cargo=obs["players"][c["expert_seat"]][2][uid][1] if role=="collector" else None
                slots.append((uid,role,cargo,int(labels[s]),{k:slotinfo[k][s] for k in ARMS}))
            turns.append((c,t,cmd,pred,slots))
    return turns

def choose_temperature(turns):
    strata=defaultdict(lambda:defaultdict(list))
    for c,t,cmd,pred,slots in turns:
        for _,_,_,label,details in slots:
            legal,logits,_=details["new"]
            if legal.sum()>1 and legal[label]:
                for tau in TEMPERATURES:
                    p=probability(logits,legal,tau)
                    strata[tau][c["seed"]].append(-math.log(max(p[label],1e-12)))
    mean={tau:float(np.mean([np.mean(z)for z in maps.values()]))for tau,maps in strata.items()}
    best=min(TEMPERATURES,key=lambda t:(mean[t],abs(t-1.)))
    return best,mean

def scorer():
    return dict(slots=0,correct=0,active=0,active_correct=0,predicted_active=0,supported=0,
                nonforced_supported=0,supported_nll_sum=0.,nll_sum=0.,brier_sum=0.,
                confidence_bins=[[0,0.,0]for _ in range(10)])

def update(a,prob,label,predicted,legal):
    hit=int(predicted==label);nll=-math.log(max(prob[label],1e-12));brier=float(np.square(prob).sum()+1.-2*prob[label])
    a["slots"]+=1;a["correct"]+=hit;a["active"]+=int(label!=0);a["active_correct"]+=int(label!=0)*hit
    a["predicted_active"]+=int(predicted!=0);a["supported"]+=int(legal[label]);a["nll_sum"]+=nll;a["brier_sum"]+=brier
    if legal[label] and legal.sum()>1:
        a["nonforced_supported"]+=1;a["supported_nll_sum"]+=nll
    q=float(max(prob));b=a["confidence_bins"][min(9,int(q*10))];b[0]+=1;b[1]+=q;b[2]+=hit

def summarize(a):
    n=a["slots"];supported=a["nonforced_supported"]
    return {"slots":n,"correct":a["correct"],"accuracy":a["correct"]/n,"true_active":a["active"],
            "predicted_active":a["predicted_active"],"active_correct":a["active_correct"],
            "active_accuracy":a["active_correct"]/a["active"] if a["active"]else None,
            "unsupported_by_free_prefix":n-a["supported"],"mean_nll":a["nll_sum"]/n,
            "nonforced_supported":supported,"nonforced_supported_nll":a["supported_nll_sum"]/supported if supported else None,
            "brier":a["brier_sum"]/n,"ece_10":sum(abs(x[1]-x[2])for x in a["confidence_bins"])/n}

def metrics(turns,tau):
    stats=defaultdict(scorer);joint=defaultdict(list);records=[]
    for c,t,cmd,pred,slots in turns:
        rows=[];arm_joint={}
        for key in ARMS:
            correct=pred[key]=={k:v for k,v in cmd.items()if v and v!="NONE"}
            joint[key].append(int(correct));arm_joint[key]=correct
        for uid,role,cargo,label,details in slots:
            trace={}
            for key in ARMS:
                legal,logits,choice=details[key];prob=probability(logits,legal,tau if key=="new"else 1.)
                groups=["all",role,"map/"+str(c["seed"]),"expert_seat/"+str(c["expert_seat"]),
                        "period/"+str(min(2,t//100))]
                if role=="collector" and cargo>=100:
                    groups+=["loaded_100","loaded_100/expert_seat/"+str(c["expert_seat"])]
                for cell in groups:update(stats[key,cell],prob,label,choice,legal)
                trace[key]=[int(choice),round(float(prob[label]),7),int(legal[label]),int(sum(legal))]
            rows.append([uid,role,cargo,label,trace])
        records.append({"raw_sha256":c["raw_sha256"],"t":t,"map":c["seed"],"expert_seat":c["expert_seat"],
                        "joint_correct":arm_joint,"slots":rows})
    result={}
    for arm in ARMS:
        groups={group:summarize(a)for (key,group),a in stats.items()if key==arm}
        maps=[row for group,row in groups.items() if group.startswith("map/")]
        result[arm]={"group":groups,"map_equal":{m:float(np.mean([r[m]for r in maps]))for m in
                       ("accuracy","mean_nll","nonforced_supported_nll","brier","ece_10")},
                     "joint_turns":len(joint[arm]),"joint_correct":sum(joint[arm])}
    return result,records

def main():
    start=time.process_time();validate()
    train=load(HERE/"fit.json")
    assert train["weights_sha256"]==sha(HERE/"model.npz") and train["input_protocol_sha256"]==sha(CONTRACT)
    sources=load(SOURCES);files={**BASELINES,"new":HERE/"model.npz"}
    weights={k:dict(np.load(v,allow_pickle=False))for k,v in files.items()}
    validation=prep(sources["val"],weights);tau,choice=choose_temperature(validation)
    val,vreceipts=metrics(validation,tau);del validation
    print("validation temperature",tau,choice,flush=True)
    test,treceipts=metrics(prep(sources["test"],weights),tau)
    save(HERE/"held_orders.json.gz",dict(val=vreceipts,test=treceipts))
    save(HERE/"held.json",dict(validation=val,diagnostic=test,temperature=tau,validation_temperature_nll=choice,
         weight_sha256={k:sha(p)for k,p in files.items()},held_source_sha256=sha(SOURCES),protocol_sha256=sha(CONTRACT),
         held_orders_sha256=sha(HERE/"held_orders.json.gz"),code_sha256=sha(HERE/"evaluate.py"),cpu_affinity=sorted(os.sched_getaffinity(0)),
         elapsed_cpu_seconds=time.process_time()-start,metric="Map-equal mean NLL over every actual physical action slot with p floored at 1e-12, including illegal teacher command under a model-generated prefix. Nonforced supported NLL separately reported; only this was used to calibrate temperature."))
    print("diagnostic", {k:v["map_equal"]for k,v in test.items()},flush=True)
if __name__=="__main__":main()
