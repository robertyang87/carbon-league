"""Train class logits on elite public decision prefixes, and separately measure each frozen held-out group."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from build import ROOT,BASE,DEV,HERE,read,write,sha,training_module
from features import _WORKFORCE_NAMES

SPLIT=DEV/"planter_option/value_split.json"
CACHE=ROOT.parent/"sapling-all-top5-20261119"
MODULE=None
N_POOLS=2


def init_worker():
    global MODULE
    MODULE=training_module()


def extract(job):
    split,g=job;ns=MODULE
    path=CACHE/g["path"];assert sha(path)==g["sha256"]
    raw=read(path);fs=raw["steps"];cfg=raw["configuration"]
    assert cfg["randomSeed"]==g["seed"] and all(cfg[k]==v for k,v in {"size":15,"recCollectorCost":30,"recPlanterCost":30,"recCollectorIncreaseCost":0,"recPlanterIncreaseCost":0}.items())
    terrain=hashlib.sha256(bytes(fs[0][0]["observation"]["full_carbon"])).hexdigest()
    assert all("master3" not in ''.join(c for c in name.lower() if c.isalnum()) for name in g["seat_names"])
    rows=[];excluded=Counter();wdls=[]
    for seat,actor_id in enumerate(g["seat_strategy_ids"]):
        if actor_id is None:continue
        scores=raw["rewards"]
        wdl="win" if scores[seat]>scores[1-seat] else "loss" if scores[seat]<scores[1-seat] else "draw"
        wdls.append(wdl)
        for t,frame in enumerate(fs[:-1]):
            if t>170:break
            ob=frame[0]["observation"]
            obs=dict(step=t,player=seat,players=ob["players"],trees=ob["trees"],carbon=frame[seat]["observation"]["carbon"])
            if t==0:ns["agent"](obs,cfg);continue
            f=ns["_FEAT"];x=f.observe(obs)
            if t<8 or (t-8)%3:continue
            if x["cash"]<30 or x["slots"]<=0:excluded["all_bases_cannot_hire"]+=1;continue
            acts=fs[t+1][seat]["action"] or {}
            label=f.labels(acts)
            enc=[]
            for net in ns["_NETS"]:
                toks,ctx=net.encode(x,ns["_HYBRID"]["style"]);enc.append((net,toks,ctx,net.lin(ctx,"h0")))
            got,data=ns["_wf_decode"](x,f.aux,enc,False,teacher=label)
            if got is None:excluded["expert_joint_illegal_on_recorded_prefix"]+=1;continue
            for r in data:
                if not all(r["legal"]):excluded["no_shared_hiring_choice"]+=1;continue
                y=int(label[r["slot"]]);assert 0<=y<=2
                rows.append([seat,t,r["slot"],y,r["scores"].tolist(),r["features"].tolist()])
    assert wdls
    return dict(split=split,id=g["id"],seed=g["seed"],terrain=terrain,names=g["seat_names"],wdls=wdls,excluded=dict(excluded),rows=rows)


def score(data,beta,mu,sd):
    X=np.clip((data["x"]-mu)/sd,-4,4);X[:,0]=1.
    adj=np.column_stack([np.zeros(len(X)),np.clip(X@beta,-3,3)])
    scores=data["offset"]+adj
    ll=logsumexp(scores,axis=1)-scores[np.arange(len(X)),data["y"]]
    pr=scores.argmax(axis=1);raw=data["offset"].argmax(axis=1);y=data["y"]
    return dict(legal_base_rows=len(y),measured_maps=len(set(data["gid"])),actual_counts=np.bincount(y,minlength=3).tolist(),
                fitted_counts=np.bincount(pr,minlength=3).tolist(),native_counts=np.bincount(raw,minlength=3).tolist(),
                map_mean_nll=float(np.mean([ll[data["gid"]==g].mean() for g in set(data["gid"])])),
                conditional_accuracy=float(np.mean(y==pr)),
                conditional_recall=[float(np.mean(pr[y==k]==k)) if np.any(y==k) else None for k in range(3)])


def run():
    mapping=read(SPLIT)["game_lists"];alljobs=[(group,g) for group,gs in mapping.items() for g in gs]
    assert len({g["seed"] for _,g in alljobs})==len(alljobs)
    opp=lambda gs:{g["seat_names"][1-i].casefold() for g in gs for i,id_ in enumerate(g["seat_strategy_ids"]) if id_}
    assert not opp(mapping["train"])&opp(mapping["test_unseen_opponent"])
    with ProcessPoolExecutor(max_workers=N_POOLS,initializer=init_worker) as pool: records=list(pool.map(extract,alljobs,chunksize=2))
    assert len({r["terrain"] for r in records})==len(records)
    datasets={}
    for group in mapping:
        rows=[(r["id"],z) for r in records if r["split"]==group for z in r["rows"]]
        datasets[group]=dict(gid=np.array([k for k,v in rows]),x=np.asarray([v[5] for k,v in rows]),offset=np.asarray([v[4] for k,v in rows]),y=np.array([v[3] for k,v in rows]))
    train=datasets["train"];ng=Counter(train["gid"]);w=np.array([1/(len(ng)*ng[g]) for g in train["gid"]]);y=train["y"];xx=train["x"];prior=train["offset"]
    mu=np.average(xx,axis=0,weights=w);sd=np.maximum(np.sqrt(np.average((xx-mu)**2,axis=0,weights=w)),.05);mu[0]=0;sd[0]=1
    xx=np.clip((xx-mu)/sd,-4,4);n=xx.shape[1];reg=np.ones((n,2))*.025;reg[0]=0
    def loss(v):
        B=v.reshape(n,2);delta=xx@B;logits=prior+np.column_stack([np.zeros(len(xx)),np.clip(delta,-3,3)])
        z=logsumexp(logits,axis=1);p=np.exp(logits-z[:,None]);v0=np.sum(w*(z-logits[np.arange(len(y)),y]))+.5*np.sum(reg*B*B)
        p[np.arange(len(y)),y]-=1;grad=xx.T@((w[:,None]*p[:,1:])*(np.abs(delta)<3))+reg*B
        return v0,grad.ravel()
    opt=minimize(loss,np.zeros(2*n),method="L-BFGS-B",jac=True,options={"maxiter":400,"ftol":1e-12})
    assert opt.success and np.isfinite(opt.fun)
    B=opt.x.reshape(n,2)
    model=dict(features=_WORKFORCE_NAMES,mu=mu.tolist(),scale=sd.tolist(),coefficients=B.tolist(),source_split_sha256=sha(SPLIT),
               parent_sha256=read(HERE/"protocol.json")["parent_sha256"],fit_source_sha256=sha(Path(__file__)),
               training_only="No game outcome, identity or native examiner action used as input.")
    write(HERE/"weights.json",model)
    evaluation={}
    for group,data in datasets.items():
        a=score(data,np.zeros_like(B),mu,sd);b=score(data,B,mu,sd)
        c=b["actual_counts"];p=b["fitted_counts"]
        evaluation[group]=dict(prior=a,model=b,improved=b["map_mean_nll"]<a["map_mean_nll"],
            planter_excess_rate=(p[2]-c[2])/sum(c),actor_wdl=dict(Counter(v for r in records if r["split"]==group for v in r["wdls"])))
    passed=all(evaluation[k]["improved"] and evaluation[k]["planter_excess_rate"]<=.10 for k in ["test_seen_pool","test_unseen_opponent"])
    write(HERE/"evaluation.json",dict(passed=passed,weight_sha256=sha(HERE/"weights.json"),groups=evaluation,
        unseen_opponents=sorted(opp(mapping["test_unseen_opponent"])),train_unseen_opponent_overlap=0,
        per_map=[{**{k:v for k,v in r.items() if k!="rows"},"legal_rows":len(r["rows"]),"label_counts":dict(Counter(z[3] for z in r["rows"]))} for r in records]))
    print("trained",passed,[(k,v["prior"]["map_mean_nll"],v["model"]["map_mean_nll"],v["planter_excess_rate"]) for k,v in evaluation.items()],flush=True)


if __name__=="__main__":run()
