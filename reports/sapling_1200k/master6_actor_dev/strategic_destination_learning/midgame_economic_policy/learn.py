"""Fit a pair of Sapling policies on complete teams from public midgame observations."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from pydantic_core import from_json
import torch

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ACTOR=STUDY.parent
sys.path.insert(0,str(ACTOR.parent/"all_top5"))
from context import Game

spec=importlib.util.spec_from_file_location("prefix_shared",STUDY/"free_prefix_training/run.py")
prefix=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefix)
c,native=prefix.c,prefix.native
RULE=c.read(HERE/"protocol.json")
PARENT=STUDY/"bc.py"
WEIGHTS=STUDY/"bc.npz"
CACHE=Path("/workspace/sapling-all-top5-20261119/replays")
WORK=Path("/tmp/sapling-midgame-"+c.sha(HERE/"protocol.json")[:12])
ARMS=("uniform","weighted")
TIMES=RULE["times"]
BATCH=16  # Registered in the learner field of protocol.json.


def require(ok,why):
    if not ok:
        raise ValueError(why)


def cards():
    p=ACTOR/"broader_elite_backbone_dev/panel.json"
    require(c.sha(p)==RULE["panel"] and c.sha(PARENT)==RULE["parent"] and c.sha(WEIGHTS)==RULE["weights"],"sourcing identity")
    exclude=set()
    for name,signature in RULE["exclude_hashes"].items():
        path=STUDY/"opponent_matched_economy"/name
        require(c.sha(path)==signature,"blocked identities changed")
        exclude.update(r["meta"]["terrain"] for r in c.read(path))
    selected=[r for r in c.read(p)["games"] if r["terrain_sha256"] not in exclude]
    require(dict(Counter(r["split"] for r in selected))==RULE["games_by_split"] and
            len(selected)==len({r["terrain_sha256"] for r in selected})==len({r["id"] for r in selected}),"physical map overlap")
    return selected


def archive(card):
    path=CACHE/(card["id"]+".json.gz");raw=path.read_bytes()
    require(sha256(raw).hexdigest()==card["sha256"],"wrong replay "+card["id"])
    game=Game(from_json(gzip.decompress(raw)))
    require(game.turns==card["turns"] and
            sha256(bytes(game._frames[0][0]["observation"]["full_carbon"])).hexdigest()==card["terrain_sha256"],"terrain/time changed")
    return game


def distance(a,b):
    x=abs(a%15-b%15);y=abs(a//15-b//15)
    return min(x,15-x)+min(y,15-y)


def board_state(obs):
    sides=[obs["players"][obs["player"]],obs["players"][1-obs["player"]]]
    v=[];reach=[]
    for s,(cash,bases,workers,trees) in enumerate(sides):
        other=sides[1-s][2]
        carriers=[(p,a) for p,a,k in workers.values() if k=="COLLECTOR"]
        heavy=[(p,a) for p,a in carriers if a>=75]
        contest=[]
        for p,a in heavy:
            if any(kind=="COLLECTOR" and weight<a and distance(p,q)<=3 for q,weight,kind in other.values()):
                contest.append(a)
        reach.append(bool(contest))
        v.extend([cash,len(workers),len(carriers),len(trees),sum(a for _,a in carriers),len(heavy),sum(contest),
                  sum(a/(1+min(distance(p,b) for b in bases.values())) for p,a in carriers)])
    return [obs["step"]/100,*v],("both_threats" if all(reach) else "protect_own" if reach[0] else "contest_opponent" if reach[1] else "neither"),sides[0][0]-sides[1][0]


def source_rows(card,game):
    seat=card["seat"]
    for t in TIMES:
        if t+RULE["future_horizon"]>game.turns:continue
        obs=game._obs(t,seat);state,opportunity,lead=board_state(obs)
        end=game._frames[t+RULE["future_horizon"]][0]["observation"]["players"]
        y=(end[seat][0]-end[1-seat][0])-lead
        yield t,dict(gid=card["id"],terrain=card["terrain_sha256"],seat=seat,t=t,teacher=card["teacher"],split=card["split"],
                     opportunity=opportunity,cash_margin=lead,regressors=state,target=y)


def prepare():
    WORK.mkdir(exist_ok=True)
    actor=c.package(PARENT);reader=prefix.common.prepare.teacher_reader(actor); F=sys.modules["cnet2_feat"]
    rows=[];meta=[]
    for card in cards():
        if card["split"]!="train":continue
        game=archive(card);selected=dict(source_rows(card,game));f=F.Feat(card["seat"])
        for t in range(min(game.turns,max(TIMES)+1)):
            obs=game._obs(t,card["seat"]);x=f.observe(obs)
            if t not in selected:continue
            r=selected[t];sent,rejected,enc=prefix.free(actor,x,f)
            expert=game._frames[t+1][card["seat"]]["action"] or {}
            b=c.actual_features(f,x,expert,actor)
            cut=int(b.pop("cut"));ok=not np.any(x["tok_mask"][cut:F.OWN]) and b["eligible"].any()
            r.update(supported=bool(ok),guard_rejected=bool(rejected),cut=cut,free_matches_expert=bool(f.to_actions(sent)==expert))
            if ok:
                fixed=np.zeros_like(b["residual"]);prior=np.zeros_like(fixed)
                reader(x,f.aux,enc,not rejected,b["y"],fixed,prior,cut)
                del b["residual"];b.update(fixed=fixed,prior=prior);r["index"]=len(rows);rows.append(b)
            meta.append(r)
    require(len({r["terrain"] for r in meta})==234 and len({r["terrain"] for r in meta if r["supported"]})==234,
            "training map lacks a complete expert action")
    np.savez(WORK/"training.npz",**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    c.save(HERE/"training_rows.json.gz",meta)


def regfit(train):
    X=np.array([r["regressors"] for r in train],dtype=np.float64);Y=np.array([r["target"] for r in train])
    counts=Counter(r["terrain"] for r in train);w=np.array([1/counts[r["terrain"]] for r in train])
    mu=np.average(X,axis=0,weights=w);sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=w)).clip(min=1e-6)
    z=(X-mu)/sd;mean=np.average(Y,weights=w)
    beta=np.linalg.solve(z.T@(w[:,None]*z)+RULE["ridge"]*np.eye(z.shape[1]),z.T@(w*(Y-mean)))
    return dict(mu=mu.tolist(),sd=sd.tolist(),beta=beta.tolist(),mean=mean)


def predict(model,rows):
    return (np.array([r["regressors"] for r in rows])-model["mu"])/model["sd"]@np.array(model["beta"])+model["mean"]


def crossfit(rows):
    maps=sorted({r["terrain"] for r in rows})
    groups={m:i%5 for i,m in enumerate(maps)}
    models=[]
    for fold in range(5):
        training=[r for r in rows if groups[r["terrain"]]!=fold]
        test=[r for r in rows if groups[r["terrain"]]==fold]
        m=regfit(training);pred=predict(m,test)
        for r,v in zip(test,pred):r["fold"]=fold;r["baseline"]=float(v);r["residual"]=r["target"]-v
        models.append(m)
    res=np.array([r["residual"] for r in rows])
    scale=max(1.,1.4826*float(np.median(np.abs(res-np.median(res)))))
    for r in rows:r["unnormalized"]=1+.75*np.tanh(r["residual"]/scale)
    for terrain in maps:
        z=[r for r in rows if r["terrain"]==terrain]
        norm=np.mean([r["unnormalized"] for r in z])
        for r in z:r["weight"]=r["unnormalized"]/norm
    return dict(folds=models,scale=scale,train_all=regfit(rows))


def publish(net,model,name):
    with np.load(WEIGHTS,allow_pickle=False) as original:
        updated=prefix.fpweights(net,model._NETS[1].w,original)
        weights=HERE/(name+".npz");np.savez(weights,**updated,meta=original["meta"])
        fields=native.repack.quantize(weights,original)
    src=PARENT.read_text();path=HERE/(name+".py");c.atomic(path,native.repack.substitute(src,fields).encode())
    require(native.repack.substitute(path.read_text(),native.repack.literal(src,native.repack.STUDENT))==src and path.stat().st_size<=10_000_000,
            "release code, partner, or byte envelope")
    q=c.package(path);changed={k:int(np.count_nonzero(v!=q._NETS[1].w[k])) for k,v in model._NETS[1].w.items() if not k.endswith(".T")}
    require(all(changed[k]>0 for k in ("whead.weight","gru.weight_ih","stem.weight")) and
            not any(v for k,v in changed.items() if k.startswith(native.VALUE)),"wrong scopes")
    return dict(package=c.sha(path),weights=c.sha(weights),size=path.stat().st_size,parameter_changes=changed)


def train():
    rows=c.read(HERE/"training_rows.json.gz");used=[r for r in rows if r["supported"]]
    require([r["index"] for r in used]==list(range(len(used))) and len({r["terrain"] for r in used})==234,"training identities")
    baseline=crossfit(used);c.save(HERE/"ridge.json",baseline);c.save(HERE/"advantage.json.gz",used)
    counts=Counter(r["terrain"] for r in used)
    map_weight=np.array([len(used)/len(counts)/counts[r["terrain"]] for r in used])
    with np.load(WORK/"training.npz",allow_pickle=False) as z:data={k:z[k] for k in z}
    rng=np.random.default_rng(RULE["seed"]);order=rng.permutation(len(used))
    model=c.package(PARENT)
    result={}
    for arm in ARMS:
        torch.manual_seed(RULE["seed"]);net=c.torch_model(model,1);net.eval()
        for k,v in net.named_parameters():v.requires_grad_(not k.startswith(native.VALUE))
        params=[v for v in net.parameters() if v.requires_grad]
        opt=torch.optim.AdamW(params,lr=RULE["lr"],weight_decay=RULE["weight_decay"]);losses=[]
        weights=np.array([r["weight"] if arm=="weighted" else 1. for r in used])*map_weight
        for begin in range(0,len(order),BATCH):
            ids=order[begin:begin+BATCH];b=native.batch(data,ids)
            scores=prefix.common.logits(net,b); lp=scores.log_softmax(-1)
            prev=torch.from_numpy(data["prior"][ids]).masked_fill(~b["legal"],-1e9).log_softmax(-1)
            ce=-lp.gather(-1,b["y"].clamp_min(0)[...,None]).squeeze(-1)
            kld=(prev.exp()*(prev-lp)).sum(-1);mask=b["eligible"].float()
            per=mask/mask.sum(-1,keepdim=True).clamp_min(1)
            imitation=(ce*per).sum(1);kl=(kld*per).sum(1)
            if begin==0:
                valid=(mask.bool()[:,:,None]&b["legal"]);diff=scores.detach()-torch.from_numpy(data["prior"][ids])
                adj=diff-diff.masked_fill(~valid,0).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
                drift=float(adj[valid].abs().max());require(drift<.003,"initial deployed logit parity")
            loss=(imitation*torch.tensor(weights[ids],dtype=torch.float32)+RULE["kl"]*kl*torch.tensor(map_weight[ids],dtype=torch.float32)).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(params,RULE["clip"],error_if_nonfinite=True));opt.step()
            losses.append([float(loss.detach()),grad])
        result[arm]=dict(**publish(net,model,arm),updates=len(losses),loss_grad=losses,initial_logit_drift=drift)
    c.save(HERE/"fit.json",dict(arms=result,registered=c.sha(HERE/"protocol.json"),source=c.sha(Path(__file__)),n=len(used),
                                ridge=c.sha(HERE/"ridge.json"),labels=c.sha(HERE/"advantage.json.gz")))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("prepare","train","all"));a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if a.phase in ("all","prepare"):prepare()
    if a.phase in ("all","train"):train()
