"""Matched real policy updates with out-of-map mining advantage estimates and a frozen-parent KL."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as TF
from data import ACTOR, HERE, CACHE, OLD, sha, read, dump
spec=importlib.util.spec_from_file_location("signed_parent_trainer",ACTOR/"train.py")
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
P=read(HERE/"protocol.json")
SEED=658613

def regress(rows):
    x=np.asarray([r["observed"] for g,r in rows])
    y=np.log1p([r["new_settlement"][0] for g,r in rows])
    ids=np.array([g["gid"] for g,r in rows]);_,ix,num=np.unique(ids,return_inverse=True,return_counts=True)
    w=1/num[ix];w=w/w.sum()
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-6]=1.
    design=np.column_stack([np.ones(len(x)),np.clip((x-mu)/sd,-4,4)])
    reg=np.eye(design.shape[1])*.1;reg[0,0]=0
    beta=np.linalg.solve(design.T@(w[:,None]*design)+reg,design.T@(w*y))
    return {"mu":mu.tolist(),"sd":sd.tolist(),"beta":beta.tolist(),"ridge":.1}

def implied(r,reg):
    return reg["beta"][0]+np.clip((np.asarray(r["observed"])-reg["mu"])/reg["sd"],-4,4)@reg["beta"][1:]

def fold(g):
    return int(sha(("late-state-fold1|"+g["terrain_sha256"]).encode()),16)%4

def assemble(games):
    fitted=[(g["card"],r) for g in games if g["card"]["split"]=="train" for r in g["sample"]]
    regs={str(i):regress([(g,r) for g,r in fitted if fold(g)!=i]) for i in range(4)}
    regs["held"]=regress(fitted)
    phase={}; weighted=[]; cohort={}
    pmap={g["card"]["opponent"] for g in games if g["card"]["split"]=="train"}
    for part in ("train","held"):
        maps=[g for g in games if g["card"]["split"]==part]
        matrices=[];tg=[];ws=[]
        frozen=HERE/(part+"_inputs.npz")
        archive=None
        if frozen.exists():
            receipt=read(HERE/"label_audit.json")["public_input_receipts"][part]
            if sha(frozen.read_bytes())!=receipt["sha256"]:raise ValueError("public input archive digest")
            with np.load(frozen,allow_pickle=False) as z:archive={k:z[k] for k in z.files}
        for packet in maps:
            g=packet["card"];reg=regs[str(fold(g)) if part=="train" else "held"]
            path=(OLD if part=="train" else CACHE)/(g["id"]+".npz")
            if archive is None:
                if sha(path.read_bytes())!=packet["shard_sha256"]:raise ValueError("shard digest")
                with np.load(path,allow_pickle=False) as z: d={k:z[k] for k in base.trainer.FIELDS}
            else:
                belongs=archive["gid"]==g["gid"]
                d={k:v[belongs] for k,v in archive.items()}
                # The shared trainer's conversion helper requires legacy target columns; public-only actor fitting never reads them.
                for k in ("carbon","value"):
                    dtype,shape=base.trainer.FIELDS[k]
                    d[k]=np.zeros((len(d["gid"]),*shape),dtype=dtype)
            if not len(d["step"]):continue
            rows={int(t):i for i,t in enumerate(d["step"])}
            active=np.zeros(d["y"].shape,bool);weight=np.ones(d["y"].shape,np.float32)
            for r in packet["sample"]:
                i=rows[r["t"]];j=r["slot"];y=float(np.log1p(r["new_settlement"][0]));pred=float(implied(r,reg))
                value=float(np.exp(np.clip((y-pred)/2,-.7,.7)))
                active[i,j]=True;weight[i,j]=value
                weighted.append([g["id"],r["t"],r["uid"],r["slot"],y,pred,value,str(fold(g)) if part=="train" else "held"])
            cohort[int(g["gid"])]=dict(**g,is_seen_train_opponent=g["opponent"] in pmap)
            matrices.append(d);tg.append(active);ws.append(weight)
        phase[part]=(base.join_parts(matrices),np.concatenate(tg),np.concatenate(ws))
    dump(HERE/"state_baseline.json",regs);dump(HERE/"advantage.json.gz",{"schema":["game","t","uid","slot","log_settled_new","predicted_without_map","weight","baseline_fold"],"rows":weighted})
    return phase,cohort

def score(net, d, use, weights, part, cards):
    prob=base.predict(base.quantized(net),d)
    acts=d["y"].clip(0)[...,None];nll=-np.log(np.take_along_axis(prob,acts,axis=-1)[...,0].clip(1e-12))
    output={}
    for gid in np.unique(d["gid"]):
        mask=use & (d["gid"]==gid)[:,None]
        if not mask.any():continue
        r=cards[int(gid)]; idx=np.where(mask);correct=(prob.argmax(-1)[idx]==d["y"][idx]); ww=weights[idx]
        output[r["id"]]={"gid":int(gid),"slots":len(correct),"late_nll":float(nll[idx].mean()),
                         "weighted_nll":float(np.average(nll[idx],weights=ww)),"agreement":float(correct.mean()),
                         "wdl":r["wdl"],"seat":r["seat"],"opponent":r["opponent"],"seen":r["is_seen_train_opponent"]}
    return output,prob

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
    records=read(HERE/"settlements.json.gz")["games"]
    phase,cohort=assemble(records)
    parent,meta=base.model(base.WEIGHTS/"master6_actor_v1.npz",P["parent"])
    dt,tar,wts=phase["train"];late=np.flatnonzero(tar.any(1));early=np.flatnonzero(dt["step"]<225)
    order=[];rng=np.random.default_rng(SEED)
    for _ in range(48):
        order.append(np.concatenate([rng.choice(late,12,replace=False),rng.choice(early,4,replace=False)]))
    order=np.asarray(order);teacher=base.predict(parent,dt)
    dh,th,wh=phase["held"]; outcomes={}; probabilities={}
    outcomes["parent"],probabilities["parent"]=score(parent,dh,th,wh,"held",cohort)
    train_result={}
    for arm in ("control","candidate"):
        torch.manual_seed(SEED)
        net,_=base.model(base.WEIGHTS/"master6_actor_v1.npz",P["parent"])
        for name,p in net.named_parameters():p.requires_grad_(name.startswith(("whead.","U.","V.")))
        opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.00016,weight_decay=.0001)
        net.train();losses=[]
        for ix in order:
            b=base.batch(dt,ix); logits=net(b)[0]; y=b["y"];valid=(y>=0)&(b["legal"].sum(-1)>=2)
            m=torch.from_numpy(tar[ix]); weights=torch.from_numpy(wts[ix]) if arm=="candidate" else torch.ones_like(m,dtype=torch.float32)
            kl=base.divergence(logits,torch.from_numpy(teacher[ix]));ce=TF.cross_entropy(logits.transpose(1,2),y,ignore_index=-1,reduction="none")
            target=(.55*(weights[m]*ce[m]).sum()/weights[m].sum()+.45*kl[m].mean())
            loss=target+kl[valid&~m].mean()
            if not torch.isfinite(loss):raise ValueError("nonfinite loss")
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            losses.append(loss.item())
        file=HERE/(arm+".npz");base.trainer.export(net,str(file),dict(meta,late_advantage=arm))
        changes={n:int((p.detach()!=dict(parent.named_parameters())[n].detach()).sum()) for n,p in net.named_parameters() if p.requires_grad}
        train_result[arm]={"sha256":sha(file.read_bytes()),"changed_from_parent":changes,"physical_updates":len(order),"losses":losses}
        outcomes[arm],probabilities[arm]=score(net,dh,th,wh,"held",cohort)
        print(arm,train_result[arm]["sha256"],sum(changes.values()),flush=True)
    sc={}
    all_ids=list(outcomes["candidate"])
    for key,ii in {"all":all_ids,"opponent_unseen":[i for i in all_ids if not outcomes["parent"][i]["seen"]],
                   **{s:[i for i in all_ids if outcomes["parent"][i]["wdl"]==s] for s in "WDL"},
                   **{f"seat{s}":[i for i in all_ids if outcomes["parent"][i]["seat"]==s] for s in (0,1)}}.items():
        sc[key]={"maps":len(ii),"arms":{a:{m:float(np.mean([outcomes[a][i][m] for i in ii])) for m in ("late_nll","weighted_nll","agreement")} for a in outcomes} if ii else {}}
    ac=probabilities["candidate"].argmax(-1);co=probabilities["control"].argmax(-1);pa=probabilities["parent"].argmax(-1)
    comps={"late_changed_teacher_prefix_vs_control":int(((ac!=co)&th).sum()),"late_changed_teacher_prefix_vs_parent":int(((ac!=pa)&th).sum()),"eligible":int(th.sum())}
    pair=np.array([[outcomes["candidate"][k][m]-outcomes["control"][k][m] for m in ("weighted_nll","late_nll")] for k in all_ids])
    boot=rng.integers(0,len(pair),size=(2000,len(pair)));ci=np.quantile(pair[boot].mean(axis=1),[.025,.975],axis=0)
    dump(HERE/"fit_result.json",{"protocol_sha256":sha((HERE/"protocol.json").read_bytes()),"weights":train_result,"order_sha256":sha(order.tobytes()),
             "root_slot_checks":comps,"panels":sc,"by_game":outcomes,"paired_map_CI_candidate_minus_control_weighted_and_plain":ci.tolist()})
if __name__=="__main__":main()
