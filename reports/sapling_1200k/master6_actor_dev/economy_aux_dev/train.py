"""An actual shared-encoder supervised experiment with a detach-gradient control."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import io
import json
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
TOP=ACTOR.parent/"all_top5"
sys.path.insert(0,str(TOP))
from economy_extract import visible_ledger
spec=importlib.util.spec_from_file_location("original_master6_actor_fit",ACTOR/"train.py")
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P=json.loads((HERE/"protocol.json").read_bytes())
CACHE=Path("/workspace/sapling-all-top5-20261119/replays")
SHARDS=Path("/workspace/sapling-master6-actor-data")
PARTS=("train","val","holdout")
NAMES=("own_delivery", "own_tree", "rival_delivery", "rival_tree")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def targets(records,data):
    y=np.zeros((len(data["gid"]),8),np.float32)
    valid=np.zeros_like(y,dtype=bool)
    sources={}
    for g in records:
        ix=np.flatnonzero(data["gid"]==g["gid"])
        if not len(ix):
            continue
        path=CACHE/(g["id"]+".json.gz")
        raw=path.read_bytes()
        if base.digest(raw)!=g["sha256"]:
            raise ValueError("wrong replay")
        src=json.loads(gzip.decompress(raw))
        if len(src["steps"])-1!=g["turns"]:
            raise ValueError("timeline length differs")
        flows=np.asarray(visible_ledger(src["steps"],src["configuration"])[0],dtype=np.float64)
        if flows.shape[:2]!=(g["turns"],2):
            raise ValueError("cash ledger alignment")
        side=g["seat"]
        series=flows[:,[side,1-side]][:,:,[0,2]].reshape(g["turns"],4)
        if (series<-.00001).any() or not np.isfinite(series).all():
            raise ValueError("negative or nonfinite gross cash")
        cum=np.vstack([np.zeros((1,4)),np.cumsum(series,axis=0)])
        for h_index,h in enumerate((10,30)):
            dest=slice(h_index*4,h_index*4+4)
            use=ix[data["step"][ix]+h<=g["turns"]]
            t=data["step"][use].astype(int)
            y[use,dest]=np.log1p(np.maximum(0,cum[t+h]-cum[t]))
            valid[use,dest]=True
        sources[g["id"]]=g["sha256"]
    return y,valid,sources


def objective(logits,parent_logits,batch,role_scale):
    y=batch["y"]
    active=(y>=0)&(batch["legal"].sum(-1)>=2)
    kinds=batch["tok_kind"][:,:active.shape[1]].long().clamp(0,2)
    weights=active*role_scale[kinds]
    ce=F.cross_entropy(logits.transpose(1,2),y,ignore_index=-1,reduction="none")
    kl=base.divergence(logits,parent_logits)
    return ((.8*ce+.2*kl)*weights).sum()/len(y)


def readout_loss(pred,y,valid):
    if not valid.any():
        return pred.sum()*0
    return F.smooth_l1_loss(pred[valid],y[valid])


def chosen_nll(probabilities, labels):
    index=np.maximum(labels.astype(int),0)[...,None]
    return -np.log(np.maximum(np.take_along_axis(probabilities,index,axis=-1)[...,0],1e-12))


@torch.no_grad()
def evaluate(net,head,part,data,y,mask,parent_logits):
    quant=base.quantized(net)
    logits=base.predict(quant,data)
    ce=chosen_nll(logits,data["y"])
    valid=base.actionable(data)
    variant=logits.argmax(-1); prev=parent_logits.argmax(-1)
    errors=[]; net.eval()
    for start in range(0,len(y),32):
        ix=np.arange(start,min(start+32,len(y))); b=base.batch(data,ix)
        _,ctx,_=net.encode(b)
        errors.append(F.smooth_l1_loss(head(ctx),torch.from_numpy(y[ix]),reduction="none").numpy())
    err=np.concatenate(errors)
    return {"roots":len(y),"valid_choices":int(valid.sum()),"action_nll":float(ce[valid].mean()),
            "action_agreement":float((variant[valid]==data["y"][valid]).mean()),
            "action_change_vs_parent":float((variant[valid]!=prev[valid]).mean()),
            "income_loss":float(err[mask].mean()),
            "income_loss_by_target":[float(err[mask[:,i],i].mean()) for i in range(8)],
            "map_nll":{str(g):float(ce[valid & (data['gid']==g)[:,None]].mean()) for g in np.unique(data["gid"])}}


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    records=[g for g in json.loads((ACTOR/"data_split.json").read_bytes())["games"]
             if g["split"] in PARTS and "master3s2" not in g["opponent"].lower()]
    if len(records)!=len({g["terrain_sha256"] for g in records}):
        raise ValueError("duplicate terrain")
    recs=json.loads(gzip.decompress((ACTOR/"data_receipts.json.gz").read_bytes()))
    if recs["split_sha256"]!=sha(ACTOR/"data_split.json"):
        raise ValueError("split differs")
    receipts={g["id"]:g["sha256"] for g in recs["games"] if "sha256" in g}
    parent,meta=base.model(base.WEIGHTS/"master6_actor_v1.npz",P["starting_actor"])
    all_data={}; all_hashes={}; panels={}
    for part in PARTS:
        d,hashes=base.public(part,records,SHARDS,receipts)
        y,mask,sources=targets(records,d)
        all_data[part]=(d,y,mask)
        all_hashes[part]={"shards":hashes,"replays":sources}
    d,y,mask=all_data["train"]
    center=np.array([y[mask[:,i],i].mean() for i in range(8)],dtype=np.float32)
    scale=np.array([max(.1,y[mask[:,i],i].std()) for i in range(8)],dtype=np.float32)
    for part,(dt,yt,mt) in all_data.items():
        yt-=center; yt/=scale
    teacher={p:base.predict(parent,dt) for p,(dt,_,_) in all_data.items()}
    qparent=base.quantized(parent)
    parent8={p:base.predict(qparent,dt) for p,(dt,_,_) in all_data.items() if p!="train"}
    for part in ("val","holdout"):
        dt=all_data[part][0]; mask_slots=base.actionable(dt)
        oldloss=chosen_nll(parent8[part],dt["y"])
        panels[part]={"parent_nll":float(oldloss[mask_slots].mean())}
    active=base.actionable(d)
    rows=np.flatnonzero(active.any(1))
    order=np.random.default_rng(P["seed"]).permutation(rows)[:P["steps"]*P["batch"]]
    if len(order)!=P["steps"]*P["batch"]:
        raise ValueError("insufficient eligible training roots")
    kinds=d["tok_kind"][:,:active.shape[1]]
    counts=np.array([np.count_nonzero(active & (kinds==i)) for i in range(3)])
    role_scale=torch.tensor(np.array(P["role_weights"])*len(rows)/counts,dtype=torch.float32)
    options=json.loads((ACTOR/"training_protocol.json").read_bytes())["learn"]
    products={}
    for arm in ("control","candidate"):
        torch.manual_seed(P["seed"])
        net,_=base.model(base.WEIGHTS/"master6_actor_v1.npz",P["starting_actor"])
        head=torch.nn.Linear(net.h0.in_features,8)
        groups=[[],[]]
        for name,w in net.named_parameters():
            w.requires_grad_(name.startswith(tuple(options)))
            if w.requires_grad:
                groups[0 if name.startswith(tuple(options[:2])) else 1].append(w)
        opt=torch.optim.AdamW([{"params":groups[0],"lr":P["encoder_lr"]},{"params":groups[1],"lr":P["decoder_lr"]}],weight_decay=1e-4)
        headopt=torch.optim.AdamW(head.parameters(),lr=P["readout_lr"],weight_decay=1e-4)
        net.train()
        for start in range(0,len(order),P["batch"]):
            ix=order[start:start+P["batch"]]; b=base.batch(d,ix)
            tok,ctx,_=net.encode(b); lg=net.decode(b,tok,ctx)
            loss=objective(lg,torch.from_numpy(teacher["train"][ix]),b,role_scale)
            inc=readout_loss(head(ctx if arm=="candidate" else ctx.detach()),torch.from_numpy(y[ix]),torch.from_numpy(mask[ix]))
            obj=loss+P["income_coefficient"]*inc
            if not torch.isfinite(obj):
                raise ValueError("nonfinite objective")
            opt.zero_grad(set_to_none=True);headopt.zero_grad(set_to_none=True)
            obj.backward()
            torch.nn.utils.clip_grad_norm_(groups[0]+groups[1],2.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(head.parameters(),2.,error_if_nonfinite=True)
            opt.step();headopt.step()
        path=HERE/(arm+".npz")
        base.trainer.export(net,str(path),dict(meta,economy_aux=arm))
        headfile=HERE/(arm+"_readout.npz")
        np.savez(headfile,weight=head.weight.detach().numpy(),bias=head.bias.detach().numpy(),center=center,scale=scale)
        products[arm]={"actor_sha256":sha(path),"readout_sha256":sha(headfile),
                       "actor_parameters_updated":sum(int((p.detach()!=dict(parent.named_parameters())[n].detach()).sum()) for n,p in net.named_parameters())}
        for part in ("val","holdout"):
            dt,yt,mt=all_data[part]
            panels[part][arm]=evaluate(net,head,part,dt,yt,mt,parent8[part])
        print(arm,products[arm],flush=True)
    h=panels["holdout"];a=h["candidate"];c=h["control"]
    checks={"income":a["income_loss"]<c["income_loss"],"paired_policy":a["action_nll"]<=c["action_nll"]+.01,
            "parent_policy":a["action_nll"]<=h["parent_nll"]+.005,"acts":a["action_change_vs_parent"]>=.01}
    packet={"protocol_sha256":sha(HERE/"protocol.json"),"implementation_sha256":sha(Path(__file__)),
            "training_order_sha256":base.digest(order.tobytes()),"targets":[f"{s}_{h}" for h in (10,30) for s in NAMES],
            "sources":all_hashes,"weights":products,"maps_by_split":{p:len(v["shards"]) for p,v in all_hashes.items()},
            "checks":checks,"eligible_for_native_comparison":all(checks.values()),"panels":panels}
    base.write_json(HERE/"result.json",packet)
    print(checks,flush=True)


if __name__=="__main__":
    main()
