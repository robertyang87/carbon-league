"""Fit equal-budget full-policy and readout-only arms against actual original expert commands."""
import copy
import json
import platform
import time
import torch
from common import *

sys.path.insert(0,str(ACTOR/"blended_actor_dev/serving"))
import build as repack

VALUE = ("aux.", "utility.", "value.", "wdl.", "margin.")
HEAD = ("whead.", "bhead.", "U.", "V.")

def batch(d, ix):
    return tensorbatch({**d,"residual":d["fixed"]},ix)

def fit_loss(model,b,prior):
    z=(b["residual"]+.25*model(b)[0]).masked_fill(~b["legal"],-1e9)
    lp=z.log_softmax(-1)
    lp0=prior.masked_fill(~b["legal"],-1e9).log_softmax(-1)
    ce=-lp.gather(-1,b["y"].clamp_min(0).unsqueeze(-1)).squeeze(-1)
    kl=(lp0.exp()*(lp0-lp)).sum(-1)
    w=b["eligible"].float()
    kinds=b["tok_kind"][:,:w.shape[1]]
    per_role=torch.zeros_like(w);present=torch.zeros_like(w[:,:1])
    for role in range(3):
        mask=w*(kinds==role)
        count=mask.sum(1,keepdim=True)
        per_role+=mask/count.clamp_min(1)
        present+=(count>0)
    wt=.5*(per_role/present.clamp_min(1)+w/w.sum(1,keepdim=True).clamp_min(1))
    return (wt*ce).sum(1).mean(),(wt*kl).sum(1).mean()

def export(net, m, arm, output):
    raw_child={}
    init=m._NETS[1].w
    with np.load(BASE,allow_pickle=False) as ref:
        for k in net.state_dict():
            delta=net.state_dict()[k].detach().numpy()-init[k]
            raw_child[k]=ref[k]+delta
        path=output/(arm+".npz")
        np.savez(path,**raw_child,meta=ref["meta"])
        fields=repack.quantize(path,ref)
    src=PACK.read_text()
    packaged=output/(arm+".py");atomic(packaged,repack.substitute(src,fields).encode())
    fresh=package(packaged)
    if repack.substitute(packaged.read_text(),repack.literal(src,repack.STUDENT))!=src:
        raise ValueError("release code/fixed partner changed")
    changes={k:int(np.count_nonzero(init[k]!=fresh._NETS[1].w[k])) for k in net.state_dict()}
    if any(v and (k.startswith(VALUE) or (arm=="head" and not k.startswith(HEAD))) for k,v in changes.items()):
        raise ValueError("nontrainable parameters changed during packing")
    return dict(package_sha256=sha(packaged),weights_sha256=sha(path),package_bytes=packaged.stat().st_size,
                quantized_changed_by_tensor=changes,quantized_changed=sum(changes.values()),
                changed_tensors=sum(v>0 for v in changes.values()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad))

def parity_check(d,student,root_count=96):
    # Two early positions and one midgame position on every training tape.
    ix=np.concatenate([np.array([i,i+1,i+root_count//2]) for i in range(0,len(d['y']),root_count)])
    errs=[]
    with torch.inference_mode():
        for ids in np.array_split(ix,min(13,len(ix))):
            b=batch(d,ids);valid=b["eligible"][:,:,None] & b["legal"]
            z=b["residual"]+.25*student(b)[0]
            err=z-torch.from_numpy(d["parent"][ids])
            err=err-err.masked_fill(~valid,0).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
            errs.append(float(err[valid].abs().max()) if valid.any() else 0.)
    if max(errs)>=.003:raise ValueError(("production-vs-Torch legal logits",max(errs)))
    return dict(roots=len(ix),max_centered_error=max(errs))

def smoke_checkpoint(models,m,arms,init,source):
    checkpoint=HERE/"work"/"first_release";checkpoint.mkdir(exist_ok=True)
    cards=[c for c in source["cards"] if c["split"]=="train"]
    # This check uses two training tapes only. The held labels remain unopened.
    receipts={};packages={"parent":package()}
    for arm in arms:
        receipts[arm]=export(models[arm],m,arm,checkpoint)
        packages[arm]=package(checkpoint/(arm+".py"))
    stats={arm:{"calls":0,"changed_joint":0,"changed_workers":0} for arm in arms}
    for card in cards[:2]:
        for t,obs,cfg,_,take in iter_public(card,8):
            if take or t==0:
                orders={arm:n.agent(obs,cfg) for arm,n in packages.items()}
                if take:
                    s=1-card['seat']
                    for arm in arms:
                        stat=stats[arm]; stat["calls"]+=1
                        stat["changed_joint"]+=orders[arm]!=orders["parent"]
                        stat["changed_workers"]+=sum(orders[arm].get(u)!=orders["parent"].get(u) for u in obs["players"][s][2])
            else:
                for n in packages.values():n._FEAT.observe(obs)
    save(HERE/"checkpoint_78.json",dict(steps=78,split="training",arms=receipts,free_entry=stats,identity=init))

def main():
    rule=read(HERE/"protocol.json");data=read(HERE/"data.json");source=read(HERE/"sources.json")
    if os.sched_getaffinity(0)!={0} or sha(HERE/"protocol.json")!=data["protocol_sha"]:
        raise ValueError("bad affinity/contract")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(rule["seed"])
    # Initial arm construction and validation are pinned to dequantized production weights.
    m=package()
    with np.load(BASE,allow_pickle=False) as ref:
        if repack.quantize(BASE,ref)["_W64"]!=m._W64:raise ValueError("different parent weight file")
    keys=None;parts={}
    for sh in data["shards"]:
        path=HERE/"work"/sh["file"]
        if sha(path)!=sh["data_sha"]:raise ValueError("changed input shard")
        with np.load(path,allow_pickle=False) as z:
            for k in z.files:parts.setdefault(k,[]).append(z[k])
    d={k:np.concatenate(v) for k,v in parts.items()};del parts
    order=np.random.default_rng(rule["seed"]).permutation(len(d["y"])).reshape(-1,rule["batch"])
    assert len(order)==rule["final_checkpoint_steps"] and len(np.unique(order))==2496
    arms=("full","head");models={a:torch_model(m,1) for a in arms};opts={}
    for arm,net in models.items():
        for k,p in net.named_parameters():
            p.requires_grad_((not k.startswith(VALUE)) if arm=="full" else k.startswith(HEAD))
        opts[arm]=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=rule["lr"],weight_decay=rule["weight_decay"])
    parity=parity_check(d,models["full"])
    signature={k:sha(HERE/k) for k in ("common.py","fit.py","prepare_data.py","protocol.json","data.json","sources.json","training_roots.json.gz")}
    ck=HERE/"work"/"checkpoint.pt";start=0;history=[];elapsed=0
    if ck.is_file():
        cp=torch.load(ck,map_location="cpu",weights_only=False)
        if cp["signature"]!=signature:raise ValueError("checkpoint inputs changed")
        start=cp["step"];history=cp["history"];elapsed=cp['seconds']
        for arm in arms:
            models[arm].load_state_dict(cp["models"][arm]);opts[arm].load_state_dict(cp["opts"][arm])
    if start==rule["first_checkpoint_steps"] and not (HERE/"checkpoint_78.json").is_file():
        smoke_checkpoint(models,m,arms,signature,source)
    now=time.monotonic()
    for step,ix in enumerate(order[start:],start+1):
        b=batch(d,ix);prior=torch.from_numpy(d["parent"][ix]);line={}
        for arm in arms:
            net=models[arm];opt=opts[arm]
            ce,kl=fit_loss(net,b,prior)
            loss=ce+rule["kl"]*kl
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            if not torch.isfinite(loss):raise ValueError("nonfinite CE/KL")
            if arm=="full" and step==1 and net.stem.weight.grad.abs().sum()==0:raise ValueError("no gradient to trunk")
            opt.step()
            line[arm]=[float(ce.detach()),float(kl.detach()),float(norm)]
        history.append(line)
        if step%rule["checkpoint_every"]==0 or step==len(order):
            sec=elapsed+time.monotonic()-now
            tmp=ck.with_suffix(".tmp")
            torch.save(dict(signature=signature,step=step,history=history,seconds=sec,models={a:n.state_dict() for a,n in models.items()},opts={a:o.state_dict() for a,o in opts.items()}),tmp)
            os.replace(tmp,ck)
            if step==rule["first_checkpoint_steps"]:smoke_checkpoint(models,m,arms,signature,source)
            print("step",step,"/",len(order),round(sec,1),line,flush=True)
    if not (HERE/"checkpoint_78.json").is_file():raise ValueError("first checkpoint never inspected")
    result={a:export(models[a],m,a,HERE) for a in arms}
    for arm in arms:
        result[arm].update(updates=len(order),unique_roots=len(d["y"]),distinct_terrain=13,sampled_expert_seats=[0,1])
    save(HERE/"loss.json.gz",history)
    deps=[ROOT/"code"/f for f in ("fastsim.py","cnet2_model.py","cnet2_feat.py","cnet2_np.py","cnet2_pack.py")]
    deps.extend([Path(policy.__file__),Path(repack.__file__)])
    save(HERE/"fit.json",dict(arms=result,signature=signature,parent=sha(PACK),weights=sha(BASE),parent_logit_parity=parity,
        software={str(p.relative_to(ROOT)):sha(p) for p in deps},python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),
        affinity=sorted(os.sched_getaffinity(0)),seconds=elapsed+time.monotonic()-now,loss_sha256=sha(HERE/"loss.json.gz"),
        order_int64_sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()))
    print("packages",[(a,result[a]["package_sha256"]) for a in arms],flush=True)
if __name__=="__main__":main()
