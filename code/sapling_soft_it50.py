"""Distil the frozen it50 action distribution into a single Sapling on common, executed prefixes."""
import argparse
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf

from cnet2_model import CNet2
from init_sapling import resolve_preset_metadata
from sapling_bootstrap import sha
from sapling_cash_train import checked_shard, free_slots
from sapling_tree_train import ELITE_SHA, PARENT_SHA, WEIGHT

SEED = 551197
UPDATES, LR = 192, 1e-4
SOFT_BATCH, ELITE_BATCH = 26, 6
SOFT_WEIGHT, ELITE_WEIGHT = .85, .15
BEHAVIORS = ("Sapling cash V3", "it50 bare style 2")


def corpus(manifest, trainer, soft):
    doc=json.loads(manifest.read_bytes())
    games=doc["games"]
    if not soft and sha(manifest)!=ELITE_SHA:
        raise ValueError("unrecognized Top Five data")
    if len({g["seed"] for g in games})!=len(games) or len({g["id"] for g in games})!=len(games):
        raise ValueError("duplicate games or map seeds")
    if soft:
        for split, seeds in (("train",range(620176000,620176018)),("val",range(620176018,620176024))):
            part=[g for g in games if g["split"]==split]
            if {g["seed"] for g in part}!=set(seeds) or {g["behavior"] for g in part}!=set(BEHAVIORS):
                raise ValueError("different 18/6 prospective split or behavior sources")
    groups={"train":[],"val":[]}
    rng=np.random.default_rng(SEED+1)
    for g in games:
        n,split=g["rows"],g["split"]
        ix=slice(None)
        if not soft:
            take=min(64 if split=="train" else 40,n)
            ix=np.sort(rng.choice(n,take,replace=False))
        with checked_shard(manifest,g) as f:
            d={k:f[k][ix] for k in trainer.FIELDS}
            if not (d["teacher"]==2).all() or len(f["y"])!=n:
                raise ValueError("incompatible style or row count")
            y=d["y"]
            free=free_slots(d)
            if np.any((y < -1)|(y>4)) or not np.all(d["legal"][np.where(y>=0)+(y[y>=0],)]):
                raise ValueError("invalid executed action")
            if soft:
                q,mask,hard=(f[k] for k in ("teacher_probs","soft_mask","teacher_argmax"))
                active=d["tok_mask"][:,:y.shape[1]]
                if (q.dtype!=np.float32 or q.shape!=d["legal"].shape or not np.isfinite(q).all()
                        or (q<0).any() or (q[~d["legal"]]!=0).any()
                        or not np.array_equal(active,y>=0) or not np.array_equal(mask,free)
                        or not np.allclose(q.sum(-1),active,rtol=0,atol=1e-5)
                        or not np.array_equal(q.argmax(-1)[active],hard[active]) or (hard[~active]!=-1).any()):
                    raise ValueError("invalid soft targets on "+g["id"])
                d.update(q=q, label=hard, source=np.full(n,BEHAVIORS.index(g["behavior"]),dtype=np.int8))
            else:
                d["label"]=y
        if not soft and split=="val":
            raw=(manifest.parent/g["mask_shard"]).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=g["mask_sha256"]:
                raise ValueError("Top Five context mask identity")
            with np.load(io.BytesIO(raw), allow_pickle=False) as m:
                d.update(attack=m["attack"][ix]&free, defense=m["defense"][ix]&free)
        groups[split].append(d)
    data={split:{k:np.concatenate([d[k] for d in rows]) for k in rows[0]} for split,rows in groups.items()}
    return data,games


def logits(net, trainer, batch):
    scores=net(trainer.to_dev(batch,aug=False))[0]
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite policy logits")
    return scores


def divergence(z,q):
    # Both terms are zero at q=0, including illegal moves.
    return (torch.xlogy(q,q)-q*nnf.log_softmax(z,-1)).sum(-1)


@torch.inference_mode()
def evaluate(net,trainer,d,soft):
    net.eval()
    preds,losses=[],[]
    for lo in range(0,len(d["y"]),32):
        ix=np.arange(lo,min(lo+32,len(d["y"])))
        z=logits(net,trainer,trainer.fetch(d,ix))
        preds.append(z.argmax(-1).cpu().numpy())
        if soft:
            loss=divergence(z,torch.as_tensor(d["q"][ix],device=z.device))
        else:
            labels=torch.as_tensor(d["label"][ix],dtype=torch.long,device=z.device)
            loss=nnf.cross_entropy(z.transpose(1,2),labels,ignore_index=-1,reduction="none")
        losses.append(loss.cpu().numpy())
    pred,loss=np.concatenate(preds),np.concatenate(losses)
    y=d["label"]
    valid=free_slots(d)
    kind=d["tok_kind"][:,:y.shape[1]]
    masks={"all":valid,"opening_t0_59":(d["step"]<60)[:,None],
           "base":kind==2,"collector":kind==0,"planter":kind==1}
    if soft:
        for n,source in enumerate(BEHAVIORS):
            rows=(d["source"]==n)[:,None]
            masks[source]=rows
            masks[source+" / base"]=rows & (kind==2)
        masks["teacher_disagrees_with_executed"]=y!=d["y"]
    else:
        masks.update(attack=d["attack"],defense=d["defense"],ordinary=~(d["attack"]|d["defense"]))
    result={}
    for name,v in masks.items():
        m=valid & v
        n=int(m.sum())
        result[name]=dict(slots=n,correct=int((pred[m]==y[m]).sum()),games=int(len(set(d["gid"][m.any(1)]))),
                          mean_kl_or_ce=float(loss[m].mean(dtype=np.float64)) if n else None,
                          label_actions=np.bincount(y[m],minlength=5).tolist(),student_actions=np.bincount(pred[m],minlength=5).tolist())
    return result


def run(args):
    if args.weights.exists() or args.report.exists() or args.weights.resolve()==args.report.resolve():
        raise FileExistsError("fresh distinct products required")
    if any(k.startswith(("BC_","CNET2_")) for k in os.environ):
        raise ValueError("unset BC_ and CNET2_ overrides")
    os.environ["BC_DEVICE"]="cpu"
    import cnet2_train as trainer
    start=time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    if sha(args.init)!=PARENT_SHA:
        raise ValueError("unexpected V3 parent")
    with np.load(args.init,allow_pickle=False) as f:
        meta=json.loads(str(f["meta"]))
        preset,cfg,count=resolve_preset_metadata(meta)
        if preset!="sapling-1.2m" or count!=1206476:
            raise ValueError("incompatible single network")
        net=CNet2(**cfg)
        net.load_state_dict({k:torch.as_tensor(f[k]) for k in f.files if k!="meta"},strict=True)
    soft,sg=corpus(args.soft,trainer,True)
    elite,eg=corpus(args.elite,trainer,False)
    if {g["seed"] for g in sg}&{g["seed"] for g in eg}:
        raise ValueError("soft and public replay sources share maps")
    if any(len(ds["train"]["y"])<size for ds,size in ((soft,SOFT_BATCH),(elite,ELITE_BATCH))):
        raise ValueError("insufficient roots")
    for name,p in net.named_parameters():
        if name.startswith(("value.","utility.","wdl.","margin.","aux.")):
            p.requires_grad_(False)
    def measure():
        return {"soft":evaluate(net,trainer,soft["val"],True),"top5":evaluate(net,trainer,elite["val"],False)}
    before=measure()
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=LR,weight_decay=1e-4)
    net.train()
    ds,da=soft["train"],elite["train"]
    sfree,afree=free_slots(ds),free_slots(da)
    sden=SOFT_BATCH*sfree.sum()/len(sfree)
    aden=ELITE_BATCH*afree.sum()/len(afree)
    history=[]
    for step in range(UPDATES):
        si=np.sort(rng.choice(len(ds["y"]),SOFT_BATCH,replace=False))
        ai=np.sort(rng.choice(len(da["y"]),ELITE_BATCH,replace=False))
        batch={k:np.concatenate([ds[k][si],da[k][ai]]) for k in trainer.LOAD}
        z=logits(net,trainer,batch)
        q=torch.from_numpy(ds["q"][si])
        kl=divergence(z[:SOFT_BATCH],q)[torch.from_numpy(sfree[si])].sum()/sden
        lab=torch.as_tensor(da["y"][ai],dtype=torch.long)
        ce=nnf.cross_entropy(z[SOFT_BATCH:].transpose(1,2),lab,ignore_index=-1,reduction="none")
        ce=ce[torch.from_numpy(afree[ai])].sum()/aden
        loss=SOFT_WEIGHT*kl+ELITE_WEIGHT*ce
        if not torch.isfinite(loss):
            raise ValueError("nonfinite update")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
        opt.step()
        history.append([float(kl.detach()),float(ce.detach())])
        if (step+1)%32==0:
            print(f"{step+1}/{UPDATES} soft {history[-1][0]:.5f} elite {history[-1][1]:.5f}",flush=True)
    after=measure()
    recipe=dict(parent_sha256=PARENT_SHA,soft_manifest_sha256=sha(args.soft),elite_manifest_sha256=sha(args.elite),
                trainer_sha256=sha(__file__),learning_rate=LR,optimizer="AdamW",weight_decay=.0001,clip_grad=5.,
                seed=SEED,updates=UPDATES,soft_roots_per_update=SOFT_BATCH,elite_roots_per_update=ELITE_BATCH,
                soft_kl_weight=SOFT_WEIGHT,elite_hard_ce_weight=ELITE_WEIGHT,
                sampled_train_elite_roots_per_game_cap=64,sampled_validation_elite_roots_per_game_cap=40,
                sampling="Uniform roots without reference to labels; all slots with at least two legal options. "
                         "Each source loss is normalized by its expected number of slots in a batch. No data augmentation.",
                soft_evaluation="Mean forward KL and teacher-mode exact agreement; y is the behavior's executed prefix.",
                elite_evaluation="Cross entropy and exact agreement with the actual Top Five move, causal expert prefix.",
                soft_train_map_seeds=[g["seed"] for g in sg if g["split"]=="train"],
                soft_validation_map_seeds=[g["seed"] for g in sg if g["split"]=="val"])
    args.weights.parent.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.weights.parent) as wdir,tempfile.TemporaryDirectory(dir=args.report.parent) as rdir:
        w=Path(wdir)/"model.npz"
        trainer.export(net,str(w),dict(cfg=cfg,preset=preset,model_name="Sapling",params=count,untrained=False,
                       training_status="it50_soft_distillation",value_heads_untrained=True,soft_it50=recipe))
        out=dict(recipe=recipe,parameters=count,trained_sha256=sha(w),before=before,after=after,
                 train_losses_soft_kl_and_elite_ce=history,runtime=dict(seconds=time.monotonic()-start,
                 threads=args.threads,torch=str(torch.__version__),numpy=np.__version__),selected_update=UPDATES)
        r=Path(rdir)/"report.json"
        r.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
        os.link(r,args.report)
        try:
            os.link(w,args.weights)
        except BaseException:
            args.report.unlink()
            raise
    print(out["trained_sha256"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("soft","elite","weights","report"):
        parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--init",type=Path,default=WEIGHT)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    if args.threads<1:
        parser.error("positive --threads required")
    run(args)
