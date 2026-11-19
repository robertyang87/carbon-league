"""Update only V3's existing action-feature interaction layer using paired cash labels."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as fn

import cnet2_pack as packer
from cnet2_model import CNet2
from courier_train_labels import ELITE_SHA, ROOT, sha
from init_sapling import resolve_preset_metadata
from sapling_tree_train import PARENT_SHA

SEED = 551204
UPDATES = 240
BATCH = 96
TAU = 45.0


def activated(parent, trainer, src, labels):
    """Cache deployed-parent decoder states. Only U changes, so this stays exact for every update."""
    pieces = []
    marks = []
    hook = parent.U.register_forward_pre_hook(lambda module, inp: marks.append(inp[0]))
    try:
        with torch.inference_mode():
            for game in labels:
                g = game["id"]
                datafile = src.parent / (g + ".npz")
                if sha(datafile.read_bytes()) != game["shard_sha256"]:
                    raise ValueError("feature hash")
                r = game["rows"]
                indices = np.array([x["row"] for x in r],dtype=np.int64)
                # Additional outcome-blind anchors cover deployment phases omitted from courier labels.
                nanchors = 24
                with np.load(datafile, allow_pickle=False) as data:
                    anchor_rng = np.random.default_rng(int(g[:8],16)+SEED)
                    remaining = np.setdiff1d(np.arange(len(data["step"])),indices)
                    anchors = np.sort(anchor_rng.choice(remaining,nanchors,replace=False))
                    idx = np.r_[indices,anchors]
                    raw = {k:data[k][idx] for k in trainer.LOAD}
                for start in range(0,len(idx),20):
                    sl=slice(start,start+20)
                    b=trainer.to_dev({k:v[sl] for k,v in raw.items()},aug=False)
                    marks.clear()
                    logits = parent(b)[0]
                    if len(marks)!=12 or not torch.isfinite(logits).all():
                        raise ValueError("cached activations")
                    vectors = parent.V(b["mf"])   # [b,12,5,E]
                    hidden = torch.stack(marks,1)
                    basep = torch.softmax(logits,-1)
                    psize=len(logits)
                    q = np.zeros((psize,5),np.float32)
                    focus = np.full(psize,-1,np.int64)
                    checkmask = np.zeros((psize,5),bool)
                    for j in range(psize):
                        i=start+j
                        if i>=len(r):
                            continue
                        rr=r[i]; s=rr["slot"]
                        focus[j]=s
                        for act,out in rr["choices"].items():
                            q[j,int(act)]=out["q"]
                            checkmask[j,int(act)]=True
                        if not np.array_equal(raw["legal"][i,s],checkmask[j]):
                            raise ValueError("candidate legal mask changed")
                    payload={"z":logits,"h":hidden,"v":vectors,"p":basep,"q":torch.from_numpy(q),
                             "focus":torch.from_numpy(focus),"allowed":torch.from_numpy(checkmask),
                             "valid":(b["tok_mask"][:,:12] & (b["legal"].sum(-1)>1)),
                             "group":torch.full((psize,),len(pieces),dtype=torch.long)}
                    pieces.append(payload)
    finally:
        hook.remove()
    return {k:torch.cat([v[k] for v in pieces]) for k in pieces[0]}


def logits(res, d, batch):
    # With all other parameters frozen, this is exactly the result of adding res to U.
    return d["z"][batch]+(res(d["h"][batch]).unsqueeze(2)*d["v"][batch]).sum(-1)


def ref(d):
    n=len(d["q"]); row=torch.arange(n); focus=d["focus"].clamp(min=0)
    masked = d["q"].masked_fill(~d["allowed"],-1e7)
    gap = masked.max(-1).values - d["q"].masked_fill(~d["allowed"],1e7).min(-1).values
    meaningful=(d["focus"]>=0) & (gap>=10)
    target=torch.softmax((masked-masked.max(-1,keepdim=True).values)/TAU,-1)
    anchor_mask=d["valid"].clone()
    anchor_mask[row[meaningful],focus[meaningful]]=False
    return focus,meaningful,target,anchor_mask


def evaluate(res,d):
    s,meaning,_,anchor=ref(d)
    z=torch.cat([logits(res,d,ids) for ids in torch.arange(len(s)).split(64)])
    base=d["z"].argmax(-1)
    new=z.argmax(-1)
    rows=torch.nonzero(d["focus"]>=0).flatten()
    slot=s[rows]
    old=base[rows,slot]; alt=new[rows,slot]
    delta=d["q"][rows,alt]-d["q"][rows,old]
    picked=(alt!=old)
    probgain=((z.softmax(-1)[rows,slot]-d["p"][rows,slot])*d["q"][rows]).sum(-1)
    def summarize(mask):
        ds=delta[mask]
        return dict(roots=int(mask.sum()),changed=int((picked&mask).sum()),better=int((ds>.01).sum()),
                    worse=int((ds<-.01).sum()),mean_cash_plus_salvage_gain=float(ds.mean()),
                    mean_soft_gain=float(probgain[mask].mean()))
    return dict(all=summarize(torch.ones_like(picked)),informative=summarize(meaning[rows]),
                nonfocal_choice_agreement=float((base[anchor]==new[anchor]).float().mean()),
                nonfocal_free_slots=int(anchor.sum()),nonfocal_changed=int((base[anchor]!=new[anchor]).sum()))


def run(a):
    if any(x.exists() for x in (a.report,a.weights,a.pack)):
        raise FileExistsError("output already exists")
    os.environ["BC_DEVICE"]="cpu"
    import cnet2_train as trainer
    torch.set_num_threads(1); torch.manual_seed(SEED)
    if sha(a.weights_in.read_bytes()) != PARENT_SHA or sha(a.elite.read_bytes())!=ELITE_SHA:
        raise ValueError("wrong parent or split")
    with np.load(a.weights_in,allow_pickle=False) as data:
        meta=json.loads(str(data["meta"]))
        full={k:data[k] for k in data.files}
    preset,cfg,count=resolve_preset_metadata(meta)
    if count!=1206476 or preset!="sapling-1.2m" or packer.QMODE!="int8":
        raise ValueError("incompatible preset or quantizer")
    _,_,dequantized=packer.quantize(full)
    parent=CNet2(**cfg).eval()
    parent.load_state_dict({k:torch.from_numpy(v) for k,v in dequantized.items()})
    for p in parent.parameters():
        p.requires_grad_(False)
    label_raw=a.labels.read_bytes()
    games=json.loads(gzip.decompress(label_raw))["games"]
    train_maps={g["seed"] for g in games if g["split"]=="train"}
    val_maps={g["seed"] for g in games if g["split"]=="val"}
    if train_maps & val_maps or not train_maps or not val_maps:
        raise ValueError("map isolation")
    tr=activated(parent,trainer,a.elite,[g for g in games if g["split"]=="train"])
    focus,meaning,target,anchor=ref(tr)
    if meaning.sum()<80:
        raise ValueError("insufficient informative roots")
    model=torch.nn.Linear(parent.U.in_features,parent.U.out_features)
    torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.002)
    rng=np.random.default_rng(SEED)
    eligible=torch.nonzero(meaning).flatten().numpy()
    history=[]
    for it in range(UPDATES):
        ii=torch.from_numpy(rng.choice(eligible,BATCH,replace=True))
        jj=torch.from_numpy(rng.integers(0,len(focus),size=BATCH))
        p=logits(model,tr,ii)[torch.arange(BATCH),focus[ii]]
        ce=-(target[ii]*p.log_softmax(-1)).sum(-1).mean()
        za=logits(model,tr,jj)
        mask=anchor[jj]
        anchor_kl=fn.kl_div(za[mask].log_softmax(-1),tr["p"][jj][mask],reduction="batchmean")
        loss=ce+2.5*anchor_kl
        if not torch.isfinite(loss):
            raise ValueError("nonfinite gradient step")
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True)
        opt.step()
        if it % 40 == 0 or it==UPDATES-1:
            history.append(dict(update=it+1,ce=float(ce.detach()),anchor_kl=float(anchor_kl.detach())))
    recipe=dict(parent_sha256=PARENT_SHA,label_sha256=sha(label_raw),elite_manifest_sha256=ELITE_SHA,seed=SEED,
                fixed_updates=UPDATES,batch_roots=BATCH,q_temperature=TAU,meaningful_min_range=10,
                learning_rate=.0008,weight_decay=.002,anchors_per_game=24,nonfocal_forward_kl_weight=2.5,
                trainable=["U.weight","U.bias"],trainable_parameters=sum(p.numel() for p in model.parameters()),
                architecture_parameters=count,train_games=len(train_maps),validation_games=len(val_maps),
                trainer_sha256=sha(Path(__file__).read_bytes()))
    with torch.no_grad():
        trained=CNet2(**cfg)
        trained.load_state_dict({k:torch.from_numpy(v) for k,v in full.items() if k!="meta"})
        trained.U.weight.add_(model.weight);trained.U.bias.add_(model.bias)
        trainer.export(trained,str(a.weights),dict(cfg=cfg,preset=preset,params=count,model_name="Sapling",
                       training_status="cash_v3_paired_courier",untrained=False,value_heads_untrained=True,courier=recipe))
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp)/"agent.npz";shutil.copyfile(a.weights,temp)
            packer.main(str(temp),str(a.pack),teacher=2)
        # Candidate is frozen before opening validation features. Parents and students are
        # compared on exactly the same recorded prefixes; whole-game outcomes are separate.
        training=evaluate(model,tr)
        del tr
        val=activated(parent,trainer,a.elite,[g for g in games if g["split"]=="val"])
        validation=evaluate(model,val)
        outcome=dict(recipe=recipe,training=training,validation=validation,training_trace=history,
                     weights_sha256=sha(a.weights.read_bytes()),pack_sha256=sha(a.pack.read_bytes()),
                     claim="Conditional five-transition retrospective utility. Fixed logged continuations cannot substitute for paired whole-game results.")
    a.report.write_text(json.dumps(outcome,indent=2,allow_nan=False)+"\n")
    print(json.dumps(outcome["validation"],indent=2),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("elite","weights-in","labels","weights","pack","report"):
        cli.add_argument("--"+key,type=Path,required=True)
    run(cli.parse_args())
