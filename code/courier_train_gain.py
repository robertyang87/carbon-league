"""Directly maximize paired return relative to the frozen V3 action distribution."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as fn

from courier_train_fit import (ELITE_SHA, PARENT_SHA, CNet2, activated, evaluate, logits, packer,
                               resolve_preset_metadata, sha)
from courier_train_labels import ROOT

SEED=551207
UPDATES=220
LR=.00018
BATCH=128


def main(a):
    paths=[a.weights,a.pack,a.report]
    if any(p.exists() for p in paths):
        raise FileExistsError("output already exists")
    os.environ["BC_DEVICE"]="cpu"
    import cnet2_train as trainer
    torch.set_num_threads(1);torch.manual_seed(SEED)
    if sha(a.weights_in.read_bytes())!=PARENT_SHA or sha(a.elite.read_bytes())!=ELITE_SHA:
        raise ValueError("source identity")
    with np.load(a.weights_in,allow_pickle=False) as f:
        full=dict(f)
    preset,cfg,count=resolve_preset_metadata(json.loads(str(full["meta"])))
    _,_,deq=packer.quantize(full)
    parent=CNet2(**cfg).eval()
    parent.load_state_dict({k:torch.from_numpy(v) for k,v in deq.items()})
    for p in parent.parameters():
        p.requires_grad_(False)
    primary=json.loads(gzip.decompress(a.labels.read_bytes()))
    train_games=[g for g in primary["games"] if g["split"]=="train"]
    tr=activated(parent,trainer,a.elite,train_games)
    focus=tr["focus"].clamp(min=0)
    index=torch.arange(len(focus))
    p0=tr["p"][index,focus]
    # Subtract the parent's expected value to put zero on its current distribution.
    # Use a common scale across roots. A cash difference of 1/10 should carry a tenth
    # of the gradient, even when it happens to reverse the best action.
    adv=((tr["q"]-(tr["q"]*p0).sum(-1,keepdim=True))/100).clamp(-4,4)
    adv=adv.masked_fill(~tr["allowed"],0)
    available=torch.nonzero(tr["focus"]>=0).flatten().numpy()
    anchors=tr["valid"].clone()
    anchors[available,focus[available]]=False
    res=torch.nn.Linear(parent.U.in_features,parent.U.out_features)
    torch.nn.init.zeros_(res.weight);torch.nn.init.zeros_(res.bias)
    opt=torch.optim.AdamW(res.parameters(),lr=LR,weight_decay=.001)
    rng=np.random.default_rng(SEED);hist=[]
    for it in range(UPDATES):
        ii=torch.from_numpy(rng.choice(available,BATCH,replace=True))
        jj=torch.from_numpy(rng.integers(0,len(index),BATCH))
        z=logits(res,tr,ii)[torch.arange(BATCH),focus[ii]]
        logp=z.log_softmax(-1);p=logp.exp()
        gain=((p-p0[ii])*adv[ii]).sum(-1).mean()
        trust=fn.kl_div(logp,p0[ii],reduction="batchmean")
        other=logits(res,tr,jj)[anchors[jj]]
        anchor=fn.kl_div(other.log_softmax(-1),tr["p"][jj][anchors[jj]],reduction="batchmean")
        loss=-gain+.5*trust+1.5*anchor
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(res.parameters(),1,error_if_nonfinite=True);opt.step()
        if it%40==0 or it+1==UPDATES:
            hist.append([it+1,*[float(v.detach()) for v in [gain,trust,anchor]]])
    recipe=dict(parent_sha256=PARENT_SHA,source_labels_sha256=sha(a.labels.read_bytes()),
                data_sha256=ELITE_SHA,previous_attempt="courier_train_fit",train_games=len(train_games),
                phase="fixed-budget direct paired payoff",seed=SEED,updates=UPDATES,batch=BATCH,lr=LR,
                advantage="(Q - E_parent[Q])/100, clipped to [-4,4]. Max E_candidate[advantage], "
                          "with 0.5*KL(parent||candidate) on courier slots + 1.5*KL on all other free slots.",
                trainable=["U.weight","U.bias"],trainable_parameters=sum(p.numel() for p in res.parameters()),
                trainer_sha256=sha(Path(__file__).read_bytes()))
    with torch.no_grad():
        student=CNet2(**cfg)
        student.load_state_dict({k:torch.from_numpy(v) for k,v in full.items() if k!="meta"})
        student.U.weight.add_(res.weight);student.U.bias.add_(res.bias)
        trainer.export(student,str(a.weights),dict(cfg=cfg,preset=preset,params=count,model_name="Sapling",
                       training_status="cash_v3_paired_courier_gain",untrained=False,value_heads_untrained=True,courier=recipe))
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp)/"student.npz";shutil.copyfile(a.weights,temp)
            packer.main(str(temp),str(a.pack),teacher=2)
        metrics=evaluate(res,tr)
        del tr
        # Second run gets fresh, previously unexamined maps. No checkpoint selection on these maps.
        held=json.loads(gzip.decompress(a.held.read_bytes()))
        if held["first_label_sha256"]!=recipe["source_labels_sha256"] or ({g["seed"] for g in primary["games"]}
              & {g["seed"] for g in held["games"]}) or len(held["games"])!=11:
            raise ValueError("reserve provenance")
        v=activated(parent,trainer,a.elite,held["games"])
        result=dict(recipe=recipe,training=metrics,validation=evaluate(res,v),validation_maps=len(held["games"]),
                    validation_labels_sha256=sha(a.held.read_bytes()),training_trace_gain_kl_anchor=hist,
                    weight_sha256=sha(a.weights.read_bytes()),pack_sha256=sha(a.pack.read_bytes()),
                    scope="Conditional five-step payoff under fixed surviving recorded commands. Exclusive reserve maps first evaluated after freezing second candidate. Not a match against Master3s2.")
        # Recover actionable, inspectable paired differences under identical subsequent commands.
        z=torch.cat([logits(res,v,ii) for ii in torch.arange(len(v["focus"])).split(64)])
        rows=torch.nonzero(v["focus"]>=0).flatten()
        s=v["focus"][rows]
        old=v["z"][rows,s].argmax(-1).tolist()
        new=z[rows,s].argmax(-1).tolist()
        where=[(g,r) for g in held["games"] for r in g["rows"]]
        if len(where)!=len(rows):
            raise ValueError("held-out row mismatch")
        examples=[]
        for (g,r),x,y in zip(where,old,new):
            if x==y:
                continue
            p=r["seat"]; a0=r["choices"][str(x)]; a1=r["choices"][str(y)]
            examples.append(dict(game=g["id"],seat=p,step=r["step"],unit=r["unit"],cargo=r["load"],
                                 parent_move=x,student_move=y,delta_five_step_value=a1["q"]-a0["q"],
                                 delta_bank_own=a1["cash"][p]-a0["cash"][p],delta_bank_opp=a1["cash"][1-p]-a0["cash"][1-p],
                                 delta_taken_from_enemy=a1["transfers"][p]-a0["transfers"][p],
                                 delta_given_to_enemy=a1["transfers"][1-p]-a0["transfers"][1-p]))
        result["all_heldout_changed_couriers"]=examples
        a.report.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(result["validation"]),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("elite","weights-in","labels","held","weights","pack","report"):
        cli.add_argument("--"+key,type=Path,required=True)
    main(cli.parse_args())
