"""Give each training root in the frozen Top Five corpus one decoder update exposure."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as fn

from cnet2_model import CNet2
from init_sapling import resolve_preset_metadata
from sapling_bootstrap import sha
from sapling_elite_bc import load_games
from sapling_soft_it50 import ELITE_SHA, PARENT_SHA
from sapling_cash_train import free_slots

SEED = 551218
DECODER = ("h0.", "aemb.", "din.", "gru.", "U.", "V.", "bhead.", "whead.")


@torch.no_grad()
def score(net, data, rows, trainer):
    net.eval()
    result = {}
    for start in range(0, len(rows), 32):
        selection = rows[start:start+32]
        batch = trainer.to_dev(trainer.fetch(data, selection), aug=False)
        z = net(batch)[0]
        target = batch["y"]
        valid = (target>=0) & (batch["legal"].sum(-1)>1)
        ce = fn.cross_entropy(z.transpose(1,2), target, ignore_index=-1, reduction="none")
        for i, name in enumerate(("collector", "planter", "base")):
            mask = valid & (batch["tok_kind"][:,:target.shape[1]]==i)
            totals = result.setdefault(name,[0,0,0.])
            totals[0] += int(mask.sum())
            totals[1] += int(((z.argmax(-1)==target)&mask).sum())
            totals[2] += float(ce[mask].sum())
    for name, (count, correct, ce_sum) in list(result.items()):
        result[name]=dict(n=count, correct=correct, mean_ce=ce_sum/count if count else None)
    return result


def run(args):
    if sha(args.init)!=PARENT_SHA or sha(args.manifest)!=ELITE_SHA:
        raise ValueError("unexpected frozen parent or Top Five split")
    if args.out.exists() or args.report.exists():
        raise FileExistsError("output already exists")
    os.environ["BC_DEVICE"]="cpu"
    import cnet2_train as trainer
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    data,games,manifest_hash=load_games(args.manifest,trainer)
    if len({g["seed"] for g in games})!=len(games):
        raise ValueError("training and validation maps must be distinct")
    with np.load(args.init,allow_pickle=False) as archive:
        meta=json.loads(str(archive["meta"]))
        preset,cfg,nparam=resolve_preset_metadata(meta)
        net=CNet2(**cfg)
        net.load_state_dict({k:torch.from_numpy(archive[k]) for k in archive.files if k!="meta"},strict=True)
    if preset!="sapling-1.2m" or nparam!=1206476: raise ValueError("unexpected architecture")
    for key,param in net.named_parameters():
        param.requires_grad_(key.startswith(DECODER))
    train,val=data["train"],data["val"]
    val_rows=np.sort(np.concatenate([rng.choice(np.flatnonzero(val["gid"]==game),32,replace=False)
                                     for game in np.unique(val["gid"])]))
    before=score(net,val,val_rows,trainer)
    permutation=rng.permutation(len(train["y"]))
    optimizer=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=1e-4,weight_decay=1e-4)
    history=[]
    seen=0
    free=free_slots(train)
    expected_slots=args.bs*free.sum()/len(permutation)
    net.train()
    start=time.monotonic()
    for idx in range(0,len(permutation),args.bs):
        indices=permutation[idx:idx+args.bs]
        b=trainer.to_dev(trainer.fetch(train,indices),aug=False)
        with torch.no_grad():
            tok,ctx,_=net.encode(b)
        z=net.decode(b,tok,ctx)
        mask=torch.from_numpy(free[indices])
        ce=fn.cross_entropy(z.transpose(1,2),b["y"],ignore_index=-1,reduction="none")
        loss=ce[mask].sum()/(expected_slots*len(indices)/args.bs)
        if not torch.isfinite(loss): raise ValueError("nonfinite cross entropy")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        seen+=int(mask.sum())
        if idx%(args.bs*100)==0:
            history.append([idx,float(loss.detach())])
            print(idx,len(permutation),history[-1][1],flush=True)
    if seen!=int(free.sum()): raise ValueError("incomplete training pass")
    after=score(net,val,val_rows,trainer)
    recipe=dict(parent=PARENT_SHA,dataset=manifest_hash,code=sha(__file__),seed=SEED,
                batch_size=args.bs,threads=args.threads,train_roots=len(permutation),train_slots=seen,
                train_games=sum(g["split"]=="train" for g in games),val_games=sum(g["split"]=="val" for g in games),
                validation_roots=len(val_rows),shuffle_sha256=__import__("hashlib").sha256(permutation.astype("<i8").tobytes()).hexdigest(),
                learnable=sum(p.numel() for p in net.parameters() if p.requires_grad),
                decoder=list(DECODER),optimizer="AdamW",lr=.0001,weight_decay=.0001,
                objective="unweighted actionable expert choices on their recorded prefixes; one complete shuffled epoch")
    trainer._check_finite_arrays({k:v.detach().numpy() for k,v in net.state_dict().items()},"trained full pass")
    meta.update(training_status="full_top5_decoder_pass",full_pass=recipe)
    temp=args.out.with_suffix(".tmp.npz")
    trainer.export(net,str(temp),meta)
    os.replace(temp,args.out)
    with args.report.open("x") as f:
        json.dump(dict(recipe=recipe,before=before,after=after,history=history,train_seconds=time.monotonic()-start,
                       weight_sha256=sha(args.out)),f,indent=2,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","init","out","report"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--bs",type=int,default=48)
    p.add_argument("--threads",type=int,default=1)
    run(p.parse_args())
