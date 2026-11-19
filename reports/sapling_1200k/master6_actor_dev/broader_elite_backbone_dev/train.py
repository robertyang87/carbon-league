"""One fixed full-policy-backbone update pass on observed expert decisions."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as nnf

from freeze_data import HERE, ACTOR, START, PARENT, SHARDS, load, sha, save

def imported(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
ref=imported(ACTOR/"blended_actor_dev/fit.py","broader_reference")
common=ref.common
sys.path.insert(0,str(ACTOR/"blended_actor_dev/serving"))
import build as pack
import probe

def assemble():
    panel=load(HERE/"panel.json");cards=panel["games"];shards=load(HERE/"shards.json")
    if shards["panel"]!=sha((HERE/"panel.json").read_bytes()):raise ValueError("wrong shards")
    bank={r["id"]:r for r in shards["games"]}
    return [dict(g,shard_sha256=bank[g["id"]]["shard_sha256"],rows=bank[g["id"]]["rows"]) for g in cards]

def freeze():
    p=load(HERE/"protocol.json")
    save(HERE/"recipe.json",dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()),
        trainer_sha256=sha(Path(__file__).read_bytes()), evaluator_sha256=sha((HERE/"serve.py").read_bytes()),
        data_sha256=p["panel_sha256"], common_sha256=sha(Path(common.__file__).read_bytes()),
        ref_sha256=sha(Path(ref.__file__).read_bytes()), deployment_builder_sha256=sha(Path(pack.__file__).read_bytes()),
        probe_sha256=sha(Path(probe.__file__).read_bytes()), loss="Per root role mean .75 plus pooled .25, uniform per map and populated phase; one seeded shuffled pass",
        parent_reference="Frozen FP32 parent for strict legal-action forward KL and its deployed int8 reconstruction for held-out behavior",
        training="One optimizer update per batch; backbone/input embeddings/context and action decoder/heads train together. Fixed final step only.",
        offline_scaleup="Validation pure deployment-expert agreement must increase; test reports signed changes. A match/strength improvement can only be made with an original reacting examiner."))

def check():
    r=load(HERE/"recipe.json"); p=load(HERE/"protocol.json")
    if (sha(Path(__file__).read_bytes())!=r["trainer_sha256"] or sha((HERE/"protocol.json").read_bytes())!=r["protocol_sha256"]
        or sha((HERE/"panel.json").read_bytes())!=p["panel_sha256"] or sha(PARENT.read_bytes())!=p["parent_deployed_sha256"]
        or sha(Path(common.__file__).read_bytes())!=r["common_sha256"] or sha(Path(ref.__file__).read_bytes())!=r["ref_sha256"]
        or sha(Path(pack.__file__).read_bytes())!=r["deployment_builder_sha256"]):raise ValueError("changed recipe component")
    return p

def logits(net,b):return net(b)[0]

def mean_roles(cell, m, kinds):
    n=m.sum(1)
    raw=(cell*m).sum(1)/n.clamp(min=1)
    roles=torch.zeros_like(raw); present=torch.zeros_like(raw)
    for role in (0,1,2):
        rr=m & (kinds==role); count=rr.sum(1)
        roles+=(cell*rr).sum(1)/count.clamp(min=1)
        present+=(count>0)
    return .75*roles/present.clamp(min=1)+.25*raw

def policy_terms(z,y):
    return nnf.cross_entropy(z.transpose(1,2),y,ignore_index=-1,reduction="none")

def train():
    p=check();cards=assemble();train_cards=[g for g in cards if g["split"]=="train"]
    torch.manual_seed(p["seed"]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    oracle=ref.expert_model().eval();init,meta=common.model(START,p["parent_npz_sha256"]);init.eval().requires_grad_(False)
    net=copy.deepcopy(init)
    frozen=("aux.","utility.","value.","wdl.","margin.")
    for name,tensor in net.named_parameters():tensor.requires_grad_(not name.startswith(frozen))
    trainable={k:v for k,v in net.named_parameters() if v.requires_grad}
    if not any(k.startswith("blocks.") for k in trainable):
        if not any("conv" in k or k.startswith(("backbone.","res.")) for k in trainable):raise ValueError("backbone absent from optimizer")
    data=ref.load(train_cards,SHARDS)
    # Load utility rejects illegal roots; these two auxiliary placeholders contain no future information.
    if np.any(data["value"]) or np.any(data["carbon"]):raise ValueError("real privileged targets in actor loader")
    weights=np.zeros(len(data["step"]),np.float32)
    for g in train_cards:
        per_game=data["gid"]==g["gid"]
        phases=np.minimum(3,data["step"]//75)
        distinct=np.unique(phases[per_game])
        for phase in distinct:
            mask=per_game & (phases==phase)
            weights[mask]=len(weights)/(len(train_cards)*len(distinct)*int(mask.sum()))
    if not np.isclose(weights.mean(),1.,atol=1e-6) or np.any(weights<=0):raise ValueError("map/phase weights")
    rng=np.random.default_rng(p["seed"]);order=rng.permutation(len(weights))
    opt=torch.optim.AdamW(trainable.values(),lr=p["lr"],weight_decay=.001)
    log=[]; start=time.monotonic()
    for pos in range(0,len(order),p["batch"]):
        indices=order[pos:pos+p["batch"]];b=common.batch(data,indices)
        y=b["y"];mask=common.actionable(b);role=b["tok_kind"][:,:y.shape[1]]
        with torch.no_grad():
            parent=logits(init,b).log_softmax(-1);q=parent.exp()
            e=logits(oracle,b)
            if (q.masked_select(~b["legal"] & mask[...,None])!=0).any():raise ValueError("illegal reference probability")
        student=logits(net,b); slp=student.log_softmax(-1)
        blend=.75*e+.25*student
        ce=mean_roles(policy_terms(student,y),mask,role)
        ce_m=mean_roles(policy_terms(blend,y),mask,role)
        kl=mean_roles((q*(parent-slp)).sum(-1),mask,role)
        w=torch.from_numpy(weights[indices]); terms=ce+.5*ce_m+.16*kl
        loss=(w*terms).mean()
        if not torch.isfinite(loss) or float(kl.detach().min()) < -1e-5:raise ValueError("invalid KL/loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(trainable.values(),1.,error_if_nonfinite=True);opt.step()
        log.append([float(ce.detach().mean()),float(ce_m.detach().mean()),float(kl.detach().mean()),float(norm)])
        if (len(log)%60)==0:
            print(f"train {len(log)}/{(len(weights)+15)//16}; {time.monotonic()-start:.1f}s; CE {np.mean(log[-60:],axis=0)}",flush=True)
    wpath=HERE/"candidate.npz"
    if wpath.exists():raise FileExistsError(wpath)
    tmp=HERE/"saving.npz"
    common.trainer.export(net,str(tmp),dict(meta,training_status="broader_elite_full_backbone",recipe_sha256=sha((HERE/"recipe.json").read_bytes())))
    os.replace(tmp,wpath)
    restored,_=common.model(wpath); ns=dict(net.named_parameters())
    start_params=dict(init.named_parameters());changes={}
    for k,w0 in start_params.items():
        v=ns[k].detach()
        if not torch.isfinite(v).all() or not torch.equal(v,dict(restored.named_parameters())[k]):raise ValueError("weight export differs")
        diff=(v-w0.detach()).abs(); key=k.split(".")[0];c=changes.setdefault(key,dict(params=0,changed=0,max=0.))
        c["params"]+=v.numel();c["changed"]+=int((diff!=0).sum());c["max"]=max(c["max"],float(diff.max()))
        if k.startswith(frozen) and c["changed"]:raise ValueError("frozen heads changed")
    if sum(m["changed"] for m in changes.values())<500_000:raise ValueError("insufficient backbone change")
    save(HERE/"fit.json",dict(recipe_sha256=sha((HERE/"recipe.json").read_bytes()),weight_sha256=sha(wpath.read_bytes()),
            init_sha256=p["parent_npz_sha256"],updates=len(log),train_roots=len(weights),phases=np.bincount(np.minimum(3,data["step"]//75)).tolist(),
            map_count=len(train_cards),trainable_parameters=sum(v.numel() for v in trainable.values()),
            changed_by_module=changes,first60_means=np.mean(log[:60],axis=0).tolist(),last60_means=np.mean(log[-60:],axis=0).tolist(),
            metric_order=["CE_student","CE_composite","forward_KL_parent_student","preclip_grad_norm"],elapsed_seconds=time.monotonic()-start,
            epoch_order_hash=sha(order.astype("<i8").tobytes()),empirical_match_strength_tested=False))
    base,old,expert_hash=pack.validate_baseline()
    with np.load(START,allow_pickle=False) as refnp: vals=pack.quantize(wpath,refnp)
    deployed=pack.substitute(base,vals)
    if pack.substitute(deployed,old)!=pack.substitute(base,old) or pack.literal(deployed,pack.STUDENT)!=vals:raise ValueError("deployment mismatch outside weight fields")
    with (HERE/"candidate.py").open("x") as f:f.write(deployed)
    common.pack.main(str(wpath),str(HERE/"standalone.py"),2)
    save(HERE/"package.json",dict(candidate_sha256=sha((HERE/"candidate.py").read_bytes()),standalone_sha256=sha((HERE/"standalone.py").read_bytes()),
        parent_sha256=p["parent_deployed_sha256"],weights_sha256=sha(wpath.read_bytes()),unchanged_expert=expert_hash,
        combo_weight_student=.25,student_int8_sha256=sha(pack.encode(vals)),candidate_bytes=len(deployed.encode()),standalone_bytes=(HERE/"standalone.py").stat().st_size))
    print("done",len(log),"updates",flush=True)

if __name__=="__main__":
    args=argparse.ArgumentParser();args.add_argument("mode",choices=["freeze","train"]);mode=args.parse_args().mode
    freeze() if mode=="freeze" else train()
