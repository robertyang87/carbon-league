"""Capacity control for frozen V5: append six identity-initialized residual blocks and repeat the exact recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as nnf

import sapling_soft_it50 as v5
from cnet2_model import CNet2
from init_sapling import resolve_preset_metadata
from sapling_cash_train import free_slots

SMALL_TRAINER_SHA = "364f8c93967af926350ee77df3e6357bf425f34da16ac5d043af86fa83141fb5"
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/sapling_1200k"
WEIGHTS = ROOT / "data/weights/sapling_1200k"
ARCH = "sapling-experimental-c93-nb12-deepening-v1"


def construct(checkpoint):
    if v5.sha(checkpoint) != v5.PARENT_SHA:
        raise ValueError("V3 parent changed")
    with np.load(checkpoint,allow_pickle=False) as z:
        meta=json.loads(str(z["meta"]))
        preset, cfg, count=resolve_preset_metadata(meta)
        state={k:torch.from_numpy(z[k]) for k in z.files if k!="meta"}
    if preset!="sapling-1.2m" or count!=1206476 or cfg["NB"]!=6 or cfg["TA"]!=0:
        raise ValueError("incompatible parent architecture")
    small=CNet2(**cfg)
    small.load_state_dict(state,strict=True)
    deep=CNet2(**dict(cfg,NB=12))
    missing,unexpected=deep.load_state_dict(state,strict=False)
    expected={k for k in deep.state_dict() if k.startswith(tuple(f"blocks.{i}." for i in range(6,12)))}
    if set(missing)!=expected or unexpected or not all(torch.equal(deep.state_dict()[k],v) for k,v in state.items()):
        raise ValueError("port changed an original tensor or skipped a non-added tensor")
    with torch.no_grad():
        for block in deep.blocks[6:]:
            block.n2.weight.zero_()
            block.n2.bias.zero_()
    if sum(p.numel() for p in deep.parameters())!=2143916:
        raise ValueError("wrong parameter count")
    return small.eval(), deep.eval()


@torch.inference_mode()
def verify_identity(small,deep,soft,elite,trainer):
    details={}
    for name,data in (("soft",soft),("elite",elite)):
        ds=data["val"]
        sample=np.linspace(0,len(ds["y"])-1,32,dtype=int)
        b=trainer.to_dev(trainer.fetch(ds,sample),False)
        za=small.forward_heads(b)
        zb=deep.forward_heads(b)
        errors=[float((a-b).abs().max()) for a,b in zip(za,zb)]
        if any(errors):
            raise ValueError(("initial equivalence failure",name,errors))
        details[name]=dict(roots=len(sample),legal_slots=int(free_slots(trainer.fetch(ds,sample)).sum()),
                           max_abs_policy_value_aux_utility_wdl_margin=errors)
    return details


def profile_model(net,b):
    """Compare counted convolution/matmul FLOPs on identical inputs and thread count."""
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                        with_flops=True) as prof:
        net(b)
    counted={evt.key:int(evt.flops) for evt in prof.key_averages() if evt.flops}
    if not any("conv" in k for k in counted):
        raise ValueError("profiler did not count convolution FLOPs")
    return dict(counted_by_op=counted,total=sum(counted.values()))


def export(net,out,step):
    import cnet2_train as trainer
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        local=Path(tmp)/"raw.npz"
        trainer.export(net,str(local),dict(cfg=dict(net.cfg),model_name="SaplingExperimental",
                       architecture_id=ARCH,parent_preset="sapling-1.2m",params=2143916,untrained=False,
                       training_status="same_recipe_as_soft_it50_v5",value_heads_untrained=True,
                       parent_sha256=v5.PARENT_SHA,selected_update=step,capacity_trainer_sha256=v5.sha(__file__)))
        os.link(local,out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--soft",type=Path,required=True)
    p.add_argument("--elite",type=Path,required=True)
    p.add_argument("--weights",type=Path,default=WEIGHTS/"soft_it50_deep214.npz")
    p.add_argument("--report",type=Path,default=RUN/"capacity_ab_214_training.json")
    p.add_argument("--progress",type=Path,required=True)
    a=p.parse_args()
    if any(k.startswith(("BC_","CNET2_")) for k in os.environ):
        raise ValueError("unset all BC_ and CNET2_ overrides")
    if a.weights.exists() or a.report.exists() or a.progress.exists():
        raise FileExistsError("fresh outputs required")
    os.environ["BC_DEVICE"]="cpu"
    import cnet2_train as trainer
    torch.set_num_threads(1)
    torch.manual_seed(v5.SEED)
    t0=time.monotonic()
    rng=np.random.default_rng(v5.SEED)
    small,deep=construct(v5.WEIGHT)
    control=json.loads((RUN/"soft_it50_v5_training.json").read_text())
    if (v5.sha(Path(v5.__file__))!=control["recipe"]["trainer_sha256"] or v5.sha(Path(v5.__file__))!=SMALL_TRAINER_SHA
            or v5.sha(a.soft)!=control["recipe"]["soft_manifest_sha256"] or v5.sha(a.elite)!=control["recipe"]["elite_manifest_sha256"]
            or v5.sha(WEIGHTS/"soft_it50_v5.npz")!=control["trained_sha256"] or control["selected_update"]!=192):
        raise ValueError("V5 control or frozen recipe differs")
    soft,_=v5.corpus(a.soft,trainer,True)
    elite,_=v5.corpus(a.elite,trainer,False)
    ds,dh=soft["train"],elite["train"]
    parity=verify_identity(small,deep,soft,elite,trainer)
    b=trainer.to_dev(trainer.fetch(ds,np.arange(4)),False)
    flops={name:profile_model(n,b) for name,n in (("v5",small),("deep",deep))}
    flop_ratio=flops["deep"]["total"]/flops["v5"]["total"]
    mid_step=round(v5.UPDATES/flop_ratio)
    del small,b
    for name,param in deep.named_parameters():
        if name.startswith(("value.","utility.","wdl.","margin.","aux.")):
            param.requires_grad_(False)
    def measure():
        return {k:v5.evaluate(deep,trainer,d["val"],soft_labels) for k,d,soft_labels
                in (("soft",soft,True),("top5",elite,False))}
    deep.train()
    opt=torch.optim.AdamW([p for p in deep.parameters() if p.requires_grad],lr=v5.LR,weight_decay=1e-4)
    ms,mh=free_slots(ds),free_slots(dh)
    dsden=v5.SOFT_BATCH*ms.sum()/len(ms)
    dhden=v5.ELITE_BATCH*mh.sum()/len(mh)
    digest=hashlib.sha256()
    history=[]
    train_wall=train_cpu=0.
    mid=None
    w,c=time.monotonic(),time.process_time()
    with a.progress.open("x") as progress:
        for step in range(1,v5.UPDATES+1):
            si=np.sort(rng.choice(len(ds["y"]),v5.SOFT_BATCH,replace=False))
            hi=np.sort(rng.choice(len(dh["y"]),v5.ELITE_BATCH,replace=False))
            for x in (si,hi):
                digest.update(x.astype("<i8").tobytes())
            batch={k:np.concatenate([ds[k][si],dh[k][hi]]) for k in trainer.LOAD}
            z=v5.logits(deep,trainer,batch)
            kl=v5.divergence(z[:v5.SOFT_BATCH],torch.from_numpy(ds["q"][si]))[torch.from_numpy(ms[si])].sum()/dsden
            ce=nnf.cross_entropy(z[v5.SOFT_BATCH:].transpose(1,2),torch.as_tensor(dh["y"][hi],dtype=torch.long),
                                 ignore_index=-1,reduction="none")[torch.from_numpy(mh[hi])].sum()/dhden
            loss=v5.SOFT_WEIGHT*kl+v5.ELITE_WEIGHT*ce
            if not torch.isfinite(loss):
                raise ValueError("nonfinite update")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(deep.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            history.append([float(kl.detach()),float(ce.detach())])
            if step==1 and not np.allclose(history[0],control["train_losses_soft_kl_and_elite_ce"][0],rtol=1e-5,atol=1e-6):
                raise ValueError("control step-1 loss does not match deep candidate")
            if step%16==0 or step in (1,mid_step,v5.UPDATES):
                progress.write(json.dumps(dict(step=step,loss=history[-1],elapsed=time.monotonic()-t0))+"\n")
                progress.flush()
            if step==mid_step:
                train_wall+=time.monotonic()-w
                train_cpu+=time.process_time()-c
                mid=dict(step=step,measured=measure(),training_seconds=train_wall,training_cpu_seconds=train_cpu)
                deep.train()
                w,c=time.monotonic(),time.process_time()
    train_wall+=time.monotonic()-w
    train_cpu+=time.process_time()-c
    metrics=measure()
    export(deep,a.weights,v5.UPDATES)
    result=dict(architecture_id=ARCH,parent=v5.PARENT_SHA,original_parameter_count=1206476,experimental_parameter_count=2143916,
                original_blocks=6,experimental_blocks=12,same_graph_except_extra_blocks=True,
                new_blocks="ordinary trainable CNet2 residual blocks with zero-initialized final GroupNorm scale and bias",
                original_tensor_identity="every named tensor exactly equal on port",
                initial_logits_and_heads=parity,recipe_reference=control["recipe"],original_weight=control["trained_sha256"],
                selected_update=v5.UPDATES,trained_sha256=v5.sha(a.weights),
                batch_indices_sha256=digest.hexdigest(),batch_indices_format="V5 RNG choice; each sorted int64 array little endian, soft then elite, for each update",
                original_step1_loss=control["train_losses_soft_kl_and_elite_ce"][0],deep_training_losses=history,
                flop_measurement="counted forward FLOPs, batch=4 for both, same inputs; comparisons approximate backward compute and exclude data/scheduling cost",
                flops=flops,forward_flop_ratio=flop_ratio,compute_matched_snapshot=mid,after=metrics,
                v5_after=control["after"],runtime=dict(threads=1,wall_total_seconds=time.monotonic()-t0,
                wall_training_seconds=train_wall,cpu_training_seconds=train_cpu,original_training_plus_eval_reported_seconds=control["runtime"]["seconds"],
                original_threads=control["runtime"]["threads"],torch=torch.__version__,numpy=np.__version__))
    with a.report.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(a.weights,result["trained_sha256"],flop_ratio,mid_step,flush=True)


if __name__=="__main__":
    main()
