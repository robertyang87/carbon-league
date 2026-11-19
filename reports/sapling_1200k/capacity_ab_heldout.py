"""Compare the two frozen int8 packages on held-out real prefixes, keeping each source game intact."""
import argparse
import base64
import copy
import json
import os
from pathlib import Path
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as nnf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"code"))
os.environ["BC_DEVICE"]="cpu"
import cnet2_train as trainer
import sapling_soft_it50 as v5
from build_sapling_cash_dagger import module
from cnet2_model import CNet2
from sapling_cash_train import free_slots
from capacity_ab_inference import ARMS, sha


def unpack(path):
    p=module(path,"recovered_"+path.stem)
    weights=p._dequantize(zlib.decompress(base64.b64decode(p._W64)),p._QMETA)
    net=CNet2(**json.loads(p._WMETA)["cfg"])
    net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()},strict=True)
    return net.eval()


@torch.inference_mode()
def measure(nets,ds,soft,ids):
    arms={}
    valid=free_slots(ds)
    for name,net in nets.items():
        ce,hit=[],[]
        for off in range(0,len(ds["y"]),24):
            sl=np.arange(off,min(off+24,len(ds["y"])))
            z=v5.logits(net,trainer,trainer.fetch(ds,sl))
            label=torch.as_tensor(ds["label"][sl],dtype=torch.long)
            loss=(v5.divergence(z,torch.from_numpy(ds["q"][sl])) if soft else
                  nnf.cross_entropy(z.transpose(1,2),label,ignore_index=-1,reduction="none"))
            ce.append(loss.numpy())
            hit.append((z.argmax(-1)==label).numpy())
        ce,hit=np.concatenate(ce),np.concatenate(hit)
        def score(mask):
            n=int(mask.sum())
            return dict(slots=n,loss=float(ce[mask].mean(dtype=np.float64)),accuracy=float(hit[mask].mean()))
        parts={"all":score(valid)}
        if not soft:
            parts.update({k:score(valid&ds[k]) for k in ("attack","defense")})
        rows=[score(valid&(ds["gid"]==i)[:,None]) for i in ids]
        arms[name]=dict(slot_level=parts,per_game=rows)
    pairs=np.array([[a["loss"]-b["loss"],a["accuracy"]-b["accuracy"]]
                    for a,b in zip(arms["deep"]["per_game"],arms["v5"]["per_game"])])
    rng=np.random.default_rng(551220)
    boot=pairs[rng.integers(len(ids),size=(10000,len(ids)))].mean(axis=1)
    ablation=np.array([[a["loss"]-b["loss"],a["accuracy"]-b["accuracy"]]
                       for a,b in zip(arms["deep"]["per_game"],arms["deep_added_blocks_disabled"]["per_game"])])
    return dict(arms=arms,map_ids=[int(i) for i in ids],equal_game_weight_delta_loss_accuracy=pairs.mean(0).tolist(),
                deep_minus_post_training_residual_ablation=ablation.mean(0).tolist(),
                percentile_paired_map_bootstrap_95=np.quantile(boot,[.025,.975],axis=0).T.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--soft",type=Path,required=True)
    p.add_argument("--elite",type=Path,required=True)
    p.add_argument("--out",type=Path,default=HERE/"capacity_ab_int8_heldout.json")
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(1)
    nets={k:unpack(v) for k,v in ARMS.items()}
    stripped=copy.deepcopy(nets["deep"])
    scale_rms=[]
    with torch.no_grad():
        for b in stripped.blocks[6:]:
            scale_rms.append(float(b.n2.weight.square().mean().sqrt()))
            b.n2.weight.zero_()
            b.n2.bias.zero_()
    nets["deep_added_blocks_disabled"]=stripped
    out=dict(weights={k:sha(p) for k,p in ARMS.items()},estimand="Per-game mean difference deep minus V5 on real-prefix imitation metrics. Soft loss is forward KL to it50; elite loss is observed joint-prefix CE. Percentile bootstrap with 6 or 21 games is descriptive.")
    out.update(trained_residual_scale_rms=scale_rms,
               ablation="Diagnostic forward pass with six learned residual branches disabled after training. Original layers retain their deep-arm updates.")
    for key,path,soft in (("soft",a.soft,True),("elite",a.elite,False)):
        ds,games=v5.corpus(path,trainer,soft)
        val=ds["val"]
        ids=np.unique(val["gid"])
        if len(ids)!=sum(g["split"]=="val" for g in games):
            raise ValueError("per-map grouping not unique")
        out[key]=measure(nets,val,soft,ids)
        del ds,val
    with a.out.open("x") as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write("\n")
    print({k:out[k]["equal_game_weight_delta_loss_accuracy"] for k in ("soft","elite")})


if __name__=="__main__":
    main()
