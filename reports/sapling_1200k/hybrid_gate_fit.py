"""Fit a gated raw-logit blend on train maps and compare with constant 25% on held-out Top Five maps."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib

import numpy as np
import torch

from hybrid_gate_context import hybrid_categories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
os.environ["BC_DEVICE"] = "cpu"
import cnet2_feat as feat
import cnet2_train as trainer
from cnet2_model import CNet2
from build_sapling_cash_dagger import digest, module
from sapling_cash_train import checked_shard, free_slots
from sapling_tree_train import ELITE_SHA
from build_it50_v3 import INPUT_SHA256, IT50, STUDENT

GRID = np.arange(1, 19) / 20
PRIOR = .25
SEED = 551209
NROOT = 24
STRENGTH = 160
ROLES = ("collector", "planter", "base")


def network(path):
    if digest(path) != INPUT_SHA256[path]:
        raise ValueError("changed frozen pack")
    p = module(path, "gate_pack_" + path.stem)
    blob, qmeta, wmeta = (p._B1, p._Q1, p._M1) if path == IT50 else (p._W64, p._QMETA, p._WMETA)
    vals = p._dequantize(zlib.decompress(base64.b64decode(blob)), qmeta)
    net = CNet2(**json.loads(wmeta)["cfg"])
    missing, extra = net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in vals.items()},strict=False)
    if extra or not set(missing) <= {h+"."+k for h in ("wdl","utility","margin") for k in ("bias","weight")}:
        raise ValueError((missing, extra))
    return net.eval()


@torch.inference_mode()
def collect(manifest):
    if digest(manifest) != ELITE_SHA:
        raise ValueError("unexpected Top Five manifest")
    games=json.loads(manifest.read_text())["games"]
    if len({g["seed"] for g in games}) != 100 or sum(g["split"]=="train" for g in games)!=79:
        raise ValueError("unexpected split")
    nets=[network(p) for p in (IT50,STUDENT)]
    data={s:[] for s in ("train","val")}
    roots={}
    for game in games:
        local_seed = SEED ^ int(hashlib.sha256(game["id"].encode()).hexdigest()[:8],16)
        ix=np.sort(np.random.default_rng(local_seed).choice(game["rows"],NROOT,replace=False))
        roots[game["id"]]=ix.tolist()
        with checked_shard(manifest,game) as shard:
            d={k:shard[k][ix] for k in trainer.FIELDS}
        if not (d["teacher"]==2).all():
            raise ValueError("unexpected style")
        valid=free_slots(d)
        b=trainer.to_dev(d,False)
        e,s=[net(b)[0].numpy()[valid] for net in nets]
        g=hybrid_categories(d["tok_kind"],d["tok_pos"],d["tok_mask"],d["tok_cargo"],d["step"],feat.DIST)[valid]
        ys=d["y"][valid].astype(np.int64)
        legal=d["legal"][valid]
        if not np.isfinite(e).all() or not np.isfinite(s).all() or not legal[np.arange(len(ys)),ys].all():
            raise ValueError("nonfinite or illegal decision")
        group=game["split"]
        data[group].append(dict(e=e,s=s,y=ys,group=g,role=g//12,map=np.full(len(g),game["seed"])))
    return {k:{f:np.concatenate([d[f] for d in v]) for f in v[0]} for k,v in data.items()},roots


def metrics(d,a):
    e,s=d["e"],d["s"]
    a=np.broadcast_to(a,(len(e),))[:,None]
    z=(1-a)*e+a*s
    z=z-z.max(-1,keepdims=True)
    ll=np.log(np.exp(z).sum(-1))-z[np.arange(len(z)),d["y"]]
    action=z.argmax(-1)
    ex=e.argmax(-1)
    st=s.argmax(-1)
    disagree=ex!=st
    return dict(nll=ll,agree=(action==d["y"]),dis=disagree,student=disagree&(action==st),
                third=disagree&(action!=st)&(action!=ex))


def fit(d):
    ll=np.stack([metrics(d,a)["nll"] for a in GRID],axis=-1)
    counts=np.bincount(d["group"],minlength=36)
    # Penalize alpha departures from 0.25 in sparse cells. No validation rows
    # or branch accuracy are used to choose alphas.
    table=np.full(36,PRIOR)
    penalty=STRENGTH*(GRID-PRIOR)**2
    for k,n in enumerate(counts):
        if n:
            table[k]=GRID[np.argmin(ll[d["group"]==k].sum(axis=0)+penalty)]
    global_a=float(GRID[np.argmin(ll.sum(axis=0)+penalty)])
    return table,global_a,counts


def summarize(d,arr):
    result={}
    for name,mask in [("all",np.ones(len(d["y"]),bool))]+[(r,d["role"]==i) for i,r in enumerate(ROLES)]:
        result[name]=dict(n=int(mask.sum()),nll=float(arr["nll"][mask].mean()),
                          accuracy=float(arr["agree"][mask].mean()),disagreements=int(arr["dis"][mask].sum()),
                          student_picks=int(arr["student"][mask].sum()),third_picks=int(arr["third"][mask].sum()))
    return result


def heldout(d,base,cand):
    maps=[]
    for seed in np.unique(d["map"]):
        mask=d["map"]==seed
        maps.append(dict(map=int(seed),slots=int(mask.sum()),
                         dnll=float((cand["nll"][mask]-base["nll"][mask]).mean()),
                         dacc=float(cand["agree"][mask].mean()-base["agree"][mask].mean())))
    arr=np.asarray([[r["dnll"],r["dacc"]] for r in maps])
    rng=np.random.default_rng(SEED+1)
    boot=arr[rng.integers(len(arr),size=(10000,len(arr)))].mean(1)
    return dict(maps=maps,dnll=float(arr.mean(0)[0]),dacc=float(arr.mean(0)[1]),
                dnll_ci=np.quantile(boot[:,0],[.025,.975]).tolist(),dacc_ci=np.quantile(boot[:,1],[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    ds,roots=collect(a.manifest)
    table,global_a,counts=fit(ds["train"])
    results={}
    for split,d in ds.items():
        arms={k:metrics(d,v) for k,v in (("expert",0.),("student",1.),("constant_25",PRIOR),
                                          ("global_fit",global_a),("context_gate",table[d["group"]]))}
        results[split]={k:summarize(d,arr) for k,arr in arms.items()}
        if split=="val":
            pair=heldout(d,arms["constant_25"],arms["context_gate"])
            versus_global=heldout(d,arms["global_fit"],arms["context_gate"])
    base=results["val"]["constant_25"]
    gated=results["val"]["context_gate"]
    minimum_delta=min(gated[r]["accuracy"]-base[r]["accuracy"] for r in ROLES)
    frac=gated["all"]["student_picks"]/max(1,gated["all"]["disagreements"])
    promote=(pair["dnll"]<=-.005 and pair["dnll_ci"][1]<0 and pair["dacc"]>0
             and minimum_delta>=-.02 and frac>=.05 and gated["all"]["student_picks"]>=50)
    out=dict(schema="sapling-gated-blend/1",gate="held-out map-macro delta NLL <= -0.005 with paired-map bootstrap upper limit <0, delta accuracy >0, every slot-level role accuracy delta >= -0.02, student argmax chosen on >=5% of disagreements and >=50 choices",
             deployable=bool(promote),data_sha256=digest(a.manifest),weights={str(p.relative_to(ROOT)):digest(p) for p in INPUT_SHA256},
             seed=SEED,fixed_sample_rows=roots,slot_sampling="all legal choices on 24 uniformly sampled rows/game, both seats available",
             quantized_teacher_forced_logits=True,source="100 observed Top Five matches; 79 maps train and 21 historical development maps. No Master3s2 outcome is used.",
             fit=dict(student_alpha_grid=GRID.tolist(),shrinkage=STRENGTH,global_alpha=global_a,group_counts=counts.tolist(),student_table=table.tolist(),
                      cells="role collector/planter/base x stage 0..99/100..199/200..298 x visible enemy worker within toroidal distance 2 x collector holding >=30",
                      objective="sum next-action NLL + 160*(alpha-0.25)^2 per cell"),
             validation_pair_vs_constant=pair,validation_pair_vs_global=versus_global,metrics=results)
    with a.out.open("x") as fp:
        json.dump(out,fp,indent=2,allow_nan=False)
        fp.write("\n")
    print(out["deployable"],pair["dnll"],pair["dnll_ci"],pair["dacc"],minimum_delta,frac,flush=True)


if __name__=="__main__":
    main()
