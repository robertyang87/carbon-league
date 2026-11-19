"""Fit an independent Sapling to official paired outcomes, with and without public proposer sampling."""
import importlib.util
import os
import sys
os.environ.setdefault("BC_DEVICE","cpu")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as nnf
from propose import HERE,ACTOR,ROOT,TREE,read,sha,hashb,check,dump,clean
spec=importlib.util.spec_from_file_location("pure_actor_tools",ACTOR/"train.py")
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PARENT="f82bfd47a904aa1fbfa4e95fbfcdd13e623c1f1eb216a8e736694a1d282cc713"


def predict(net,data,rows):
    q=base.predict(base.quantized(net),data)
    return np.array([[q[r["feature_index"],r["slot"],i] for i in r["pair_ids"]] for r in rows])


def choice(stat):
    return np.sign(stat[:,1]-stat[:,0]).astype(int)


def analysis(rows,scores):
    result={}
    for split in ("train","hold"):
        keep=np.array([r["split"]==split for r in rows])
        rs=[r for r in rows if r["split"]==split]
        seeds=sorted({r["seed"] for r in rs})
        delta=np.array([r["margin_delta"] for r in rs]);members=np.array([r["seed"] for r in rs]);truth=delta>0
        res={}
        for arm,ps in scores.items():
            s=ps[keep];choose_alt=choice(s)
            d=choose_alt.astype(float)*.5+.5
            actual=np.where(d==.5,.5,d)*delta
            regret=np.maximum(delta,0.)-actual
            permap={str(seed):float(regret[members==seed].mean()) for seed in seeds}
            res[arm]={"roots":len(s),"maps":len(seeds),"matches_terminal_preference":float(np.sum(np.where(choose_alt==0,.5,(choose_alt>0)==truth))),
                "map_equal_regret":float(np.mean(list(permap.values()))),"map_regret":permap,
                "alternative_preferred":int((choose_alt>0).sum()),"tie":int((choose_alt==0).sum()),
                "predicted_pair_probabilities":s.tolist(),"ids":[r["case"] for r in rs],
                "map_equal_actual_margin_difference_from_always_parent":float(np.mean([actual[members==seed].mean() for seed in seeds])),
                "alternative_or_v1_credit_only":"constrained two registered commands with other units and early prefix fixed"}
        result[split]=res
    held=result["hold"];c=held["proposal"];p=held["parent"];a=held["control"]
    new=choice(np.asarray(c["predicted_pair_probabilities"]));ctl=choice(np.asarray(a["predicted_pair_probabilities"]))
    checks={"mean_better_than_parent":c["map_equal_regret"]<p["map_equal_regret"],
           "mean_better_than_matched_control":c["map_equal_regret"]<a["map_equal_regret"],
           "parent_regret_reduced_on_three_maps":sum(c["map_regret"][k]<v for k,v in p["map_regret"].items())>=3,
           "two_distinct_held_choices_vs_control":int(np.count_nonzero(new!=ctl))>=2}
    return result,checks


def fit(data,rows,teacher):
    parent_path=base.WEIGHTS/"master6_actor_v1.npz"
    par,meta=base.model(parent_path,PARENT)
    root_train=np.array([r["feature_index"] for r in rows if r["split"]=="train"])
    keyed={r["feature_index"]:r for r in rows}
    rng=np.random.default_rng(620244);order=[]
    while len(order)<256:order.extend(rng.permutation(root_train))
    order=np.asarray(order[:256])
    if not root_train.size:raise ValueError("no legal pair")
    scores={"parent":predict(par,data,rows)};weights={}
    learn=("aemb.","din.","gru.","U.","V.","bhead.","whead.","h0.")
    for arm in ("control","proposal"):
        torch.manual_seed(620244);net,_=base.model(parent_path,PARENT);net.train()
        params=[]
        for k,w in net.named_parameters():
            w.requires_grad_(k.startswith(learn))
            if w.requires_grad:params.append(w)
        optimizer=torch.optim.AdamW(params,lr=.00008)
        for ix in order.reshape(32,8):
            b=base.batch(data,ix)
            token,ctx,_=net.encode(b);lg=net.decode(b,token,ctx)
            x=[keyed[int(i)] for i in ix]
            sl=np.array([r["slot"] for r in x])
            pair=np.array([r["pair_ids"] for r in x])
            pa=lg[np.arange(len(ix)),sl,pair[:,0]];alt=lg[np.arange(len(ix)),sl,pair[:,1]]
            sign=torch.tensor([1. if r["margin_delta"]>0 else -1. for r in x])
            weight=torch.tensor([1.+int(arm=="proposal" and r["nominates_registered_alternative"]) for r in x])
            supervised=(nnf.softplus(-sign*(alt-pa))*weight).sum()/weight.sum()
            valid=(b["y"]>=0)&(b["legal"].sum(-1)>1)
            prior=base.divergence(lg,torch.from_numpy(teacher[ix]))[valid].mean()
            loss=supervised+.35*prior
            check(bool(loss.isfinite()),"nonfinite loss")
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,2.,error_if_nonfinite=True);optimizer.step()
        f=HERE/(arm+".npz");base.trainer.export(net,str(f),dict(meta,terminal_pair_finetune=arm))
        weights[arm]={"sha256":sha(f),"parameters":sum(p.numel() for p in net.parameters()),
            "changed_vs_parent":sum((p.detach()!=dict(par.named_parameters())[k].detach()).sum().item() for k,p in net.named_parameters())}
        scores[arm]=predict(net.eval(),data,rows)
    return scores,weights,order


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    meta=read(HERE/"proposals.json.gz");z=HERE/"public_features.npz";check(sha(z)==meta["feature_sha256"],"stored feature drift")
    with np.load(z) as f:data={k:f[k] for k in f.files}
    rows=[r for r in meta["rows"] if "feature_index" in r]
    check(len(rows)==len(data["gid"]) and len(rows)>0 and len({r["feature_index"] for r in rows})==len(rows),"invalid pair features")
    parent_path=base.WEIGHTS/"master6_actor_v1.npz"
    par,_=base.model(parent_path,PARENT);teacher=base.predict(par,data)
    scores,weights,order=fit(data,rows,teacher);result,checks=analysis(rows,scores)
    serving=HERE/"sapling_only.py"
    base.pack.main(str(HERE/"proposal.npz"),str(serving),teacher=2)
    values={"proposals_sha256":sha(HERE/"proposals.json.gz"),"fit_source_sha256":sha(Path(__file__)),"weights":weights,
            "parent_sha256":PARENT,"order_sha256":hashb(order.tobytes()),"standalone_sha256":sha(serving),
            "result":result,"gate":checks,"eligible":all(checks.values())}
    dump(HERE/"fit.json",values)
    print(weights,checks,flush=True)


if __name__=="__main__":main()
