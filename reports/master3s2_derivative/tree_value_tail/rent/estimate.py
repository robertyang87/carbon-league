"""Predict post-H8 original-holder receipts conditional on successful planting."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "ladder/tree_payback"
sys.path.insert(0, str(SOURCE))
import labels
import register
import runtime
import study

PROTOCOL = "4d6c633e76701e1e699cd3ac4a207aabfa8d95f68a2848583079c5d9c86f9e3e"
AMOUNTS = ("ground", "enemy_cargo", "direct_collision")
PREDICTORS = ("mean", "median", "ridge", "linear_raw")


def read_bound(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Identity mismatch {path.name}")
    return raw


def join_key(game, row):
    return (game, row["seat"], row["t"], row["tree"], row["cell"], row["planter"])


def recover(split, protocol):
    tickets = json.loads(read_bound(SOURCE/"maps.json", protocol["maps_sha256"]))["tickets"]
    raw = read_bound(SOURCE/f"{split}_trees.json.gz", protocol["feature_table_sha256"][split])
    rows = json.loads(gzip.decompress(raw))
    index = {join_key(r["game"],r): r for r in rows}
    selected = [t for t in tickets if t["split"]==split]
    if len(index)!=len(rows) or len(selected)!={"train":24,"held":8}[split]:
        raise ValueError("Duplicate feature rows or wrong group count")
    output=[]
    for ticket in selected:
        env, terrain = study.replay(ticket, register.CACHE)
        for r in labels.tree_rows(env):
            key = join_key(ticket["id"],r)
            old = index.pop(key)
            if terrain!=old["terrain"] or abs(r["net"]-old["net"])>1e-7 or len(old["X"])!=23:
                raise ValueError("Source identity, label or feature mismatch")
            start = r["t"]+8
            tail = [sum(c[j] for c in r["cash"] if c[0]>=start) for j in (1,2,3)]
            early = sum(sum(c[1:]) for c in r["cash"] if c[0]<start)
            if abs(early+sum(tail)-r["gross"])>1e-6 or min(tail)<-1e-8:
                raise ValueError("Partition of original cash failed")
            remaining = env["configuration"]["episodeSteps"]-1-r["t"]
            if remaining<=8 and sum(tail)>1e-8:
                raise ValueError("Late income beyond game horizon")
            output.append(dict(game=ticket["id"],terrain=terrain,seat=r["seat"],t=r["t"],
                               cell=r["cell"],tree=r["tree"],planter=r["planter"],cost=r["cost"],
                               early=early,exit=r["exit"],end_t=r["end_t"],net=r["net"],remaining=remaining,
                               usable=old["usable"],X=old["X"],late=dict(zip(AMOUNTS,tail)),target=sum(tail)))
    if index:
        raise ValueError("Registered plant lost in reconstruction")
    return output


def weighted_median(values, weights):
    order=np.argsort(values,kind="stable")
    return float(values[order][np.searchsorted(np.cumsum(weights[order]),0.5)])


def fit(rows):
    x=np.array([r["X"] for r in rows])
    y=np.array([r["target"] for r in rows])
    counts=Counter(r["terrain"] for r in rows)
    w=np.array([1/(len(counts)*counts[r["terrain"]]) for r in rows])
    if len(counts)!=24 or abs(w.sum()-1)>1e-10:
        raise ValueError("Unrepresented training terrain")
    centre=w@x
    scale=np.maximum(.05,np.sqrt(w@((x-centre)**2)))
    z=(x-centre)/scale
    intercept=float(w@y)
    beta=np.linalg.solve(z.T@(w[:,None]*z)+.1*np.eye(x.shape[1]),z.T@(w*(y-intercept)))
    return dict(protocol=PROTOCOL,features=runtime.FEATURES,mean=intercept,median=weighted_median(y,w),
                center=centre.tolist(),scale=scale.tolist(),beta=beta.tolist(),intercept=intercept)


def predict(row, head):
    z=(np.array(row["X"])-head["center"])/head["scale"]
    raw=float(head["intercept"]+z@head["beta"])
    live=row["remaining"]>8
    return dict(linear_raw=raw,ridge=max(0.,raw) if live else 0.,
                mean=head["mean"] if live else 0.,median=head["median"] if live else 0.)


def stats(rows):
    y=np.array([r["target"] for r in rows])
    errors={k:float(np.mean([abs(r["prediction"][k]-r["target"]) for r in rows])) for k in PREDICTORS}
    return dict(n=len(y),quantiles=dict(zip(("min","q25","median","q75","q90","max"),np.quantile(y,[0,.25,.5,.75,.9,1]))),
                zero=int((y<1e-8).sum()),first_tenure_net_negative=sum(r["net"]<-1e-8 for r in rows),
                direct_collision_positive=sum(r["late"]["direct_collision"]>0 for r in rows),
                components={k:sum(r["late"][k] for r in rows) for k in AMOUNTS},mae=errors)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("split",choices=("train","held"))
    parser.add_argument("--weight-sha")
    args=parser.parse_args()
    protocol=json.loads(read_bound(HERE/"protocol.json",PROTOCOL))
    read_bound(SOURCE/"labels.py",protocol["receipt_code_sha256"])
    if args.split=="held":
        head=json.loads(read_bound(HERE/"weights.json",args.weight_sha))
    recovered=recover(args.split,protocol)
    rows=[r for r in recovered if r["usable"]]
    if args.split=="train":
        head=fit(rows)
        weight_sha=study.dump(HERE/"weights.json",head)
    else:
        weight_sha=args.weight_sha
    for r in rows:
        r["prediction"]=predict(r,head)
    per_map={s:stats([r for r in rows if r["game"]==s]) for s in sorted({r["game"] for r in rows})}
    if len(per_map)!={"train":24,"held":8}[args.split]:
        raise ValueError("Missing usable terrain")
    macro={k:float(np.mean([s["mae"][k] for s in per_map.values()])) for k in PREDICTORS}
    strict={k:sum(s["mae"]["ridge"]<s["mae"][k] for s in per_map.values()) for k in ("mean","median")}
    counter={}
    measures=dict(over=lambda r:r["prediction"]["ridge"]-r["target"],under=lambda r:r["target"]-r["prediction"]["ridge"],
                  worse_than_median=lambda r:abs(r["prediction"]["ridge"]-r["target"])-abs(r["prediction"]["median"]-r["target"]))
    for key,func in measures.items():
        worst=max(rows,key=func)
        counter[key]={k:v for k,v in worst.items() if k!="X"}
    output=dict(split=args.split,protocol=PROTOCOL,weights_sha256=weight_sha,
                map_mean_mae=macro,map_improved=strict,distribution=stats(rows),
                by_map=per_map,excluded=len(recovered)-len(rows),counterexamples=counter,
                mean_of_map_medians=float(np.mean([v["quantiles"]["median"] for v in per_map.values()])),
                rows_sha256=study.dump(HERE/f"{args.split}_rows.json.gz",recovered))
    print("weights",weight_sha,args.split,study.dump(HERE/f"{args.split}.json",output),macro,strict,flush=True)


if __name__=="__main__":
    main()
