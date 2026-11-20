"""Fit two frozen one-step planting readouts and score untouched maps."""

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from labels import tree_rows
import register
import runtime

HERE = Path(__file__).resolve().parent
PROTOCOL = "b92f8ad0e73320b2ebcfce3689012fa867884d1934ee94a6cf64793ecdca0ea1"
HAJIMI = "3f0a1dfa0e1ac4a00a9f63f9cf644af9-1"
EXPOSED_CELLS = {(246,214), (246,56), (255,89), (259,134), (261,120)}
sys.path.insert(0,str(register.ROOT/"reports/sapling_1200k/all_top5"))
from context import Game


def dump(path,data):
    raw = (json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    if path.suffix==".gz":
        raw=gzip.compress(raw,mtime=0)
    path.write_bytes(raw)
    return register.sha(raw)


def replay(ticket, cache):
    if "source" in ticket:
        local=cache/(ticket["id"]+".json.gz")
        if local.is_file():
            raw=local.read_bytes()
        else:
            with tarfile.open(fileobj=io.BytesIO(register.blob(ticket["source"]["archive"])),mode="r:*") as archive:
                raw=archive.extractfile(ticket["source"]["member"]).read()
    else:
        raw=(HERE.parent/"replays"/(ticket["id"]+".json.gz")).read_bytes()
    if register.sha(raw)!=ticket["compressed_sha256"]:
        raise ValueError(f"Changed replay {ticket['id']}")
    data=json.loads(gzip.decompress(raw))
    terrain=register.sha(bytes(data["steps"][0][0]["observation"]["full_carbon"]))
    if ticket.get("terrain",terrain)!=terrain:
        raise ValueError("Opening terrain does not match registration")
    return data,terrain


def collect(tickets,cache):
    output=[]
    for ticket in tickets:
        data,terrain=replay(ticket,cache)
        game=Game(data)
        receipts=tree_rows(data)
        frames=data["steps"]
        cfg=data["configuration"]
        for receipt in receipts:
            t,seat=receipt["t"],receipt["seat"]
            # Build the public observation and past rosters in isolation. Labels stay on the other side.
            public=game._obs(t,seat)
            past={k:game._obs(t-k,seat)["players"] for k in (3,5) if t>=k}
            actor=receipt["planter"]
            if public["players"][seat][2][actor][0]!=receipt["cell"]:
                raise ValueError("Plant location disagrees with public worker location")
            X=runtime.features(public,actor,cfg,past)
            price=runtime.quoted_price(public["players"],seat,cfg)
            if abs(price-receipt["cost"])>1e-7:
                raise ValueError("Public quotation mismatch")
            retained={k:receipt[k] for k in ("t","seat","cell","tree","planter","cost","gross","net","exit","end_t","break_even_t")}
            if receipt["exit"]=="early_terminal":
                # Both agents must have DONE without faults and at least one official economic extinction.
                end_players=frames[-1][0]["observation"]["players"]
                known=any(not player[2] and player[0]<min(cfg["recCollectorCost"],cfg["recPlanterCost"]) for player in end_players)
            else:
                known=True
            output.append(dict(game=ticket["id"],terrain=terrain,usable=known,nonpayback=int(receipt["net"] < -1e-8),X=X,**retained))
    return output


def fit(rows,nfeat):
    x=np.asarray([r["X"][:nfeat] for r in rows],dtype=float)
    y=np.asarray([r["nonpayback"] for r in rows],dtype=float)
    counts=Counter(r["terrain"] for r in rows)
    sample=np.asarray([1/len(counts)/counts[r["terrain"]] for r in rows])
    mean=sample@x
    scale=np.maximum(0.05,np.sqrt(sample@((x-mean)**2)))
    z=(x-mean)/scale
    def loss(w):
        logits=w[0]+z@w[1:]
        diff=sample*(expit(logits)-y)
        cost=sample@(np.logaddexp(0,logits)-y*logits)+0.01*(w[1:]@w[1:])
        return cost,np.r_[sum(diff),z.T@diff+0.02*w[1:]]
    result=minimize(loss,np.zeros(nfeat+1),jac=True,method="L-BFGS-B",options={"maxiter":200,"ftol":1e-12})
    if not result.success:
        raise ValueError(str(result.message))
    return dict(features=runtime.FEATURES[:nfeat],feature_mean=mean.tolist(),feature_scale=scale.tolist(),
                weights=result.x[1:].tolist(),intercept=float(result.x[0]),iterations=int(result.nit))


def score_part(part,name):
    y=np.array([r["nonpayback"] for r in part]);p=np.array([r[name] for r in part])
    pp=int(y.sum());nn=len(y)-pp
    auc=(rankdata(p)[y==1].sum()-pp*(pp+1)/2)/(pp*nn) if pp and nn else None
    nll=np.mean(-y*np.log(np.clip(p,1e-12,1.))-(1-y)*np.log(np.clip(1-p,1e-12,1.)))
    return dict(brier=float(np.mean((p-y)**2)),nll=float(nll),auc=None if auc is None else float(auc))


def grade(rows,heads):
    maps={}
    for r in rows:
        r["reference"]=runtime.predict(r["X"],heads["reference"])
        r["full"]=runtime.predict(r["X"],heads["full"])
        r["constant"]=heads["constant"]
    for m in sorted({r["game"] for r in rows}):
        all_rows=[r for r in rows if r["game"]==m]
        rr=[r for r in all_rows if r["usable"]]
        if not rr:
            maps[m]={"n":0,"censored":len(all_rows)}
            continue
        maps[m]=dict(n=len(rr),censored=len(all_rows)-len(rr),nonpayback=sum(r["nonpayback"] for r in rr),
            exit=dict(Counter(r["exit"] for r in rr)),
            nonpayback_by_exit=dict(Counter(r["exit"] for r in rr if r["nonpayback"])),
            cash_net_sum=sum(r["net"] for r in rr),cost_sum=sum(r["cost"] for r in rr),
            flagged=sum(r["full"]>=.5 for r in rr),profitable_flagged=sum(r["full"]>=.5 and not r["nonpayback"] for r in rr),
            profitable_flagged_net=sum(r["net"] for r in rr if r["full"]>=.5 and not r["nonpayback"]),
            positive_flagged=sum(r["full"]>=.5 and r["nonpayback"] for r in rr),
            models={k:score_part(rr,k) for k in ("constant","reference","full")})
    valid=[m for m in maps.values() if m["n"]]
    ranked=[m for m in valid if m["models"]["full"]["auc"] is not None]
    avg={k:{metric:float(np.mean([m["models"][k][metric] for m in (ranked if metric=="auc" else valid)]))
                   if (ranked if metric=="auc" else valid) else None
            for metric in ("brier","nll","auc")} for k in ("constant","reference","full")}
    improved=sum(m["models"]["full"]["brier"]<m["models"]["reference"]["brier"] for m in valid)
    return dict(by_game=maps,map_mean=avg,improved_brier=improved,rankable_maps=len(ranked),sampled_maps=len(maps),
        verdict=("INDETERMINATE" if len(valid)!=8 or len(ranked)<4 else "PASS"
                 if avg["full"]["brier"]<avg["reference"]["brier"] and improved>=5 and
                 avg["full"]["auc"]>=avg["reference"]["auc"] else "FAIL"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("split",choices=("train","held","exposed"));parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--cache",type=Path,default=register.CACHE)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=(HERE/"maps.json").read_bytes()
    if register.sha((HERE/"protocol.json").read_bytes())!=PROTOCOL:
        raise ValueError("Altered study protocol")
    ids=json.loads(manifest)["tickets"];msha=register.sha(manifest)
    if msha != "f4c2265d0f88843095e0fb12aec866314ceba631e10cb1959ba5298acf33701a":
        raise ValueError("Wrong frozen map registration")
    if args.split=="exposed":
        path=HERE.parent/"replays"/(HAJIMI+".json.gz")
        tickets=[dict(id=HAJIMI,compressed_sha256=register.sha(path.read_bytes()))]
    else:
        tickets=[r for r in ids if r["split"]==args.split]
    rows=collect(tickets,args.cache)
    if args.split=="train":
        rr=[r for r in rows if r["usable"]]
        grouped={t["id"]:[r["nonpayback"] for r in rr if r["game"]==t["id"]] for t in tickets}
        weights=dict(protocol=PROTOCOL,maps=msha,n_by_game={k:len(v) for k,v in grouped.items()},
            constant=float(np.mean([np.mean(v) for v in grouped.values() if v])),
            reference=fit(rr,len(runtime.REFERENCE)),full=fit(rr,len(runtime.FEATURES)))
        print("weights",dump(args.out/"weights.json",weights),len(rr),flush=True)
    weights_raw=(args.out/"weights.json").read_bytes()
    if args.split != "train" and register.sha(weights_raw) != "c7a21bb50bdeedb40032029ecf7b71c37c58a842bd55f92b881d2a8a12c4e8fd":
        raise ValueError("Held grading requires the frozen train-only weight bytes")
    weights=json.loads(weights_raw)
    if weights["maps"]!=msha or weights["protocol"]!=PROTOCOL:
        raise ValueError("Wrong fitted weights")
    result=grade(rows,weights)
    if args.split!="held":
        result["verdict"]="DESCRIPTIVE"
    if args.split=="exposed":
        result["exposed_plantings"]=[{k:v for k,v in r.items() if k!="X"} for r in rows
                                     if r["seat"]==1 and (r["t"],r["cell"]) in EXPOSED_CELLS]
        if len(result["exposed_plantings"])!=5:
            raise ValueError("Missing the registered exposed examples")
    result.update(protocol=PROTOCOL,maps_sha256=msha,weights_sha256=register.sha(weights_raw),
                  rows_sha256=dump(args.out/(args.split+"_trees.json.gz"),rows))
    print(args.split,dump(args.out/(args.split+".json"),result),len(rows),result["verdict"])


if __name__=="__main__":
    main()
