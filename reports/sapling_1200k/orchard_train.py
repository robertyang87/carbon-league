"""Fit first-tenure gross cash from observed Top Five purchases, split by map."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from orchard_value import orchard_features, orchard_predict

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"master_family/deep_review"))
from recover_public import canonical


def events(game, card):
    steps=game["steps"]
    cfg=game["configuration"]
    active={}
    rows=[]
    for t in range(len(steps)-1):
        prev=steps[t][0]["observation"]
        after=steps[t+1][0]["observation"]
        for seat in (0,1):
            old=prev["players"][seat][3]
            new=after["players"][seat][3]
            for tid in old.keys()-new.keys():
                active.pop((seat,tid),None)
            planters={p for p,_,r in prev["players"][seat][2].values() if r=="PLANTER"}
            for tid in new.keys()-old.keys():
                pos, age=new[tid]
                if pos not in planters: raise ValueError("missing pre-action planter")
                feat=orchard_features(prev,steps[t][seat]["observation"]["carbon"],seat,pos,t,cfg)
                if feat is None: continue
                kind, price, x=feat
                if (kind=="seize") != (tid in prev["players"][1-seat][3]):
                    continue  # ambiguous within-turn expiry
                rec=dict(game=card["id"],map=card["seed"],seat=seat,step=t,tree=tid,kind=kind,price=price,x=x,gross=0.)
                active[seat,tid]=rec
                rows.append(rec)
            for tid in new:
                if (seat,tid) in active:
                    active[seat,tid]["gross"]+=float(after["trees"][tid][1])
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    manifest=HERE/"top5_h2h_data.json"
    cards=json.loads(manifest.read_text())["games"]
    if len({c["seed"] for c in cards})!=len(cards): raise ValueError("map split collision")
    data=[]
    for card in cards:
        game=json.loads(gzip.decompress((args.replays/(card["id"]+".json.gz")).read_bytes()))
        if hashlib.sha256(canonical(game)).hexdigest()!=card["canonical_sha256"]: raise ValueError(card["id"])
        if game["statuses"]!=["DONE","DONE"]: raise ValueError("partial game")
        data.extend(dict(r,split=card["split"]) for r in events(game,card))
    result=dict(model={},validation={},train_games=sum(c["split"]=="train" for c in cards),
                validation_games=sum(c["split"]=="val" for c in cards),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    for kind in ("plant","seize"):
        subset=[r for r in data if r["kind"]==kind]
        train=[r for r in subset if r["split"]=="train"]
        val=[r for r in subset if r["split"]=="val"]
        x=np.array([r["x"] for r in train])
        y=np.array([r["gross"] for r in train])
        mu=x.mean(0)
        sd=np.maximum(.05,x.std(0))
        a=(x-mu)/sd
        # Fixed ridge, unlopped gross cash and 30 units of regularization in
        # standardized coordinates. No development-game or validation tuning.
        w=np.linalg.solve(a.T@a+30*np.eye(a.shape[1]),a.T@(y-y.mean()))/sd
        model=dict(intercept=float(y.mean()-mu@w),weights=w.tolist())
        result["model"][kind]=model
        def score(rows):
            actual=np.array([r["gross"] for r in rows])
            price=np.array([r["price"] for r in rows])
            pred=np.array([orchard_predict(model,r["x"]) for r in rows])
            low=pred+25<price
            high=pred>price+50
            return dict(n=len(rows),observed_gross_mean=float(actual.mean()),mae=float(abs(actual-pred).mean()),
                        train_mean_only_mae=float(abs(actual-y.mean()).mean()),
                        low_n=int(low.sum()),low_net=float((actual-price)[low].mean()) if low.any() else None,
                        high_n=int(high.sum()),high_net=float((actual-price)[high].mean()) if high.any() else None)
        result["validation"][kind]=dict(training_n=len(train),pooled=score(val),per_game={
            k:score([r for r in val if r["game"]==k]) for k in sorted({r["game"] for r in val})})
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":
    main()
