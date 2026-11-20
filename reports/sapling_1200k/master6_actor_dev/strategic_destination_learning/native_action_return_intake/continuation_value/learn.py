"""Fit a horizon-scaled value readout only on signed, same-BC natural games."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
INTAKE=HERE.parent
STUDY=INTAKE.parent
ROOT=STUDY.parents[3]
sys.path.insert(0,str(INTAKE))
import select_orders as selection
sys.path.insert(0,str(ROOT/"code"))
import cnet2_feat as F
common=selection.common
check=common.need
BC=common.SIGNED["bc"]
ORIGINAL=common.SIGNED["original"]
GROUPS=[("development","bc",[500,501]),("replication","bc",[510,511,512,513]),
        ("elite_direct_policy/development","control",[520,521,522,523])]
SEED_BASE=621758000
WIDTH=24
PRIOR=16.
CHANNELS=list(range(20))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,data):
    raw=(json.dumps(data,allow_nan=False,sort_keys=True,indent=2)+"\n").encode()
    if path.suffix==".gz":
        raw=gzip.compress(raw,mtime=0)
    if path.exists() and path.read_bytes()!=raw:
        raise ValueError("Refusing to replace a frozen output "+str(path))
    if not path.exists():
        path.write_bytes(raw)


def dist(a,b):
    dy=abs(a//15-b//15)
    dx=abs(a%15-b%15)
    return min(dy,15-dy)+min(dx,15-dx)


class History:
    def __init__(self,seat):
        self.seat=seat
        self.feat=F.Feat(seat)
        self.cash=[]

    def observe(self,obs):
        seat=self.seat
        t=obs["step"]
        check(len(self.cash)==t and obs["player"]==seat,"nonconsecutive public observations")
        by_seat=obs["players"]
        money=np.asarray([by_seat[seat][0],by_seat[1-seat][0]],np.float64)
        self.cash.append(money)
        x=self.feat.observe(obs)
        gap=float(money[0]-money[1])
        left=299-t
        features=[left/150.,*money/5000.]
        for lag in (3,10,30):
            features.extend((money-self.cash[max(0,t-lag)])/500.)
        for side in (seat,1-seat):
            _,bases,units,trees=by_seat[side]
            enemies=by_seat[1-side][2]
            carrier=[(pos,c) for pos,c,k in units.values() if k=="COLLECTOR"]
            planters=[pos for pos,c,k in units.values() if k=="PLANTER"]
            rival=[(q,v) for q,v,k in enemies.values() if k=="COLLECTOR"]
            positions=[(pos,min(dist(pos,b) for b in bases.values()),c) for pos,c in carrier]
            prospects=[]
            for pos,home,cargo in positions:
                light=min((dist(pos,q) for q,v in rival if v<=cargo),default=16)
                prospects.append((cargo,home,light))
            incomes=[obs["trees"][tid][1] for tid in trees]
            features.extend([len(carrier)/10.,len(planters)/10.,len(trees)/25.,
                sum(c for c,home,light in prospects)/1000.,
                sum(c for c,home,light in prospects if home<=2)/1000.,
                sum(c/(1+home) for c,home,light in prospects)/500.,
                sum(c for c,home,light in prospects if light<=2)/1000.,
                sum(c for c,home,light in prospects if home<=left and light>1)/1000.,
                sum(min(dist(pos,b) for b in bases.values()) for pos in planters)/100.,
                sum(max(0.,min(left,50-age)) for _,age in trees.values())/500.,
                sum(incomes)/100.,sum(dist(pos,min(bases.values(),key=lambda q:dist(q,pos))) for pos,age in trees.values())/100.])
        features.extend(np.array(x["scal"],np.float64).tolist())
        # Encoder maps summarize public distance, position, carbon estimates and recent observation history.
        features.extend(np.asarray(x["maps"],np.float64)[CHANNELS].mean(axis=1).tolist())
        check(np.isfinite(features).all(),"nonfinite public vector")
        return np.asarray(features),gap,left/150.


def captures():
    cfg={**common.collect.CFG,"actTimeout":1,"agentTimeout":20,"runTimeout":1500}
    records=[]
    archive_ids=[]
    roots=selection.read_json(INTAKE/"roots.json")["tickets"]
    excluded={r["terrain"] for r in roots}
    for lane,arm,tails in GROUPS:
        repo=STUDY/lane/"hosted"
        report=json.loads((repo/"settlement.json").read_bytes())
        for cart in report["archives"]:
            seed,seat=cart["seed"],cart["seat"]
            check(seed-SEED_BASE in tails and seat in (0,1),"unregistered training source")
            path=repo/(str(cart["id"])+".zip")
            check(digest(path)==cart["sha256"],"training archive identity")
            with zipfile.ZipFile(path) as bundle:
                reg=json.loads(bundle.read(arm+"/registration.json"))
                raw=bundle.read(arm+"/receipt.json")
                receipt=json.loads(raw)
                tape=bundle.read(arm+"/turns.jsonl")
            official,=[z for z in report["games"] if (z["seed"],z["seat"],z["arm"])==(seed,seat,arm)]
            check(reg["files"]["candidate"]["sha256"]==BC and reg["files"]["opponent"]["sha256"]==ORIGINAL and
                  reg["candidate_seat"]==seat and reg["seed"]==seed and receipt["counted"] and
                  hashlib.sha256(tape).hexdigest()==receipt["turns_sha256"] and
                  hashlib.sha256(raw).hexdigest()==official["receipt_sha256"],"source actors or tape changed")
            env=common.collect.begin(seed,cfg)
            terrain=hashlib.sha256(bytes(env.states[0].observation.full_carbon)).hexdigest()
            check(terrain==reg["physical_map_sha256"]==official["terrain"] and terrain not in excluded, "map overlap")
            bank=[5.,5.]
            state=History(seat)
            log=[]
            for t,line in enumerate(tape.splitlines()):
                tr=json.loads(line)
                raw_obs=common.public(env,seat,bank)
                obs={k:raw_obs[k] for k in ("step","player","carbon","players","trees","remainingOverageTime")}
                check(t==obs["step"]==tr["step"] and hashlib.sha256((json.dumps(obs,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()==tr["public_observation_sha256"][seat],
                      "visible history mismatch")
                features,cash,horizon=state.observe(obs)
                if 150<=t<=297 and t%3==0:
                    log.append((t,features,cash,horizon))
                check(all(c["status"]=="ok" for c in tr["calls"]),"failed match")
                env.step([c["response"]["action"] for c in tr["calls"]],[])
                bank=list(tr["overage_remaining_seconds"])
            finals=[s.reward for s in env.states]
            check(env.done and finals==receipt["official_scores_by_seat"]==official["official_scores_by_seat"],"terminal score mismatch")
            end=float(finals[seat]-finals[1-seat])
            for t,x,cash,horizon in log:
                records.append((seed,seat,t,x,cash,horizon,end))
            archive_ids.append(dict(seed=seed,seat=seat,terrain=terrain,arm=arm,archive=str(path.relative_to(ROOT)),
                                    archive_sha256=cart["sha256"],turns=receipt["turns_sha256"],final=end))
    check(len(archive_ids)==len(set((r["seed"],r["seat"]) for r in archive_ids))==20 and
          len({r["terrain"] for r in archive_ids})==10,"incomplete source")
    return records,archive_ids


def representation(x,mean,scale,W,b):
    z=np.clip((x-mean)/scale,-5,5)
    return np.column_stack([np.ones(len(z)),z,np.tanh(z@W+b)])


def fit(x,y,h):
    mean=x.mean(0)
    scale=np.maximum(x.std(0),.05)
    rng=np.random.default_rng(631891)
    W=rng.normal(0.,1./np.sqrt(x.shape[1]),(x.shape[1],WIDTH))
    b=rng.normal(0.,.3,WIDTH)
    phi=representation(x,mean,scale,W,b)*h[:,None]
    penalty=np.full(phi.shape[1],PRIOR)
    penalty[0]=.1
    beta=np.linalg.solve(phi.T@phi+np.diag(penalty),phi.T@y)
    return dict(mean=mean,scale=scale,W=W,b=b,beta=beta)


def predict(model,x,h,cash):
    phi=representation(x,model["mean"],model["scale"],model["W"],model["b"])
    return cash+1000.*h*(phi@model["beta"])


def main():
    records,cards=captures()
    seeds=np.array([r[0] for r in records])
    x=np.stack([r[3] for r in records])
    cash,h,terminal=np.array([r[4:] for r in records]).T
    y=(terminal-cash)/1000.
    report=[]
    preds=[]
    for seed in sorted(set(seeds)):
        m=seeds!=seed
        model=fit(x[m],y[m],h[m])
        estimates=predict(model,x[~m],h[~m],cash[~m])
        t=terminal[~m];v=cash[~m]
        report.append(dict(seed=int(seed),n=int((~m).sum()),model_mae=float(np.abs(estimates-t).mean()),
            cash_mae=float(np.abs(v-t).mean()),model_signed_bias=float((estimates-t).mean())))
        for idx,value in zip(np.flatnonzero(~m),estimates):
            n,s,step,_,_,_,_=records[idx]
            preds.append([n,s,step,cash[idx],terminal[idx],value])
    model=fit(x,y,h)
    np.savez(HERE/"weights.npz",**model)
    save(HERE/"fit.json",dict(protocol=digest(HERE/"protocol.json"),weights=digest(HERE/"weights.npz"),trainer=digest(Path(__file__)),
         source_games=cards,distinct_maps=10,n=len(records),features=x.shape[1],held_by_map=report,
         mean_mae=float(np.mean([r["model_mae"] for r in report])),cash_mean_mae=float(np.mean([r["cash_mae"] for r in report])),
         better_maps=sum(r["model_mae"]<r["cash_mae"] for r in report)))
    save(HERE/"out_of_map.json.gz",preds)


if __name__=="__main__":
    main()
