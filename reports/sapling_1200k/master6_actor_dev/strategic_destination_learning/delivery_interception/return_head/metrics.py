"""Map-weighted paired discrimination, calibration and correctness switches."""
from collections import Counter
import numpy as np


def score(y,a,p,cash):
    y,a,p,cash=map(np.asarray,(y,a,p,cash))
    chosen=p>=.5
    return dict(n=len(y),positives=int(y.sum()),probability=float(p.mean()),observed=float(y.mean()),
                brier=float(np.square(p-y).mean()),nll=float(-(y*np.log(np.clip(p,1e-12,1))+
                      (1-y)*np.log(np.clip(1-p,1e-12,1))).mean()),cash_mae=float(abs(a-cash).mean()),
                cash_bias=float((cash-a).mean()),false_positive=int(np.count_nonzero(chosen&(y==0))),
                false_negative=int(np.count_nonzero(~chosen&(y==1))))


def switches(y,p,q):
    before=(p>=.5)==y
    after=(q>=.5)==y
    return dict(n=len(y),changed=int(np.count_nonzero(before!=after)),
                corrected=int(np.count_nonzero(~before&after)),broken=int(np.count_nonzero(before&~after)))


def evaluate(rows,pred):
    y=np.array([r["amount"]>0 for r in rows]);a=np.array([r["amount"] for r in rows])
    terr=np.array([r["terrain"] for r in rows])
    maps=np.unique(terr)
    g,n=pred["geometry"][0],pred["neural"][0]
    cards=[]
    for terrain in maps:
        take=terr==terrain
        pos=np.flatnonzero(take)[0]
        card=dict(terrain=str(terrain),seed=rows[pos]["seed"],n=int(take.sum()))
        card.update({k:score(y[take],a[take],p[take],cash[take]) for k,(p,cash) in pred.items()})
        card["switches"]=switches(y[take],g[take],n[take])
        cards.append(card)
    mapweights=np.array([1/(len(maps)*np.count_nonzero(terr==m)) for m in terr])
    calibration={}
    summary={}
    measures=("brier","nll","probability","observed","cash_mae","cash_bias")
    for k,(p,cash) in pred.items():
        summary[k]={m:float(np.mean([card[k][m] for card in cards])) for m in measures}
        bins=np.minimum((5*p).astype(int),4)
        cells=[]
        for j in range(5):
            take=bins==j
            wt=mapweights[take]
            cells.append(dict(lower=j/5,upper=(j+1)/5,n=int(take.sum()),weight=float(wt.sum()),
                              probability=float(np.average(p[take],weights=wt)) if take.any() else None,
                              observed=float(np.average(y[take],weights=wt)) if take.any() else None,
                              positives=int(y[take].sum())))
        calibration[k]=cells
        summary[k]["ece"]=float(sum(abs(c["probability"]-c["observed"])*c["weight"] for c in cells if c["n"]))
        for typ in ("false_positive","false_negative"):
            summary[k][typ]=sum(card[k][typ] for card in cards)
    draws=np.random.default_rng(4752).integers(len(cards),size=(10000,len(cards)))
    diff={}
    for m in ("brier","nll","cash_mae"):
        ds=np.array([c["neural"][m]-c["geometry"][m] for c in cards])
        diff[m]=dict(map_mean=float(ds.mean()),map_95_percentile=np.quantile(ds[draws].mean(axis=1),[.025,.975]).tolist(),
                     neural_lower_maps=int(np.count_nonzero(ds<0)),physical_maps=len(ds))
    return dict(roots=len(rows),positives=int(y.sum()),split_terrain_count=len(cards),summary=summary,
                differences=diff,threshold_switches=switches(y,g,n),maps=cards,calibration=calibration)
