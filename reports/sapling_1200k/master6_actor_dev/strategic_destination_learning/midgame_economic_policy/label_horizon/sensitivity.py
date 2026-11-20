"""Distinguish disjoint increments from dependence caused by adjacent scheduled labels and map-normalized weights."""
from collections import Counter
from hashlib import sha256
import numpy as np
from summarize import LANE,HERE,SEED,read,write


def correlation_ci(x,y,boot):
    def corr(x,y,axis):
        x=x-x.mean(axis=axis,keepdims=True);y=y-y.mean(axis=axis,keepdims=True)
        return np.sum(x*y,axis=axis)/np.sqrt(np.sum(x*x,axis=axis)*np.sum(y*y,axis=axis))
    return dict(r=float(corr(x,y,0)),cluster_95=np.quantile(corr(x[boot],y[boot],1),[.025,.975]).tolist())


def check(rows,adv):
    # One row per map. Fixed clock removes time mix; hash sample adds clock indicators.
    X=np.array([adv[(r["gid"],r["t"])]["regressors"] for r in rows])
    variables=["crossfit_residual","before_map_normalization","actual_weight","future_20_60"]
    responses=np.array([[r["residual"],adv[(r["gid"],r["t"])]["unnormalized"],r["weight"],
                     r["sides"][0]["late"]["net_cash"]-r["sides"][1]["late"]["net_cash"]] for r in rows])
    folds=np.array([r["fold"] for r in rows])
    z=np.zeros_like(responses)
    for fold in sorted(set(folds)):
        training=folds!=fold
        mu=X[training].mean(0);sd=np.maximum(1.,X[training].std(0))
        train=(X[training]-mu)/sd
        ymu=responses[training].mean(0)
        coef=np.linalg.solve(train.T@train+2*np.eye(X.shape[1]),train.T@(responses[training]-ymu))
        z[~training]=responses[~training]-ymu-((X[~training]-mu)/sd)@coef
    boot=np.random.default_rng(SEED).integers(len(rows),size=(5000,len(rows)))
    return dict(n=len(rows),maps=len({r["terrain"] for r in rows}),by_time=Counter(str(r["t"]) for r in rows),
                dimensions=variables,outcome="cash margin gain from t+20 until t+60, no transition in first 20 included",
                unadjusted={v:correlation_ci(responses[:,i],responses[:,-1],boot) for i,v in enumerate(variables[:-1])},
                adjusted={v:correlation_ci(z[:,i],z[:,-1],boot) for i,v in enumerate(variables[:-1])})


def main():
    rows=read(HERE/"training_cash.json.gz");adv={(r["gid"],r["t"]):r for r in read(LANE/"advantage.json.gz")}
    def pick(r):
        h=sha256(("midgame-horizon-audit-one-root-v1|"+r["terrain"]).encode()).digest()
        return (30,50,70,90,110)[int.from_bytes(h[:8],"big")%5]==r["t"]
    hash_rows=[r for r in rows if pick(r) and r["complete_60"]]
    last=[r for r in rows if r["t"]==110 and r["complete_60"]]
    write(HERE/"sensitivity.json",dict(one_clock_by_hash=check(hash_rows,adv),last_clock_only=check(last,adv),
         hash_allocated=234,hash_missing_full_60=234-len(hash_rows),last_allocated=234,last_missing_full_60=234-len(last),
         adjustment="Within each of the original five geography folds, predict the three fixed weight statistics and disjoint gain separately from the 17 public state variables with ridge penalty 2 using the other four folds. Correlate held predictions' residuals. Bootstrap treats these crossfitted predictions and original fitted weights as fixed. This descriptive check does not identify causal returns.",
         zero_forward_label_overlap="At t110 the normalization across this map's earlier roots uses only historical or current cash intervals; no other sampled label touches the t130..170 outcome.",
         uses_no_other_games=True))


if __name__=="__main__":
    main()
