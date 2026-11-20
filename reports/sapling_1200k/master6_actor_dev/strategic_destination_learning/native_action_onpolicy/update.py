"""One fixed-policy, one-pass episode score update, with map-excluded evaluation."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from build import HERE, build, sha
from kernel import physical

ZERO=np.zeros(24,np.float32)
INTAKE=HERE.parent/"native_action_onpolicy_intake"


def need(value,message):
    if not value:
        raise ValueError(message)


def load(path):
    raw=path.read_bytes()
    if path.suffix==".gz":
        raw=gzip.decompress(raw)
    return json.loads(raw)


def save(path,data):
    raw=(json.dumps(data,sort_keys=True,allow_nan=False,indent=2)+"\n").encode()
    if path.suffix==".gz":
        raw=gzip.compress(raw,mtime=0)
    if path.exists() and path.read_bytes()!=raw:
        raise ValueError("A different result is already frozen at "+str(path))
    path.write_bytes(raw)


def targets(margin):
    margin=np.asarray(margin,dtype=np.float64)
    wdl=(margin>0).astype(float)+.5*(margin==0)
    utility=.5*wdl+.5/(1.+np.exp(-np.clip(margin/1000.,-700.,700.)))
    return np.stack((utility,wdl,margin),axis=-1)


def baseline(pool,t,cash_gap):
    need(t%6==0 and 0<=t<=294,"Unexpected pivot")
    return targets([cash_gap+g["remaining"][t//6] for g in pool]).mean(axis=0)


def prepare(g,pool):
    steps=g["steps"]
    margin=float(g["official_margin"])
    scores=g["official_scores_by_seat"]
    seat=g["seat"]
    need(abs(margin-(scores[seat]-scores[1-seat]))<1e-6 and abs(g["u"]-targets(margin)[0])<1e-8,"Episode U/score drift")
    need(0<len(steps)<=299 and [r["step"] for r in steps]==list(range(len(steps))),"Incomplete action history")
    pivots=[]
    grad=np.zeros(24)
    val=targets(margin)
    ordinary=0
    for row in steps:
        t=row["step"]
        menu=row["policy_features"]
        orders=row["orders"]
        k=row["issued_order_index"]
        physical_keys=[tuple(sorted(o["action"].items())) for o in orders]
        need(len(set(physical_keys))==len(orders)>0 and 0<=k<len(orders),"Duplicate/unknown issued order")
        old,score=physical(menu,len(orders),ZERO)
        p0=np.array([o["probability"] for o in orders])
        need(np.all(p0>0) and np.max(abs(old-p0))<5e-6 and abs(old[k]-row["behavior_probability"])<5e-6
             and abs(np.log(old[k])-row["logp"])<5e-6,"Unmatched behavior likelihood")
        if menu is None or len(orders)==1:
            need(len(orders)==1 and row["logp"]==0. and np.max(abs(score))<1e-8,"Unscored deterministic decision")
            ordinary+=1
            continue
        need(0<t<=294 and t%6==0,"Unregistered stochastic step")
        b=baseline(pool,t,row["cash_margin_before"])[0]
        a=float(val[0]-b)
        grad+=a*score[k]
        pivots.append((row,old,b))
    b0=baseline(pool,0,steps[0]["cash_margin_before"])
    return dict(seed=g["seed"],seat=seat,terrain=g["terrain"],n=len(steps),ordinary=ordinary,pivots=pivots,grad=grad,
                val=val,b0=b0)


def metrics(g,w,record=False):
    logratio=0.
    kls=[]
    tvs=[]
    changed=0
    ledger=[]
    for row,old,b in g["pivots"]:
        p,_=physical(row["policy_features"],len(old),w)
        issue=row["issued_order_index"]
        old_l=float(np.log(old[issue]))
        new_l=float(np.log(p[issue]))
        logratio+=new_l-old_l
        tvs.append(float(.5*np.abs(p-old).sum()))
        kls.append(max(0.,float(np.sum(old*(np.log(old)-np.log(p))))))
        stream=f'Sapling-policy-v1:{g["terrain"]}:{g["seat"]}:{row["step"]}'.encode()
        uniform=int.from_bytes(hashlib.sha256(stream).digest()[:8],"big")/2**64
        changed+=int(np.searchsorted(old.cumsum(),uniform)!=np.searchsorted(p.cumsum(),uniform))
        if record:
            ledger.append(dict(step=row["step"],issued_order_index=issue,old_logp=old_l,new_logp=new_l,
                               U_baseline=b,old_probabilities=old.tolist(),new_probabilities=p.tolist()))
    return dict(logratio=logratio,mean_kl=float(np.mean(kls)) if kls else 0.,tv_sum=sum(tvs),max_tv=max(tvs,default=0.),
                n=len(tvs),changed=changed,ledger=ledger)


def update(games):
    grad=np.mean([g["grad"] for g in games],axis=0)
    weight=(.2*grad/max(.1,np.linalg.norm(grad))).astype(np.float32)
    for halvings in range(13):
        per=[metrics(g,weight) for g in games]
        n=sum(r["n"] for r in per)
        kl=sum(r["mean_kl"]*r["n"] for r in per)/max(n,1)
        if kl<=.002 and max(r["max_tv"] for r in per)<=.08 and max(abs(r["logratio"]) for r in per)<=.5:
            return weight,dict(halvings=halvings,grad_norm=float(np.linalg.norm(grad)),mean_kl=kl,n=n,
                               max_tv=max(r["max_tv"] for r in per),max_abs_logratio=max(abs(r["logratio"]) for r in per))
        weight*=.5
    raise ValueError("Unable to certify even the smallest allowed update")


def summarize(x):
    rng=np.random.default_rng(25953)
    bs=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    return dict(mean=float(np.mean(x)),positive_maps=int((x>0).sum()),bootstrap_interval=np.quantile(bs,[.025,.975]).tolist())


def fit():
    data_path=INTAKE/"training_rows.json.gz"
    data=load(data_path)
    gate=load(HERE/"gate.json")
    reg=load(INTAKE/"protocol.json")
    need(gate["eligible_for_training_collection"] and sha(HERE/"sampler.py")==data["behavior_sha256"]==gate["sampler"]
         and sha(HERE/"behavior.npz")==data["weight_sha256"]==gate["head"],"Different behavior network")
    games=data["games"]
    need(len(games)==8 and Counter(g["seat"] for g in games)=={0:4,1:4} and
         len({g["terrain"] for g in games})==8 and
         {str(g["seed"]):g["terrain"] for g in games}==reg["physical_boards"],"Unregistered terrain, seat or denominator")
    past=load(HERE/"baseline.json")["games"]
    need(not {g["terrain"] for g in games}&{r["terrain"] for r in past},"Baseline leaked experimental terrain")
    valid=[prepare(g,[r for r in past if r["seat"]==g["seat"]]) for g in games if g["gradient_eligible"]]
    if len(valid)<2:
        save(HERE/"fit.json",dict(source=sha(data_path),eligible=False,games=len(games),usable=len(valid),reason="Fewer than two usable episodes"))
        return
    full,limits=update(valid)
    tests=[]
    fullrows=[]
    for g in valid:
        hold,lim=update([other for other in valid if other is not g])
        m=metrics(g,hold,record=True)
        m_all=metrics(g,full,record=True)
        ratio=float(np.exp(m["logratio"]))
        estimates=(ratio-1)*(g["val"]-g["b0"])
        tests.append(dict(seed=g["seed"],terrain=g["terrain"],seat=g["seat"],utility=float(g["val"][0]),wdl=float(g["val"][1]),
                          margin=float(g["val"][2]),start_baseline=g["b0"].tolist(),weights=hold.tolist(),limits=lim,
                          held_logratio=m["logratio"],held_ratio=ratio,held_tv=m["max_tv"],held_mean_kl=m["mean_kl"],held_gain=estimates.tolist()))
        fullrows.append(dict(seed=g["seed"],terrain=g["terrain"],seat=g["seat"],ordinary=g["ordinary"],
                             sampled=m_all["n"],changed=m_all["changed"],full_logratio=m_all["logratio"],tv_sum=m_all["tv_sum"],
                             full=m_all["ledger"],held=m["ledger"]))
    estimates=np.array([r["held_gain"] for r in tests])
    tv_mean=sum(r["tv_sum"] for r in fullrows)/max(1,sum(r["sampled"] for r in fullrows))
    changed_maps=sum(r["changed"]>0 for r in fullrows)
    rho=np.asarray([r["held_ratio"] for r in tests])
    checks=dict(eight_usable=len(valid)==8,nonzero_weights=bool(np.any(full)),
        all_held_likelihoods_bounded=all(abs(r["held_logratio"])<=.5 for r in tests),
        all_held_kl_tv_bounded=all(r["held_mean_kl"]<=.002 and r["held_tv"]<=.08 for r in tests),
        positive_held_mean=bool(estimates[:,0].mean()>0),five_positive_held_maps=bool((estimates[:,0]>0).sum()>=5),
        mean_changed_mass_at_least_one_percent=tv_mean>=.01,two_maps_changed_command=changed_maps>=2)
    with np.load(HERE/"behavior.npz",allow_pickle=False) as h:
        np.savez(HERE/"trained.npz",W0=h["W0"],B0=h["B0"],W1=full[:,None])
    built=build(HERE/"trained.py",HERE/"trained.npz")
    save(HERE/"ratios.json.gz",fullrows)
    choices=Counter("win" if g["val"][1]==1 else ("loss" if g["val"][1]==0 else "draw") for g in valid)
    save(HERE/"fit.json",dict(data=sha(data_path),algorithm=sha(Path(__file__)),protocol=sha(HERE/"protocol.json"),
        source_gate=sha(HERE/"gate.json"),baseline=sha(HERE/"baseline.json"),ratios=sha(HERE/"ratios.json.gz"),
        sampler=gate["sampler"],trained=built,head=sha(HERE/"trained.npz"),games=len(games),usable=len(valid),WDL=choices,
        all_game_report=[{k:g.get(k) for k in ("seed","terrain","seat","status","counted","winner_seat","wdl","official_margin","u","gradient_eligible")} for g in games],
        usable_mean_margin=float(np.mean([g["val"][2] for g in valid])),usable_mean_U=float(np.mean([g["val"][0] for g in valid])),
        limits=limits,folds=tests,held={name:summarize(estimates[:,k]) for k,name in enumerate(("U","WDL","official_margin"))},
        ESS=float(rho.sum()**2/rho.dot(rho)),total_variation=tv_mean,changed_maps=changed_maps,checks=checks,
        pre_replay_gate=all(checks.values()),eligible_for_strength_match=False,
        note="Realized development claims require independent frozen-package replay and new complete original-opponent matches."))

if __name__=="__main__":
    fit()
