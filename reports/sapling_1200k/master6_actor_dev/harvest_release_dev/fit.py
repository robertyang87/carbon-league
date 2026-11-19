"""Fit only on the original examiner seat, retaining independent map groups."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from policy import FEATURES, THRESHOLD, opportunities, choose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon

def load(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())

def save(p,x):
    b=(json.dumps(x,allow_nan=False,indent=None if p.suffix==".gz" else 2)+"\n").encode()
    p.write_bytes(gzip.compress(b,mtime=0) if p.suffix==".gz" else b)

def sources():
    cases=load(HERE.parent/"tactical_exchange_dev/paired.json.gz")
    assert len(cases)==18
    for c in cases:
        p=ROOT/c["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==c["source_sha256"]
        raw=load(p)
        assert raw["examiner_md5"].startswith("57f9a594")
        assert raw["candidate_seat"]==c["seat"] and raw["seed"]==c["seed"] and len(raw["actions"])==raw["steps"]
        yield c,raw

def split(seed):
    return "train" if seed in (621758322,621758323,621758324,621758325) else "held"

def fit():
    rows=[]
    used=[]
    for c,raw in sources():
        cfg=raw["game_configuration"]
        seat=1-c["seat"]
        sim=FastCarbon(cfg).reset(seed=c["seed"])
        prev=None
        for cmds in raw["actions"]:
            obs=sim.obs(seat)
            for uid,x in opportunities(obs,prev,cfg):
                rows.append((c["seed"],x,int((cmds[seat].get(uid) or "NONE")=="NONE")))
            sim.step(*cmds)
            prev=obs
        if not np.allclose(raw["official_score_by_seat"],sim.cash,rtol=0,atol=.011):
            raise ValueError("recorded final cash differs from engine")
        used.append({k:c[k] for k in ("seed","seat","path","source_sha256")})
    groups=np.asarray([r[0] for r in rows]);X=np.asarray([r[1] for r in rows]);y=np.asarray([r[2] for r in rows])
    tr=np.array([split(i)=="train" for i in groups]);a,b=X[tr],y[tr]
    def obj(v):
        z=a@v; w=v.copy();w[0]=0
        return float(np.logaddexp(0,z).sum()-z@b+1.5*(w@w)), a.T@(expit(z)-b)+3*w
    result=minimize(obj,np.zeros(a.shape[1]),method="L-BFGS-B",jac=True,options=dict(maxiter=1000,ftol=1e-12))
    if not result.success:raise ValueError(result.message)
    proba=expit(X@result.x)
    head=dict(features=FEATURES,weights=result.x.tolist(),threshold=THRESHOLD,rule="L2 strength 3, unpenalized intercept; uniform expert decisions",train_seeds=sorted(set(groups[tr].tolist())),held_seeds=sorted(set(groups[~tr].tolist())))
    save(HERE/"weights.json",head)
    metrics={}
    for name,mask in [("train",tr),("held",~tr)]+[(str(s),groups==s) for s in sorted(set(groups))]:
        truth=y[mask];prob=proba[mask];sel=prob>=THRESHOLD
        metrics[name]=dict(n=int(sum(mask)),actual_stays=int(truth.sum()),accuracy=float(np.mean((prob>=.5)==truth)),constant_train_accuracy=float(np.mean((np.mean(b)>=.5)==truth)),brier=float(np.mean((prob-truth)**2)),calls=int(sel.sum()),correct_calls=int(truth[sel].sum()))
    save(HERE/"fit_receipt.json",dict(sources=used,metrics=metrics))
    print(metrics["held"])

if __name__=="__main__":fit()
