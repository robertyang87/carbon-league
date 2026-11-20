"""Fit one fixed two-headed MLP and baselines; no counterfactual labels enter any optimizer."""
import datetime
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from register import H, read, dump, sha
from source import rows
from model import features

PROTOCOL='820bce03de67871bcefcea3227ae3b234a08af3e5649301301dea6ae7b4919f0'
def normalize(X,weight):
    mean=weight@X
    scale=np.sqrt(weight@((X-mean)**2))
    scale=np.maximum(scale,.05)
    return (X-mean)/scale,dict(mean=mean.tolist(),scale=scale.tolist())

def neural(X,Y,weight):
    x,params=normalize(X,weight)
    n=x.shape[1];h=24
    rng=np.random.RandomState(621034)
    w=rng.normal(0,np.sqrt(1/n),(n,h));out=rng.normal(0,np.sqrt(1/h),(h,2))
    initial=np.r_[w.ravel(),np.zeros(h),out.ravel(),np.zeros(2)]
    def unpack(z):return z[:n*h].reshape(n,h),z[n*h:n*h+h],z[n*h+h:-2].reshape(h,2),z[-2:]
    def obj(z):
        W,b,O,c=unpack(z);hidden=np.tanh(x@W+b);v=hidden@O+c
        y=np.logaddexp(0,v);e=y-Y;ae=np.abs(e)
        loss=np.dot(weight,np.mean(np.where(ae<=1,e**2/2,ae-.5),axis=1))
        dy=(weight[:,None]*.5)*np.clip(e,-1,1)*expit(v);df=(dy@O.T)*(1-hidden**2)
        return (loss+.0003*((W*W).sum()+(O*O).sum()),
                np.r_[(x.T@df+.0006*W).ravel(),df.sum(0),(hidden.T@dy+.0006*O).ravel(),dy.sum(0)])
    res=minimize(obj,initial,method='L-BFGS-B',jac=True,options={'maxiter':90,'ftol':1e-10})
    W,b,O,c=unpack(res.x)
    return dict(**params,w=W.tolist(),b=b.tolist(),out=O.tolist(),bias=c.tolist()),dict(iterations=int(res.nit),objective=float(res.fun),optimizer_status=res.message)

def main():
    assert sha((H/'protocol.json').read_bytes())==PROTOCOL
    metadata=[]; c=[];s=[];a=[];targets=[]
    for r in rows(read(H/'protocol.json')['train']):
        ck,sk,ak=features(r['players'],r['seat'],r['step'],r['order'])
        c.append(ck);s.append(sk);a.append(ak)
        targets.append(r['target'])
        metadata.append((r['terrain'],r['match'],r['seat']))
    counts=Counter(metadata);maps={k[0] for k in counts};games=Counter((g[0] for g in {(k[0],k[1]) for k in counts}))
    seats=Counter((k[:2] for k in counts))
    w=np.array([1/(len(maps)*games[k[0]]*seats[k[:2]]*counts[k]) for k in metadata])
    assert abs(w.sum()-1)<1e-8
    Y=np.array(targets)/500.; c,s,a=map(np.array,(c,s,a))
    X,baseline=normalize(c,w); Ymean=w@Y
    W=np.linalg.solve(X.T@(w[:,None]*X)+.0003*np.eye(2),X.T@(w[:,None]*(Y-Ymean)))
    heads={'cargo':dict(**baseline,w=W.tolist(),b=Ymean.tolist())};stats={}
    for name,X in [('state',s),('action',np.c_[s,a])]:
        heads[name],stats[name]=neural(X,Y,w)
    digest=dump(H/'weights.json',heads)
    print(dump(H/'fit.json',dict(protocol=PROTOCOL,weights=digest,roots=len(metadata),map_count=len(maps),
             games=len({k[:2] for k in metadata}),perspectives=len(counts),sample_hash=sha(np.c_[c,s,a,Y].astype('<f8').tobytes()),
             optimizer=stats,mean_target=(w@Y*500.).tolist(),n_train_by_terrain=dict(sorted(Counter(k[0] for k in metadata).items())))),digest,flush=True)
if __name__=='__main__':main()
