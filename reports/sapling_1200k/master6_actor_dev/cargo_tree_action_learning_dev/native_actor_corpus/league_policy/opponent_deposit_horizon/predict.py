"""Compose the two frozen public courier models; no future labels or candidate commands at inference."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ACTOR=next(p for p in HERE.parents if (p/'freight_horizon/model.json').is_file())
SHORT=ACTOR/'opponent_cargo_calibration_dev'
LONG=ACTOR/'freight_horizon'
SAP=ACTOR.parent
ROOT=SAP.parents[1]


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


short=module('public_courier_short',SHORT/'model.py')
horizons=(3,5,10)
read=lambda p:json.loads(p.read_bytes())
cargo_breaks=(25,50,100,175,300,500)
distance_breaks=(1,2,3,4,5,6,8,10,15)


def make_short_baseline():
    # Reconstruct the original distance logistic and strengthen it with a two-feature table.
    from sklearn.linear_model import LogisticRegression
    native=np.load(SHORT/'native_features.npz',allow_pickle=False)
    sources=read(SHORT/'sources.json')['native']
    tiles=np.asarray([c['terrain'] for c in sources])[native['game']]
    take=np.asarray([c['split']=='train' for c in sources])[native['game']]
    x=native['x'][take,:2].astype(float);y=native['events'][take,:2].astype(float)
    maps,idx,counts=np.unique(tiles[take],return_inverse=True,return_counts=True)
    w=len(x)/(len(maps)*counts[idx])
    head=np.load(SHORT/'head.npz',allow_pickle=False)
    dist=np.clip((x[:,1]-head['mu'][1])/head['scale'][1],-10,10)
    pairs=[LogisticRegression(C=1,solver='lbfgs',max_iter=400,tol=1e-9).fit(dist[:,None],y[:,i],sample_weight=w) for i in range(2)]
    coeff=[[float(p.coef_[0,0]),float(p.intercept_[0])] for p in pairs]
    d=np.digitize(x[:,1],distance_breaks,right=True);c=np.digitize(x[:,0],cargo_breaks,right=True)
    shape=(len(distance_breaks)+1,len(cargo_breaks)+1,2)
    table=np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            active=(d==i)&(c==j)
            dist_rows=(d==i)
            prior=np.average(y[dist_rows],weights=w[dist_rows],axis=0) if dist_rows.any() else np.array([0.,0.])
            table[i,j]=(w[active]@y[active]+25*prior)/(w[active].sum()+25)
    return dict(train_maps=sorted(maps.tolist()),train_rows=len(x),coefficient=coeff,table=table.tolist(),
                cargo_inclusive_upper=list(cargo_breaks),distance_inclusive_upper=list(distance_breaks),prior_worker_turns=25)


class Forest:
    def __init__(self,path):
        bundle=read(path)['any']
        self.bias=bundle['offset']
        self.trees=[np.array(x,dtype=float) for x in bundle['trees']]

    def probability(self,x):
        y=np.full(len(x),self.bias)
        idx=np.arange(len(x))
        for tree in self.trees:
            nodes=np.zeros(len(x),dtype=int)
            while True:
                features=tree[nodes,0].astype(int)
                internal=features>=0
                if not internal.any():break
                at=idx[internal];node=nodes[at]
                left=x[at,features[at]]<=tree[node,1]
                nodes[at]=np.where(left,tree[node,2],tree[node,3]).astype(int)
            y+=tree[nodes,4]
        return short.sigmoid(y)


class PublicDeposit:
    def __init__(self):
        self.short=short.Forecast()
        self.long=Forest(LONG/'model.json')
        self.long_base=Forest(LONG/'baseline.json')
        self.ref=read(HERE/'short_baseline.json')
        self.table=np.array(self.ref['table'])

    def arrays(self,x):
        """Return n×3 arrays for learned, distance-and-cargo, and legacy distance references."""
        w=self.short.w
        scaled=np.clip((x-w['mu'])/w['scale'],-10,10)
        y=short.sigmoid((scaled@w['weights'][:,:2]+w['bias'][:2])*w['platt_slope'][:2]+w['platt_bias'][:2])
        y[:,1]=np.maximum(y[:,0],y[:,1])
        z=self.table[np.digitize(x[:,1],distance_breaks,right=True),np.digitize(x[:,0],cargo_breaks,right=True)]
        c=np.array(self.ref['coefficient'])
        distance=short.sigmoid(scaled[:,1:2]*c[:,0]+c[:,1])
        long=self.long.probability(x);base=self.long_base.probability(x)
        return tuple(np.column_stack((v,u)) for v,u in ((y,long),(z,base),(distance,base)))

    def forecast(self,history):
        rows=short.public_rows(history)
        if not rows:return {}
        probs=self.arrays(np.stack([r[1] for r in rows]))[0]
        return {r[0]:dict(zip(horizons,p.tolist())) for r,p in zip(rows,probs)}


if __name__=='__main__':
    (HERE/'short_baseline.json').write_text(json.dumps(make_short_baseline(),indent=2,allow_nan=False)+'\n')
