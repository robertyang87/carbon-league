"""Fit calibrated H3/H5 events, freeze native predictions, then report on known and held-out registrations."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from model import HERE,RULES,ACTOR,Forecast,FEATURES,TARGETS,move_prob,sigmoid


def load():
    docs=[np.load(HERE/f'{n}_features.npz',allow_pickle=False) for n in ('native','ladder')]
    return {k:np.concatenate([d[k] for d in docs]) for k in docs[0].files}


def equal(terrain):
    keys,inv,c=np.unique(terrain,return_inverse=True,return_counts=True);return len(terrain)/(len(keys)*c[inv])


def logistic(x,y,w):
    fit=LogisticRegression(C=1,solver='lbfgs',max_iter=400,tol=1e-9)
    fit.fit(x,y,sample_weight=w)
    return fit.coef_[0],fit.intercept_[0]


def multinomial(q,a,w):
    return dict(accuracy=float(np.average(q.argmax(1)==a,weights=w)),
                logloss=float(np.average(-np.log(np.maximum(1e-12,q[np.arange(len(a)),a])),weights=w)),
                brier=float(np.average(np.sum((q-np.eye(5)[a])**2,1),weights=w)))


def event_metrics(prob,y,w):
    ev=y.astype('d');p=np.clip(prob,1e-10,1-1e-10)
    return dict(brier=float(np.average((p-ev)**2,weights=w)),logloss=float(np.average(-(ev*np.log(p)+(1-ev)*np.log1p(-p)),weights=w)),
                observed=float(np.average(ev,weights=w)),predicted=float(np.average(prob,weights=w)),events=int(y.sum()),
                average_precision=float(average_precision_score(y,p,sample_weight=w)) if np.any(y) else None)


def main():
    d=load();g=d['game'];x=d['x'].astype(float);targets=d['events'].astype(int);groups=d['group'];prior=d['prior'];labels=d['action'];last=d['last']
    src=json.loads((HERE/'sources.json').read_text());cards=src['native']+src['ladder'];split=np.array([c['split'] for c in cards])[g]
    terr=np.array([c['terrain'] for c in cards])[g];name=np.array([c['opponent'] for c in cards])[g];seat=np.array([c['seat'] for c in cards])[g]
    tr=split=='train';ca=split=='calibration'
    wt=equal(terr[tr]);wc=equal(terr[ca]);mu=np.average(x[tr],axis=0,weights=wt)
    sd=np.maximum(np.sqrt(np.average((x[tr]-mu)**2,axis=0,weights=wt)),.05);sx=np.clip((x-mu)/sd,-10,10)
    temp=[]
    for k in (0,1):
        ix=ca&(groups==k); ww=equal(terr[ix])
        fun=lambda val:multinomial(move_prob(prior[ix],np.zeros(ix.sum(),int),np.array([val])),labels[ix],ww)['logloss']
        temp.append(minimize_scalar(fun,bounds=(.25,5),method='bounded').x)
    coeff=[];bias=[];alpha=[];beta=[];ref=[];base=[]
    for j in range(4):
        b,c=logistic(sx[tr],targets[tr,j],wt);coeff.append(b);bias.append(c)
        raw=sx[ca]@b+c
        def obj(k):
            lo=k[0]*raw+k[1];return np.average(np.logaddexp(0,lo)-targets[ca,j]*lo,weights=wc)
        res=minimize(obj,[1.,0.],bounds=[(0,10),(-20,20)],method='L-BFGS-B')
        if not res.success:raise RuntimeError(res.message)
        alpha.append(res.x[0]);beta.append(res.x[1])
        chosen=1 if j<2 else FEATURES.index('loaded_enemy_distance')
        r,b0=logistic(sx[tr,chosen,None],targets[tr,j],wt);ref.append(sigmoid(sx[:,chosen]*r[0]+b0));base.append(np.average(targets[tr,j],weights=wt))
    np.savez_compressed(HERE/'head.npz',mu=mu,scale=sd,weights=np.stack(coeff,axis=1),bias=bias,platt_slope=alpha,platt_bias=beta,temperature=temp)
    q,p=Forecast().arrays(x,prior,groups);rawp=sigmoid(sx@np.stack(coeff,axis=1)+bias);refs=np.stack(ref,axis=1)
    inertia=np.full((len(g),5),.05);mask=last>=0;inertia[np.flatnonzero(mask),last[mask]]=.8;inertia[~mask]=.2
    assert np.allclose(q.sum(1),1) and np.all(p[:,1]>=p[:,0]) and np.all(p[:,3]>=p[:,2])
    def scores(ix,detail=True):
        n=int(ix.sum())
        if not n:return dict(n=0)
        w=equal(terr[ix]);res=dict(n=n,maps=len(set(terr[ix])),directions={k:multinomial(a[ix],labels[ix],w) for k,a in [('calibrated',q),('frozen_prior',prior),('inertia',inertia)]},events={})
        for j,target in enumerate(TARGETS):
            b=res['events'][target]={k:event_metrics(val,targets[ix,j],w) for k,val in [('head',p[ix,j]),('before_platt',rawp[ix,j]),('distance',refs[ix,j]),('frequency',np.full(n,base[j]))]}
            if detail:
                bins=np.minimum(np.floor(p[ix,j]*10).astype(int),9);z=[]
                for i in range(10):
                    sel=bins==i
                    if sel.any():z.append(dict(bin=i,n=int(sel.sum()),forecast=float(np.average(p[ix,j][sel],weights=w[sel])),actual=float(np.average(targets[ix,j][sel],weights=w[sel]))))
                b['reliability']=z
                delta=(p[ix,j]-targets[ix,j])**2-(refs[ix,j]-targets[ix,j])**2
                by=[float(delta[terr[ix]==t].mean()) for t in sorted(set(terr[ix]))]
                rng=np.random.default_rng(50413);bt=np.array(by)[rng.integers(0,len(by),(1500,len(by)))].mean(1)
                b['map_mean_brier_difference_vs_distance']=dict(point=float(np.mean(by)),ci95=np.quantile(bt,[.025,.975]).tolist())
        return res
    masks={s:split==s for s in sorted(set(split))}; masks['native_all_held']=np.isin(split,['recent_evaluation','held'])
    out=dict(weights_sha256=hashlib.sha256((HERE/'head.npz').read_bytes()).hexdigest(),prior_sha256=hashlib.sha256((RULES/'opponent_intent.json').read_bytes()).hexdigest(),
             features=FEATURES,temperature=temp,targets=TARGETS,benchmark_single_feature={'bank3/5':FEATURES[1],'contest3/5':'loaded_enemy_distance'},
             panels={k:scores(v) for k,v in masks.items()},by_ladder_name={k:scores(masks['ladder']&(name==k),False) for k in sorted(set(name[masks['ladder']]))},
             by_seat={k:[scores(v&(seat==s),False) for s in (0,1)] for k,v in masks.items() if k in ('native_all_held','ladder')})
    (HERE/'evaluation.json').write_text(json.dumps(out,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
    audit=[]
    for i,c in enumerate(cards):
        subset=g==i
        if subset.any():
            z=scores(subset,False);z.update({k:c[k] for k in ('id','seed','seat','terrain','opponent','split')});audit.append(z)
    (HERE/'map_metrics.json.gz').write_bytes(gzip.compress(json.dumps(audit,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    print('fit complete',[(k,int(v.sum())) for k,v in masks.items()])


if __name__=='__main__':main()
