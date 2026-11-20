"""Bounded AWR with a cross-map phase baseline, an identically fitted BC control and zero increment."""
import argparse
from collections import Counter
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from data import HERE,P,save,feature_model,collect,facts,sha

ROLES=('worker','base')

def phase_offsets(games):
    by_map={}
    for g,rows in games:
        roots={r['t']:g['Y']-r['preV'] for r in rows}
        by_map.setdefault(g['map'],[]).append([float(np.mean([a for t,a in roots.items() if t//100==j])) for j in range(3)])
    a={k:np.mean(v,0) for k,v in by_map.items()}
    return {k:np.mean([v for n,v in a.items() if n!=k],0).tolist() for k in a},np.mean(list(a.values()),0).tolist()

def examples(games,role,offsets,global_offset):
    rows=[];root_to_game=[];metadata=[]
    for g,us in games:
        subset=[r for r in us if r['role']==role]
        ns=Counter(r['t'] for r in subset)
        for r in subset:
            a=float(np.clip(g['Y']-r['preV']-offsets.get(g['map'],global_offset)[r['stage']],-1.,1.))
            rows.append([r['phi'],r['z'],r['y'],a,1/(len(games)*len(ns)*ns[r['t']])])
            metadata.append([g['map'],g['seed'],g['seat'],g['margin'],r['t'],r['uid'],r['load'],r['family']])
    return [np.stack([r[i] for r in rows]) for i in range(5)],metadata

def estimate(rows,kind):
    h,z,y,a,v=rows;y=y.astype(int);v=v/v.sum()
    avg=v@h;std=np.sqrt(v@((h-avg)**2)).clip(.03)
    F=np.column_stack(((h-avg)/std,np.ones(len(h))));native=softmax(z.astype(np.float64),axis=1)
    w=(v*np.exp(1.2*a)) if kind=='awr' else v.copy();w/=w.sum()
    nclass=z.shape[1]
    def criterion(flat):
        B=flat.reshape(97,nclass)
        t=np.tanh(F@B/1.25);s=z+1.25*t;q=softmax(s,axis=1)
        nll=logsumexp(s,axis=1)-s[np.arange(len(s)),y];kl=(native*(np.log(native.clip(1e-200))-s+logsumexp(s,axis=1)[:,None])).sum(1)
        val=w@nll+.25*(v@kl)+.0015*np.square(B[:-1]).sum()
        r=w[:,None]*q;r[np.arange(len(y)),y]-=w
        grad=F.T@((r+.25*v[:,None]*(q-native))*(1-t*t));grad[:-1]+=.003*B[:-1]
        return val,grad.ravel()
    zero=np.zeros(97*nclass);result=minimize(criterion,zero,jac=True,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-9,'gtol':1e-5})
    return dict(mu=avg.astype(np.float32),sd=std.astype(np.float32),B=result.x.reshape(97,nclass).astype(np.float32)),dict(n=len(z),iters=result.nit,optimizer_message=str(result.message),gradient=float(abs(result.jac).max()),loss0=criterion(zero)[0],loss1=result.fun,weight_range=[float(np.exp(1.2*a).min()),float(np.exp(1.2*a).max())])

def assess(games,meta,book):
    records={};group={}
    for role in ROLES:
        rows,cards=examples(games,role,meta.get('loo',{}),meta['global']);h,z,y,a,base=rows;y=y.astype(int)
        outcomes={}
        for method in ('zero','bc','awr'):
            if method=='zero':s=z.copy()
            else:
                mu,sd,B=(book[method+'_'+role+'_'+k] for k in ('mu','sd','B'))
                s=z+1.25*np.tanh((((h-mu)/sd)@B[:-1]+B[-1])/1.25)
            nll=logsumexp(s.astype(np.float64),1)-s[np.arange(len(s)),y]
            p=softmax(s,axis=1);q=softmax(z,axis=1)
            tv=.5*np.abs(q-p).sum(1)
            rowsout=[]
            for i,c in enumerate(cards):
                rowsout.append(dict(map=c[0],seed=c[1],seat=c[2],margin=c[3],t=c[4],uid=c[5],load=c[6],family=c[7],weight=float(base[i]),A=float(a[i]),nll=float(nll[i]),top=int(p[i].argmax()==y[i]),tv=float(tv[i])))
            results={}
            for name in sorted(set(c[0] for c in cards)) + ['all']:
                sel=np.array([c[0]==name or name=='all' for c in cards])
                ww=base[sel]*np.exp(1.2*a[sel])
                results[name]=dict(n=int(sel.sum()),positive=int((a[sel]>0).sum()),awr_nll=float(np.average(nll[sel],weights=ww)),
                     nll=float(np.average(nll[sel],weights=base[sel])),top=int(np.sum(p[sel].argmax(1)==y[sel])),mean_tv=float(np.average(tv[sel],weights=base[sel])))
            outcomes[method]=results
            records[role+'_'+method]=rowsout
        group[role]=outcomes
    maps={g['map']:g['seed'] for g,_ in games}
    joined={}
    for name in list(maps)+['all']:
        joined[name]={m:float(np.mean([group[r][m][name]['awr_nll'] for r in ROLES])) for m in ('zero','bc','awr')}
    return dict(by_role=group,combined=joined,map_ids=maps,improved_maps=sum(joined[n]['awr']<joined[n]['bc'] for n in maps),
                aggregate_over_comparators=joined['all']['awr']<min(joined['all']['zero'],joined['all']['bc'])),records

def main():
    cli=argparse.ArgumentParser();cli.add_argument('split',choices=('fit','held'));cli.add_argument('--weights-sha');opt=cli.parse_args()
    file=HERE/'weights.npz'
    if opt.split=='held':assert sha(file.read_bytes())==opt.weights_sha
    data=collect(feature_model(),[c for c in P['tickets'] if c['split']==opt.split])
    if opt.split=='fit':
        loo,glob=phase_offsets(data);meta={'loo':loo,'global':glob};counts={};book={}
        for role in ROLES:
            rows,_=examples(data,role,loo,glob)
            for method in ('bc','awr'):
                p,c=estimate(rows,method);book.update({method+'_'+role+'_'+k:v for k,v in p.items()});counts[role+'_'+method]=c
        np.savez_compressed(file,**book)
        save(HERE/'fit.json',dict(protocol=sha((HERE/'protocol.json').read_bytes()),weights=sha(file.read_bytes()),offsets=meta,source=facts(data),metrics=counts))
        print('freeze',sha(file.read_bytes()),flush=True)
    else:
        meta=json.loads((HERE/'fit.json').read_bytes())['offsets'];meta={'global':meta['global']}
        with np.load(file) as f:book={k:f[k] for k in f.files}
    out,rows=assess(data,meta,book);out['source']=facts(data)
    save(HERE/(opt.split+'_scores.json'),out)
    if opt.split=='held':save(HERE/'held_unit_predictions.json.gz',rows)
    print(opt.split,out['combined']['all'],out['improved_maps'],flush=True)

if __name__=='__main__':main()
