"""Comparably sized on-root BC/readouts, using cash from actual ownership intervals."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PARENT=HERE.parent/'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA='8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
MODES=('original','bc','weighted')
digest=lambda x:hashlib.sha256(x).hexdigest()

def write(path, obj):
    raw=(json.dumps(obj,allow_nan=False,ensure_ascii=False,indent=None if path.suffix=='.gz' else 2)+'\n').encode()
    tmp=path.with_suffix('.tmp')
    tmp.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)
    tmp.replace(path)
    return digest(path.read_bytes())

def encoder():
    source=PARENT.read_bytes();assert digest(source)==PARENT_SHA
    marker=b'        return lw.T, lb.T';assert source.count(marker)==1
    m=types.ModuleType('planter_training_encoder')
    exec(compile(source.replace(marker,b'        return hf.T, lw.T, lb.T'),str(PARENT),'exec'),m.__dict__)
    return m

def ownership_credit(events):
    credits={}
    for (t,uid),event in sorted(events.items()):
        for lag in (0,1,2):
            k=(t-lag,uid)
            if t<lag or k in credits and credits[k][0]<=lag: continue
            credits[k]=(lag,event)
    return credits

def collect(m,card):
    from labels import labels
    raw=(ROOT/card['path']).read_bytes()
    assert digest(raw)==card['sha256']
    replay=json.loads(gzip.decompress(raw));trace=replay['steps'];s=card['seat']
    events=labels(replay,s)
    credits=ownership_credit(events)
    tracker=m.BeliefTracker(m._Bel,s);previous=None;metadata=[];features=[];logits=[]
    seen=set()
    for t,(a,b) in enumerate(trace[:-1]):
        players=a['observation']['players'];personal=(a,b)[s]['observation']
        assert personal['player']==s if 'player' in personal else True
        carbon=personal['carbon'];est,known=tracker.step(players,t,carbon)
        issued=trace[t+1][s]['action'] or {}
        units=[(wid,w) for wid,w in players[s][2].items() if w[2]=='PLANTER' and
               ((t,wid) in credits or t%9==4)]
        if units:
            x=np.clip(np.rint(m.encode(players,t,s,est,known,previous)*100.0),0,255)*.01
            h,lw,_=m._POLICY.forward(x,m._SID)
            assert h.shape==(225,96) and lw.shape==(225,5)
            for uid,(cell,_,_) in units:
                action=issued.get(uid,'NONE')
                lag,event=credits.get((t,uid),(None,None))
                if lag==0:
                    assert action=='NONE'
                    seen.add((t,uid))
                w=math.exp(np.clip(event['net']/100.,-1.,1.)*math.exp(-lag/2)) if event else 1.
                features.append(h[cell]); logits.append(lw[cell])
                metadata.append(dict(t=t,uid=uid,cell=cell,y=m.WIDX[action],kind=event['kind'] if event else 'ordinary',
                             lag=lag,net=event['net'] if event else None,weight=w))
        tracker.set_actions(players,issued);previous=players
    assert seen==set(events)
    return metadata,np.asarray(features),np.asarray(logits)

def train(games):
    x=np.concatenate([g[1] for g in games]).astype(np.float64)
    z=np.concatenate([g[2] for g in games]).astype(np.float64)
    y=np.array([r['y'] for g in games for r in g[0]])
    weights=np.concatenate([np.full(len(g[0]),1./(len(games)*len(g[0]))) for g in games])
    mu=weights@x;sd=np.maximum(np.sqrt(weights@((x-mu)**2)),.03)
    f=np.column_stack(((x-mu)/sd,np.ones(len(x))))
    native=softmax(z,axis=1);lnative=z-logsumexp(z,axis=1,keepdims=True)
    out={};runs={}
    for name in MODES[1:]:
        w=np.concatenate([(np.array([a['weight'] for a in g[0]]) if name=='weighted' else np.ones(len(g[0]))) for g in games])
        i=0
        for g in games:
            n=len(g[0]);w[i:i+n]/=len(games)*w[i:i+n].sum();i+=n
        def objective(parameters):
            theta=parameters.reshape(97,5);t=np.tanh(f@theta/1.25);u=z+1.25*t
            lp=u-logsumexp(u,axis=1,keepdims=True);p=np.exp(lp)
            loss=-w@lp[np.arange(len(y)),y]+.25*np.sum(weights[:,None]*native*(lnative-lp))+.0015*np.square(theta[:-1]).sum()
            err=w[:,None]*p
            err[np.arange(len(y)),y]-=w
            grad=f.T@((err+.25*weights[:,None]*(p-native))*(1-t*t))
            grad[:-1]+=.003*theta[:-1]
            return loss,grad.ravel()
        fit=minimize(objective,np.zeros(485),method='L-BFGS-B',jac=True,options={'maxiter':180,'gtol':1e-5,'ftol':1e-9})
        theta=fit.x.reshape(97,5);W=theta[:-1]/sd[:,None];bias=theta[-1]-mu@W
        out[name+'_W']=W.astype(np.float32);out[name+'_b']=bias.astype(np.float32)
        assert np.isfinite(fit.fun) and np.isfinite(W).all()
        runs[name]=dict(success=bool(fit.success),status=fit.message,loss=float(fit.fun),iterations=fit.nit,
                        final_max_abs_grad=float(np.abs(fit.jac).max()),parameters=int(theta.size))
    return out,runs

def predict(g,pack,card):
    rows,x,z=g
    lp={}; npids=np.arange(len(rows))
    for name in MODES:
        updated=z if name=='original' else z+1.25*np.tanh((x@pack[name+'_W']+pack[name+'_b'])/1.25)
        lp[name]=updated.astype(np.float64)-logsumexp(updated.astype(np.float64),axis=1,keepdims=True)
    keys={k:card[k] for k in ('split','terrain','version','seat','match','opponent','opponent_id')}
    y=np.asarray([r['y'] for r in rows])
    for i,r in enumerate(rows):
        r.update(logp={n:float(lp[n][i,y[i]]) for n in MODES},top1={n:int(np.argmax(lp[n][i])) for n in MODES})
    return keys|dict(rows=rows)

def stats(cards):
    def grades(rows):
        w=np.asarray([r['weight'] for r in rows]); y=np.asarray([r['y'] for r in rows])
        safe=[r for r in rows if r['lag']==0 and r['net']>1e-7];bad=[r for r in rows if r['lag']==0 and r['net'] < -1e-7]
        def uplift(r):return r['logp']['weighted']-r['logp']['bc']
        def mean_or_none(v):return float(np.mean(v)) if len(v) else None
        v=[uplift(r) for r in safe+bad]
        auc=(float((rankdata(v)[:len(safe)].sum()-len(safe)*(len(safe)+1)/2)/(len(safe)*len(bad))) if safe and bad else None)
        def group(q):
            return dict(n=len(q),mean_uplift=mean_or_none([uplift(r) for r in q]),step_uplifts=sum(uplift(r)>1e-10 for r in q),
                        weighted_top1_hits=sum(r['top1']['weighted']==r['y'] for r in q),
                        bc_top1_hits=sum(r['top1']['bc']==r['y'] for r in q),
                        original_top1_hits=sum(r['top1']['original']==r['y'] for r in q))
        result=dict(rows=len(rows),nll={},nll_unweighted={},by_action_kind=dict(Counter(r['kind'] for r in rows if r['lag']==0)),
                    positives=group(safe),negatives=group(bad),zero_net=sum(r['lag']==0 and abs(r['net'])<=1e-7 for r in rows),
                    linked_movements=sum(r['lag'] in (1,2) and r['y']!=0 for r in rows),net_sign_uplift_auc=auc,
                    held_ordinary=sum(r['lag'] is None for r in rows))
        for n in MODES:
            l=np.array([r['logp'][n] for r in rows]);result['nll'][n]=float(-np.average(l,weights=w)); result['nll_unweighted'][n]=float(-np.mean(l))
        return result
    games=[{k:v for k,v in g.items() if k!='rows'}|grades(g['rows']) for g in cards]
    def avg(v):return float(np.mean(v)) if v else None
    pooled=grades([r for g in cards for r in g['rows']])
    avg_nll={n:avg([g['nll'][n] for g in games]) for n in MODES}
    hull={k:avg([g[k]['mean_uplift'] for g in games if g[k]['n']]) for k in ('positives','negatives')}
    checks=dict(weighted_nll_beats_original=avg_nll['weighted']<avg_nll['original'],
                weighted_nll_beats_bc=avg_nll['weighted']<avg_nll['bc'],
                profitable_observed_action_protected=hull['positives'] is not None and hull['positives']>=0,
                losing_observed_action_downweighted=hull['negatives'] is not None and hull['negatives']<0)
    return dict(by_game=games,pooled_descriptive=pooled,by_opponent={i:grades([r for g in cards if g['opponent_id']==i for r in g['rows']])
                for i in sorted({g['opponent_id'] for g in cards})},map_equal_nll=avg_nll,map_equal_observed_logp_change=hull,
                avg_within_map_auc=avg([g['net_sign_uplift_auc'] for g in games if g['net_sign_uplift_auc'] is not None]),
                checks=checks,offline_pass=all(checks.values()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('fit','held'));args=ap.parse_args()
    manifest=HERE/'protocol.json';prot=json.loads(manifest.read_bytes());cards=prot['tickets']
    if args.stage=='held':
        seal=json.loads((HERE/'fit.json').read_bytes())
        assert digest((HERE/'head.npz').read_bytes())==seal['weights_sha256'] and digest(manifest.read_bytes())==seal['protocol_sha256']
    model=encoder();selected=[c for c in cards if c['split']==args.stage]
    games=[]
    for c in selected:
        games.append(collect(model,c));print(args.stage,c['match'],len(games[-1][0]),flush=True)
    assert all(len(g[0]) for g in games)
    if args.stage=='fit':
        output,runs=train(games)
        np.savez_compressed(HERE/'head.npz',**output)
    else:
        runs=None
        with np.load(HERE/'head.npz',allow_pickle=False) as f:output={k:f[k] for k in f.files}
    classified=[predict(g,output,c) for g,c in zip(games,selected,strict=True)]
    ids=write(HERE/(args.stage+'_rows.json.gz'),classified)
    result=stats(classified)|dict(protocol_sha256=digest(manifest.read_bytes()),weights_sha256=digest((HERE/'head.npz').read_bytes()),
           parent_sha256=PARENT_SHA,source_sha256=digest(Path(__file__).read_bytes()),labels_sha256=digest((HERE/'labels.py').read_bytes()),
           rows_sha256=ids,optimizer=runs)
    write(HERE/(args.stage+'.json'),result)
    print(result['map_equal_nll'],result['checks'],result['weights_sha256'],flush=True)
if __name__=='__main__':main()
