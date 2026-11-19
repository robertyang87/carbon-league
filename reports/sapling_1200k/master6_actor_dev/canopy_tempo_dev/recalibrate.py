"""Map-fold recalibration of a compressed public-action propensity with later independent terrain transfer."""
from collections import Counter,defaultdict
from hashlib import sha256

import numpy as np
from action_audit import HERE,read,save,digest
from study import BASE,elite_list,public_turns,collect,fit,mapmean
import policy


def logit(weights, x):
    return sum(a*b for a,b in zip(weights['beta'],x))


def expit(z):
    return 1/(1+np.exp(-np.clip(z,-40,40)))


def calibrate(rows):
    folds={r['map']:int.from_bytes(sha256(r['map'].encode()).digest()[:8],'big')%4 for r in rows}
    logits=[];labels=[];maps=[]
    for fold in range(4):
        w=fit([r for r in rows if folds[r['map']]!=fold])
        for r in rows:
            if folds[r['map']]==fold and policy.stage(r['t'])=='middle':
                logits.append(logit(w,r['x']));labels.append(float(r['hold']));maps.append(r['map'])
    freq=Counter(maps); weight=np.asarray([1/(len(freq)*freq[m]) for m in maps])
    z=np.asarray(logits); y=np.asarray(labels)
    center=weight@z; scale=np.sqrt(weight@(z-center)**2)
    x=np.column_stack([np.ones(len(z)),(z-center)/scale])
    b=np.zeros(2); diag=np.diag([0.,.002])
    for _ in range(30):
        pred=expit(x@b)
        grad=x.T@(weight*(pred-y))+diag@b
        hessian=x.T@((weight*pred*(1-pred))[:,None]*x)+diag
        delta=np.linalg.solve(hessian,grad)
        b-=delta
        if np.max(np.abs(delta))<1e-9:break
    if b[1]<0: raise ValueError('negative calibration slope')
    return {'center':float(center),'scale':float(scale),'intercept':float(b[0]),'slope':float(b[1]),
            'calibration_maps':dict(Counter(folds.values())),'training_middle_rows':len(y)}


def main():
    if digest(HERE/'protocol.json')!=read(HERE/'recalibration_protocol.json')['original_protocol']:
        raise ValueError('original study identity')
    train=[];old_maps=set();training_seeds=set()
    for _,part,g in elite_list():
        old_maps.add(g['canonical_sha256']);training_seeds.add(g['seed'])
        if part!='train':continue
        train.extend(collect(public_turns(g),g['id'])[1])
    raw=read(HERE/'weights.json'); c=calibrate(train)
    out=[]; ids=[]; goals=[]
    for b in ('appendix_336e8b8c','appendix_97de1ebc'):
        for g in read(BASE/'all_top5'/b/'manifest.json')['games']:
            if any('master3' in n.lower() for n in g['seat_names']):continue
            if g['canonical_sha256'] in old_maps or g['seed'] in training_seeds:raise ValueError('duplicate training map')
            if g['seed'] in ids:raise ValueError('repeated transfer game')
            ids.append(g['seed'])
            _,rs=collect(public_turns(g),g['id'])
            for r in rs:
                if policy.stage(r['t'])!='middle' or g['seat_strategy_ids'][r['seat']] is None:continue
                z=logit(raw,r['x'])
                r['uncal']=float(expit(z));r['new']=float(expit(c['intercept']+c['slope']*(z-c['center'])/c['scale']))
                if r['new']>=.5 and r['hold']:
                    goals.append(dict(id=g['id'],seat=r['seat'],step=r['t'],uid=r['uid'],p=r['new'],
                                     price=r['price'],gross=r['gross'],net=r['gross']-r['price']))
                out.append(r)
    base=read(HERE/'study.json')['training_phase_rates']['middle']
    def metrics(rs):
        return {k:mapmean(rs,lambda r:(r['hold']-(r[k] if k!='phase' else base))**2) for k in ('uncal','new','phase')}
    stats=metrics(out); permap={k:metrics([r for r in out if r['map']==k]) for k in sorted({r['map'] for r in out})}
    support=defaultdict(list)
    for g in goals:support[g['id']].append(g['net'])
    net=float(np.mean([np.mean(v) for v in support.values()])) if support else None
    positive=sum(float(np.mean(v))>0 for v in support.values())
    passed=(stats['new']<min(stats['uncal'],stats['phase']) and len(support)>=5 and len(goals)>=12 and net>0 and positive>len(support)/2)
    bins={}
    for i in range(10):
        rr=[r for r in out if min(int(r['new']*10),9)==i]
        if rr:bins[str(i/10)]={'n':len(rr),'predicted':float(np.mean([r['new'] for r in rr])),
                                'actual':float(np.mean([r['hold'] for r in rr]))}
    save(HERE/'recalibration.json',dict(protocol_sha256=digest(HERE/'recalibration_protocol.json'),code_sha256=digest(HERE/'recalibrate.py'),
         weights=c,evaluation_maps=len(ids),middle_rows=len(out),map_equal_brier=stats,by_map=permap,by_score=bins,
         high_score_opportunities=sum(r['new']>=.5 for r in out),paid=len(goals),paid_maps=len(support),positive_maps=positive,
         mean_net_equal_maps=net,net_negative_purchases=sum(g['net']<0 for g in goals),goals=goals,passed_public_prediction_and_pay_gate=bool(passed)))


if __name__=='__main__':
    main()
