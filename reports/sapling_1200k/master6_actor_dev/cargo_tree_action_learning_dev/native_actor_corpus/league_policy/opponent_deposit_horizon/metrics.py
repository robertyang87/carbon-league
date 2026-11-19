"""Descriptive actual-cash forecasts. Row expansion never increases the independent map count."""
from collections import defaultdict
import numpy as np

H=(3,5,10)
NAMES=('head','cargo_distance','original_distance')


def assess(t,uid,cargo,dist,forecasts,cash,first=None):
    """One game's time, worker id, public load/distance, 3 forecasts, actual own-bank sums and optional next event."""
    t,uid,cargo,dist=map(np.asarray,(t,uid,cargo,dist))
    forecasts=tuple(np.asarray(p) for p in forecasts);cash=np.asarray(cash)
    if cash.shape!=(len(t),3) or any(p.shape!=cash.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in forecasts):
        raise ValueError('Invalid per-worker horizon panel')
    bins={};metrics={};alerts={};examples=[]
    for k,h in enumerate(H):
        y=cash[:,k]>0;col={}
        for name,f in zip(NAMES,forecasts):
            p=f[:,k]
            error=(p-y)**2
            ix=np.minimum((10*p).astype(int),9)
            col[name]=dict(error=float(error.sum()),forecast=float(p.sum()),
                     bins=[[int((ix==i).sum()),float(p[ix==i].sum()),int(y[ix==i].sum())] for i in range(10)])
        metrics[h]=dict(n=len(t),positive=int(y.sum()),methods=col)
        roots=[{'t':a,'ix':np.flatnonzero((t==a)&(cargo>=100))} for a in np.unique(t)]
        def pick(indices,p):return max(indices,key=lambda i:(p[i]*cargo[i],-i))
        def card(i,p):
            return dict(t=int(t[i]),uid=str(uid[i]),cargo=round(float(cargo[i]),3),distance=int(dist[i]),
                        p=round(float(p[i]),6),delivered=round(float(cash[i,k]),3),
                        first_deposit_transition=None if not y[i] or first is None else int(first[i,k]))
        alerts[h]={}
        for name,f in zip(NAMES,forecasts):
            p=f[:,k];every=[];hi=[];first_alert=lead=None
            for root in roots:
                indices=root['ix']
                if not len(indices):continue
                j=pick(indices,p)
                if root['t']%h==0:
                    item=card(j,p);every.append(item)
                    if p[j]>=.7:hi.append(item)
                eligible=indices[p[indices]>=.7]
                if first_alert is None and len(eligible):first_alert=card(pick(eligible,p),p)
                early=eligible[dist[eligible]>=(2,3,4)[k]]
                if lead is None and len(early):lead=card(pick(early,p),p)
            def totals(rows):
                return dict(n=len(rows),hit=sum(q['delivered']>0 for q in rows),actual_cash=sum(q['delivered'] for q in rows))
            alerts[h][name]=dict(block_top=totals(every),high_block_top=totals(hi),first_high=first_alert,first_early_high=lead)
            if name=='head':
                examples.extend(dict(h=h,**q) for q in (first_alert,lead) if q)
    return dict(points=metrics,alerts=alerts,example_alarms=examples)


def aggregate(games):
    by=defaultdict(list)
    for g in games:by[g['terrain']].append(g)
    def mean(x):
        x=list(x)
        return float(np.mean(x)) if x else None
    result={}
    for h in H:
        methods={};collect=[];interval=[]
        for m in NAMES:
            group=[];nrows=positives=0
            for terrain,items in by.items():
                vals=[z['evaluation']['points'][h] for z in items]
                n=sum(x['n'] for x in vals);pos=sum(x['positive'] for x in vals)
                if not n:continue
                nrows+=n;positives+=pos
                parts=[q['methods'][m] for q in vals]
                group.append((sum(p['error'] for p in parts)/n,sum(p['forecast'] for p in parts)/n,pos/n,
                              np.sum([p['bins'] for p in parts],axis=0)/n))
            if not group:continue
            summed=np.sum([g[3] for g in group],axis=0)
            calibration=[dict(lower=i/10,weight=float(n/len(group)),forecast=float(p/n),actual=float(y/n))
                         for i,(n,p,y) in enumerate(summed) if n>0]
            methods[m]=dict(brier=mean(z[0] for z in group),forecast=mean(z[1] for z in group),
                            actual=mean(z[2] for z in group),reliability=calibration,
                            expected_calibration_error=sum(q['weight']*abs(q['forecast']-q['actual']) for q in calibration))
            collect.append([z[0] for z in group])
        if collect:
            v=np.array(collect[0])-collect[1];rng=np.random.default_rng(621758423+h)
            sample=v[rng.integers(len(v),size=(3000,len(v)))].mean(axis=1)
            interval=dict(mean=float(np.mean(v)),bootstrap_95=np.quantile(sample,[.025,.975]).tolist(),
                          improved_maps=int((v<0).sum()),maps=len(v))
        alert={}
        for m in NAMES:
            rows=[g['evaluation']['alerts'][h][m] for g in games];parts={}
            for key in ('block_top','high_block_top','first_high','first_early_high'):
                if key.endswith('top'):
                    cards=[c[key] for c in rows]
                    n=sum(c['n'] for c in cards);hits=sum(c['hit'] for c in cards);cash=sum(c['actual_cash'] for c in cards)
                else:
                    cards=[c[key] for c in rows if c[key] is not None]
                    n=len(cards);hits=sum(c['delivered']>0 for c in cards);cash=sum(c['delivered'] for c in cards)
                vals=[(r[key]['n'] if key.endswith('top') else int(r[key] is not None)) for r in rows]
                parts[key]=dict(alarms=n,hits=hits,false_alarms=n-hits,hit_share=hits/n if n else None,
                                mean_cash_per_alarm=cash/n if n else None,mean_alerts_per_game=mean(vals),
                                median_alerts_per_game=float(np.median(vals)) if vals else None)
            alert[m]=parts
        result[h]=dict(worker_turns=nrows,positive_turns=positives,map_weighted=methods,head_minus_cargo_distance=interval,alerts_game_counts=alert)
    return dict(games=len(games),maps=len(by),horizons=result)
