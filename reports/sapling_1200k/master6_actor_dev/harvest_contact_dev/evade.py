"""Predict the escape of a threatened harvester from public board state, then score reachable intercepts."""
import numpy as np
from contact import _D,_TO,_DIRECTION

def situations(obs,prev):
    seat=obs['player']; own,enemy=obs['players'][seat],obs['players'][1-seat]
    if not prev:return
    for target,(p,cargo,kind) in enemy[2].items():
        old=prev['players'][1-seat][2].get(target)
        if kind!='COLLECTOR' or not old or old[0]!=p or not 5<=cargo<100 or cargo-old[1]<5:continue
        light=[w[0] for w in own[2].values() if w[2]=='COLLECTOR' and w[1]<cargo]
        if not light or min(_D[p,v] for v in light)!=1:continue
        home=list(enemy[1].values());cells=_TO[p];dist=_D[np.ix_(cells,light)].min(axis=1)
        h=_D[np.ix_(cells,home)].min(axis=1)
        others=[w[0] for k,w in enemy[2].items() if k!=target]
        tree=[w[0] for w in own[3].values()]
        x=np.column_stack((cells==p,dist==0,dist==1,dist==2,h[0]-h,h==0,
                           np.isin(cells,others),np.isin(cells,own[1].values()),np.isin(cells,tree),
                           (h[0]-h)*cargo/100,(cells==p)*(cargo-old[1])/30))
        yield target,cargo,cells,x

def intercept(obs,orders,prev,w):
    seat=obs['player']; own,enemy=obs['players'][seat],obs['players'][1-seat]
    planned={k:int(_TO[v[0],_DIRECTION.index(orders.get(k))]) for k,v in own[2].items()}
    best=None
    for target,load,cells,x in situations(obs,prev):
        z=x@w;probs=np.exp(z-z.max());probs/=probs.sum();mass=dict(zip(cells,probs))
        for uid,(pos,cargo,kind) in own[2].items():
            if kind!='COLLECTOR' or cargo>=load or cargo>25:continue
            reserved={v for k,v in planned.items() if k!=uid}
            reserved.update(v for k,v in own[1].items() if orders.get(k))
            for cmd,cell in zip(_DIRECTION,_TO[pos]):
                prob=mass.get(cell,0);delta=prob-mass.get(planned[uid],0)
                if prob<.35 or delta<.20 or cell in reserved or cell in enemy[1].values():continue
                if any(u[2]=='COLLECTOR' and u[1]<=cargo and _D[u[0],cell]<=1 for u in enemy[2].values()):continue
                gain=delta*load
                if best is None or gain>best['gain']:
                    best=dict(t=obs['step'],target=target,worker=uid,tree='',target_load=load,own_load=cargo,old=orders.get(uid) or 'NONE',alternative=cmd or 'NONE',gain=float(gain),p=float(prob),cell=int(cell))
    return best
