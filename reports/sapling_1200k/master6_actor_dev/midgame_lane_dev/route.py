"""Occupy a return lane with a light collector, leaving the team's loaded carriers free."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'intercept_response_dev'))
from forecast import _TO,_D,_DIRECTION

def propose(obs,orders,focus=None):
    seat=obs['player']; own,enemy=obs['players'][seat],obs['players'][1-seat]
    options=[]
    for target,(pos,load,kind) in enemy[2].items():
        if kind!='COLLECTOR' or load<100 or (focus and target!=focus['target']):continue
        bases=list(enemy[1].values()); d=min(_D[pos,b] for b in bases)
        if d<2 or d>5:continue
        # Count shortest paths via each non-base cell using distance and path multiplicity.
        frontier={pos:1.}; flow={}
        for depth in range(1,int(d)):
            following={}
            for p,mass in frontier.items():
                downhill=[c for c in _TO[p] if min(_D[c,b] for b in bases)==d-depth]
                for c in downhill:following[c]=following.get(c,0)+mass/len(downhill)
            flow.update({c:(depth,mass) for c,mass in following.items()})
            frontier=following
        for uid,(p,cargo,k) in own[2].items():
            if k!='COLLECTOR' or cargo>25 or (focus and uid!=focus['worker']):continue
            reserved={_TO[w[0],_DIRECTION.index(orders.get(i))] for i,w in own[2].items() if i!=uid}
            reserved.update(b for i,b in own[1].items() if orders.get(i))
            def value(cell):
                if cell in reserved or cell in bases:return -1.
                if any(w[2]=='COLLECTOR' and w[1]<=cargo and _D[cell,w[0]]<=1 for w in enemy[2].values()):return -1.
                return max((mass/depth for c,(depth,mass) in flow.items() if c==cell),default=0.)
            old=orders.get(uid); scores=[value(cell) for cell in _TO[p]]
            best=max(range(5),key=lambda i:(scores[i],_DIRECTION[i]==old))
            gain=scores[best]-scores[_DIRECTION.index(old)]
            if scores[best]>=0.3 and gain>=0.2:
                options.append(dict(worker=uid,target=target,t=obs['step'],old=old or 'NONE',alternative=_DIRECTION[best] or 'NONE',
                                    benefit=float(load*gain),cargo=cargo,target_load=load,tree=''))
    return max(options,key=lambda r:(r['benefit'],r['worker'],r['target']),default=None)

def apply(orders,decision):
    if not decision:return orders
    result=dict(orders)
    if decision['alternative']=='NONE':result.pop(decision['worker'],None)
    else:result[decision['worker']]=decision['alternative']
    return result
