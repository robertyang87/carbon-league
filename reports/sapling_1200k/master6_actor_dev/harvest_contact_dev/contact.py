"""Interrupt a recent light harvest using only pre-action public worker history and own decoded orders."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'intercept_response_dev'))
from forecast import _TO,_D,_DIRECTION

def choose(obs,orders,previous):
    s=obs['player'];own,enemy=obs['players'][s],obs['players'][1-s]
    if previous is None or previous['step']+1!=obs['step']:return None
    old_enemy=previous['players'][1-s][2];best=None
    destinations={k:int(_TO[w[0],_DIRECTION.index(orders.get(k))]) for k,w in own[2].items()}
    for target,(p,load,kind) in enemy[2].items():
        prior=old_enemy.get(target)
        if kind!='COLLECTOR' or not prior or prior[0]!=p or not 5<=load<100 or load-prior[1]<5:continue
        if p in enemy[1].values() or p in own[1].values() or p in destinations.values():continue
        for uid,(pos,cargo,role) in own[2].items():
            if role!='COLLECTOR' or cargo>=load or cargo>25 or _D[pos,p]!=1:continue
            if any(w[2]=='COLLECTOR' and w[1]<=cargo and _D[w[0],p]<=1 for w in enemy[2].values()):continue
            score=load-cargo
            r=dict(t=obs['step'],worker=uid,target=target,tree='',old=orders.get(uid) or 'NONE',alternative=_DIRECTION[list(_TO[pos]).index(p)],
                   recent_gain=load-prior[1],target_load=load,own_load=cargo,score=score)
            if best is None or score>best['score']:best=r
    return best
