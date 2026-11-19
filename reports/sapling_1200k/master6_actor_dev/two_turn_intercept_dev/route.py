"""Choose one free first destination and a second-turn intercept from a public carrier forecast."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'intercept_response_dev'))
from forecast import carriers,_D,_DIRECTION,_TO
W=np.load(HERE.parent/'intercept_response_dev/weights.npz')['weights']

def dest(pos,cmd):
    return int(_TO[pos,_DIRECTION.index(None if cmd=='NONE' else cmd)])

def options(obs,orders,uid):
    s=obs['player'];own,opp=obs['players'][s],obs['players'][1-s]
    reserved={dest(w[0],orders.get(k)) for k,w in own[2].items() if k!=uid}
    reserved.update(p for k,p in own[1].items() if orders.get(k))
    pos,load,_=own[2][uid]
    danger=[w[0] for w in opp[2].values() if w[2]=='COLLECTOR' and w[1]<=load]
    return [(cmd,int(cell)) for cmd,cell in zip(_DIRECTION,_TO[pos]) if cell not in reserved
            and cell not in opp[1].values() and all(_D[cell,p]>1 for p in danger)]

def distribution(features):
    v=features@W
    p=np.exp(v-v.max())
    return p/p.sum()

def choose(obs,orders):
    s=obs['player'];players=obs['players'];own,enemy=players[s],players[1-s]
    proposals=[]
    for target,load,sites,features in carriers(players,1-s):
        probs=distribution(features)
        for uid,(pos,cargo,kind) in own[2].items():
            if kind!='COLLECTOR' or cargo>=60 or _D[pos,enemy[2][target][0]]>4:continue
            scores={}
            for cmd,cell in options(obs,orders,uid):
                # Only shift our prospective location and the modeled carrier; other units remain at their public positions.
                next_own=dict(own[2]); next_own[uid]=[cell,cargo,kind]
                possible=np.zeros(225)
                for p,next_pos in zip(probs,sites):
                    if next_pos in enemy[1].values():continue
                    workers=dict(enemy[2]);workers[target]=[int(next_pos),load,'COLLECTOR']
                    world=list(players);world[s]=[own[0],own[1],next_own,own[3]]
                    world[1-s]=[enemy[0],enemy[1],workers,enemy[3]]
                    for name,_,ends,fx in carriers(world,1-s):
                        if name==target:
                            possible[ends]+=p*distribution(fx);break
                # Select a single intended second destination without consulting the realized next state.
                legal=[int(z) for z in _TO[cell] if z not in enemy[1].values()]
                best=max(legal,key=lambda z:possible[z])
                # Moving gives up this turn's observed local harvest; actual cash settlement is measured later.
                harvest=(obs['carbon'][pos] or 0)*0.2 if not cmd else 0.
                scores[cmd]=(load*possible[best]+harvest,best)
            old=orders.get(uid) or None
            if old not in scores:continue
            new=max(scores,key=lambda c:scores[c][0]); gain=scores[new][0]-scores[old][0]
            if new!=old and gain>=5:
                proposals.append(dict(t=obs['step'],worker=uid,target=target,old=old or 'NONE',alternative=new or 'NONE',
                    gain=float(gain),target_load=load,planned_cell=scores[new][1],baseline_estimate=scores[old][0]))
    return max(proposals,key=lambda r:(r['gain'],r['worker'],r['target']),default=None)

def finish(obs,orders,case):
    s=obs['player'];uid=case['worker'];target=case['target']
    if uid not in obs['players'][s][2]:return orders
    for name,load,cells,x in carriers(obs['players'],1-s):
        if name!=target:continue
        opts=options(obs,orders,uid)
        if not opts:return orders
        prob=dict(zip(cells,distribution(x)))
        cmd,cell=max(opts,key=lambda v:prob.get(v[1],0))
        old=dest(obs['players'][s][2][uid][0],orders.get(uid))
        if (prob.get(cell,0)-prob.get(old,0))*load < 5:return orders
        result=dict(orders)
        if cmd:result[uid]=cmd
        else:result.pop(uid,None)
        return result
    return orders
