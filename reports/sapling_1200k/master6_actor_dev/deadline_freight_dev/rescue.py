"""Route a carrier while there is still time to deliver after one detour."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'intercept_response_dev'))
from forecast import _D,_TO,_DIRECTION

def choose(obs,orders,focus=None):
    seat=obs['player'];own,enemy=obs['players'][seat],obs['players'][1-seat]
    bases=list(own[1].values()); enemies=list(enemy[1].values()); moves=[]
    for uid,(pos,load,kind) in own[2].items():
        if kind!='COLLECTOR' or load<40 or (focus and focus!=uid):continue
        near=lambda cell:min(int(_D[cell,b]) for b in bases)
        d=near(pos);left=299-obs['step']
        if not 0<d<=left<=d+3:continue
        blockers={_TO[w[0],_DIRECTION.index(orders.get(k))] for k,w in own[2].items() if k!=uid}
        blockers.update(b for k,b in own[1].items() if orders.get(k))
        danger=[w[0] for w in enemy[2].values() if w[2]=='COLLECTOR' and w[1]<=load]
        choices=[]
        for action,cell in zip(_DIRECTION,_TO[pos]):
            if cell in blockers or cell in enemies or near(cell)>=d:continue
            if cell not in bases and any(_D[cell,p]<=1 for p in danger):continue
            exits=sum(near(z)<near(cell) and all(_D[z,p]>1 for p in danger) for z in _TO[cell])
            choices.append((exits,action==orders.get(uid),action))
        if not choices:continue
        _,_,best=max(choices)
        old=orders.get(uid)
        if old!=best:
            moves.append(dict(t=obs['step'],worker=uid,cargo=load,old=old or 'NONE',alternative=best,tree='',
                              target=min(enemy[2],key=lambda k:_D[pos,enemy[2][k][0]],default=''),distance=d))
    return max(moves,key=lambda v:(v['cargo'],v['worker']),default=None)
