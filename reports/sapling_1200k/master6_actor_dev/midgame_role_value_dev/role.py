"""Single same-price substitution when two planters survive and an uncontested base can recruit."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'intercept_response_dev'))
from forecast import _TO,_DIRECTION,_D
def choose(obs,cmd,cfg):
    own,enemy=obs['players'][obs['player']],obs['players'][1-obs['player']]
    if sum(v[2]=='PLANTER' for v in own[2].values())<2:return None
    lands={_TO[w[0],_DIRECTION.index(cmd.get(k))] for k,w in own[2].items()}
    roles=[w[2] for p in obs['players'] for w in p[2].values()]
    costs={k:cfg['rec'+k.title()+'Cost']+roles.count(k)*cfg['rec'+k.title()+'IncreaseCost'] for k in ('COLLECTOR','PLANTER')}
    if costs['COLLECTOR']!=costs['PLANTER']:return None
    hires=sum(bool(cmd.get(k)) for k in own[1])
    if hires+len(own[2])>cfg['workerLimit'] or own[0]<hires*costs['COLLECTOR']:return None
    for bid,cell in own[1].items():
        if cmd.get(bid)!='RECPLANTER' or cell in lands:continue
        if any(_D[cell,w[0]]<=1 for w in enemy[2].values()):continue
        return dict(t=obs['step'],worker=bid,old='RECPLANTER',alternative='RECCOLLECTOR',
                    cell=cell,own_planters=sum(w[2]=='PLANTER' for w in own[2].values()),own_collectors=sum(w[2]=='COLLECTOR' for w in own[2].values()),price=costs['COLLECTOR'])
    return None
