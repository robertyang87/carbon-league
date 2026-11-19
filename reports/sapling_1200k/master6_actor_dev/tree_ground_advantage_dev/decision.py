"""Estimate new tree ground receipts, accounting for overlapping collection radii."""
import numpy as np
from guard_tree import _D,_TO,_DIRECTION
FEATURES=['remaining lifetime','eligible neighboring cells','remembered neighbor carbon','last-seen fraction',
          'virgin ground carbon','ground carbon overlapping our trees','ground carbon overlapping enemy trees',
          'current friendly collectors competing for cells','enemy collector nearby','enemy planter nearby',
          'friendly planter density','friendly tree count','nearest friendly base','turn']
def update_memory(obs,mem):
    t=obs['step'];v=set()
    for p,_,role in obs['players'][obs['player']][2].values():
        if role=='COLLECTOR':v.add(p)
        else:v.update(int(x) for x in _TO[p][1:])
    for p in v:mem[p]=(float(obs['carbon'][p]),t)
def values(obs,mem,uid,cell,cfg):
    t=obs['step'];s=obs['player'];own,foe=obs['players'][s],obs['players'][1-s]
    trees={int(p):(i,a) for i,side in enumerate(obs['players']) for p,a in side[3].values() if a<cfg['treeLifespan']}
    if cell in trees or any(cell in v[1].values() for v in obs['players']):return None
    neighbors=[int(z) for z in _TO[cell][1:] if z not in trees]
    ground=[0.,0.,0.];known=0;total=0.;compete=0
    for q in neighbors:
        adj=[trees[z][0] for z in map(int,_TO[q][1:]) if z in trees]
        if q in mem:
            c,at=mem[q];dt=min(t-at,50)
            c=max(0.,min(100.,c*(max(0.,1.-cfg['cellAbsorptionRate']*len(adj)) if adj else 1.+cfg['regenRate'])**dt))
            known+=1
        else:c=min(100.,11.*1.03**min(t,80))
        total+=c
        if not adj:ground[0]+=c
        else:
            ground[1]+=c*adj.count(s)/len(adj);ground[2]+=c*adj.count(1-s)/len(adj)
        compete+=any(w[2]=='COLLECTOR' and w[0]==q for w in own[2].values())
    enemy_distance=lambda role:min((int(_D[cell,w[0]]) for w in foe[2].values() if w[2]==role),default=15)
    left=min(cfg['treeLifespan'],cfg['episodeSteps']-1-t)
    return [left/50.,len(neighbors),total/100.,known/4.,*(v/100. for v in ground),compete,
            min(5,enemy_distance('COLLECTOR')),min(5,enemy_distance('PLANTER')),
            sum(w[2]=='PLANTER' and _D[cell,w[0]]<=3 for k,w in own[2].items() if k!=uid),
            len(own[3]),min(int(_D[cell,p]) for p in own[1].values()),t/299]
def quote(obs,cfg):
    own=obs['players'][obs['player']]
    live=sum(a<cfg['treeLifespan'] for side in obs['players'] for _,a in side[3].values())
    return cfg['plantCost']+(cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**live if len(own[3])>cfg['playerTreeProtectiveNumber'] else 0)
def predicted(x,w):
    return max(0.,float(np.dot([1.,*x],w)))
def consider(obs,mem,joint,uid,cfg,w):
    s=obs['player']; own,opp=obs['players'][s],obs['players'][1-s]
    cell=own[2][uid][0];command=joint.get(uid)
    if command not in _DIRECTION[1:]:return None
    new=values(obs,mem,uid,cell,cfg);dest=int(_TO[cell,_DIRECTION.index(command)])
    move=values(obs,mem,uid,dest,cfg)
    if new is None or move is None:return None
    if any(v[2] in ('COLLECTOR','PLANTER') and _D[cell,v[0]]<=1 for v in opp[2].values()):return None
    if any(_TO[v[0],_DIRECTION.index(joint.get(k))]==cell for k,v in own[2].items() if k!=uid):return None
    cost=quote(obs,cfg); reserve=0.
    sites={c:i for i,side in enumerate(obs['players']) for c,a in side[3].values() if a<cfg['treeLifespan']}
    bases={c for side in obs['players'] for c in side[1].values()}
    for k,(p,_,kind) in own[2].items():
        if k!=uid and kind=='PLANTER' and not joint.get(k) and p not in bases:
            reserve+=cost if p not in sites else (cfg['seizeCost'] if sites[p]!=s else 0)
    for base in own[1]:
        hire=joint.get(base)
        if hire:
            kind=hire[3:].title();n=sum(v[2]==kind.upper() for side in obs['players'] for v in side[2].values())
            reserve+=cfg['rec'+kind+'Cost']+n*cfg['rec'+kind+'IncreaseCost']
    if own[0]<cost+reserve:return None
    here,there=predicted(new,w),predicted(move,w)
    # Conservative alternative: grant the parent the option to collect the full predicted net if it buys there.
    gain=here-cost-max(0.,there-cost)
    return dict(worker=uid,cell=cell,destination=dest,quote=cost,ground_stay=here,ground_next=there,score=gain,accept=gain>0.)
