"""Price safe publicly supported fresh sites along the parent's movement, up to three moves in total."""
from fit import HERE,sr
from decision import _TO,_D,_DIRECTION,predicted,values,quote
from collections import Counter
def further(obs,mem,joint,uid,cfg,w):
    s=obs['player'];own=obs['players'][s];foe=obs['players'][1-s]
    if joint.get(uid) not in _DIRECTION[1:]:return dict(sites=[],excluded={})
    base=own[2][uid][0];first=int(_TO[base,_DIRECTION.index(joint[uid])]);n=Counter()
    planned={int(_TO[v[0],_DIRECTION.index(joint.get(k))]) for k,v in own[2].items() if k!=uid}
    if first in planned:return dict(sites=[],excluded={'teammate':1})
    if any(_D[v[0],first]<=1 for v in foe[2].values()):return dict(sites=[],excluded={'unsafe_first':1})
    active={p for side in obs['players'] for p,age in side[3].values() if age<cfg['treeLifespan']}
    enemy_bases=set(foe[1].values());level=[(first,{base,first})];candidates={}
    for depth in (2,3):
        nextlevel=[]
        for pos,history in level:
            for z in map(int,_TO[pos,1:]):
                if z in history or z in enemy_bases:continue
                if any(_D[v[0],z]<=depth for v in foe[2].values()):n['enemy']+=1;continue
                nextlevel.append((z,history|{z}))
                if z in candidates:continue
                x=values(obs,mem,uid,z,cfg)
                if x is None:n['occupied']+=1;continue
                neighbors=[int(k) for k in _TO[z,1:] if k not in active]
                if not neighbors or not all(k in mem for k in neighbors) or not any(obs['step']-mem[k][1]<=50 for k in neighbors):n['insufficient_view']+=1;continue
                life=min(cfg['treeLifespan'],cfg['episodeSteps']-1-obs['step'])
                gross=predicted(x,w)*max(0,life-depth)/max(life,1)
                candidates[z]=dict(cell=z,steps=depth,discounted_ground=gross,net=gross-quote(obs,cfg))
        level=nextlevel
    return dict(sites=sorted(candidates.values(),key=lambda c:(-c['net'],c['cell'])),excluded=dict(n))
