"""At first approach to a tree, put a collector within reach without guessing the enemy's current move."""
from guard_tree import _D,_TO,_DIRECTION

def choose(obs,cmd,cfg,prev):
    seat=obs['player'];own=obs['players'][seat];enemy=obs['players'][1-seat]
    land={k:int(_TO[v[0],_DIRECTION.index(cmd.get(k))]) for k,v in own[2].items()}
    hire={v for k,v in own[1].items() if cmd.get(k)}
    opts=[]
    for tree,(cell,age) in own[3].items():
        if age>=cfg['treeLifespan']-2 or enemy[0]<cfg['seizeCost']:continue
        if any(_D[cell,land[k]]<=1 for k,v in own[2].items() if v[2]=='COLLECTOR'):continue
        for foe,(f,_,kind) in enemy[2].items():
            if kind!='PLANTER' or _D[f,cell]!=1 or foe not in prev or _D[prev[foe][0],cell]<=1:continue
            for uid,(p,cargo,role) in own[2].items():
                if role!='COLLECTOR' or _D[p,cell]!=2:continue
                if land[uid] in own[1].values() and cargo>cfg['smeltCost']:continue
                route=[]
                for a,z in zip(_DIRECTION,_TO[p]):
                    if _D[z,cell]!=1 or z in hire or z in enemy[1].values() or any(v==z for k,v in land.items() if k!=uid):continue
                    if any(v[2]=='COLLECTOR' and v[1]<=cargo and _D[v[0],z]<=1 for v in enemy[2].values()):continue
                    route.append(a)
                if len(route)==1:opts.append((age,cargo,uid,tree,foe,route[0],cell))
    if not opts:return None
    age,cargo,uid,tree,foe,a,cell=min(opts)
    return dict(tree=tree,age=age,cargo=cargo,uid=uid,enemy=foe,before=cmd.get(uid),after=a,cell=cell)
