"""Meet an approaching enemy planter on the owned tree before it can stay to claim it."""
from guard_tree import _D, _DIRECTION, _TO


def preempt(obs, commands, cfg, prior):
    seat=obs['player']
    own,enemy=obs['players'][seat],obs['players'][1-seat]
    if enemy[0]<cfg['seizeCost']:
        return None
    final={uid:int(_TO[v[0],_DIRECTION.index(commands.get(uid))]) for uid,v in own[2].items()}
    options=[]
    for tree,(cell,age) in own[3].items():
        if cell in final.values() or age>=cfg['treeLifespan']-1:
            continue
        for foe,(pos,_,kind) in enemy[2].items():
            if kind!='PLANTER' or _D[pos,cell]!=1 or foe not in prior:
                continue
            if _D[prior[foe][0],cell]!=2:
                continue
            for uid,(here,cargo,role) in own[2].items():
                if role!='COLLECTOR' or _D[here,cell]!=1:
                    continue
                if final[uid] in own[1].values() and cargo>cfg['smeltCost']:
                    continue
                if any(c=='COLLECTOR' and load<=cargo and _D[p,cell]<=1 for p,load,c in enemy[2].values()):
                    continue
                action=_DIRECTION[list(_TO[here]).index(cell)]
                options.append((age,cargo,tree,uid,foe,action))
    if not options:return None
    age,cargo,tree,uid,foe,action=min(options)
    return dict(tree=tree,age=age,cargo=cargo,uid=uid,enemy=foe,before=commands.get(uid),after=action)
