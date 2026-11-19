"""Choose equally short, collision-safe homebound moves by their next exposed freight toll."""

def choose(obs, commands, cfg, dist, dest, directions):
    seat=obs['player']
    own,foe=obs['players'][seat],obs['players'][1-seat]
    home=list(own[1].values())
    if not home:return None
    trees=[p for p,age in foe[3].values() if age<cfg['treeLifespan']]
    if not trees:return None
    lands={uid:int(dest[w[0],directions.index(commands.get(uid))]) for uid,w in own[2].items()}
    bases=set(home)|set(foe[1].values())
    options=[]
    for uid,(p,cargo,role) in own[2].items():
        if role!='COLLECTOR' or cargo<50:continue
        d=int(min(dist[p,b] for b in home)); old=lands[uid]
        if old in bases or min(dist[old,b] for b in home)!=d-1:continue
        used={v for k,v in lands.items() if k!=uid}
        def toll(c):
            return sum(max(min(abs(c%15-z%15),15-abs(c%15-z%15)),
                           min(abs(c//15-z//15),15-abs(c//15-z//15)))<=1 for z in trees)
        was=toll(old)
        if not was:continue
        for i,z in enumerate(dest[p]):
            z=int(z)
            if z in used|bases or z==old or min(dist[z,b] for b in home)!=d-1:continue
            if any(k=='COLLECTOR' and load<=cargo and dist[z,pos]<=1 for pos,load,k in foe[2].values()):continue
            now=toll(z);saving=cargo*cfg['collectorAbsorptionRate']*(was-now)
            if saving>=5:options.append((-saving,uid,i,was,now,float(cargo),int(old),z))
    if not options:return None
    neg,uid,i,was,now,cargo,old,z=min(options)
    return dict(worker=uid,old=commands[uid],alternative=directions[i],old_trees=int(was),
                new_trees=int(now),load=cargo,old_cell=old,new_cell=z,nominal_saving=float(-neg))
