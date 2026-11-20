"""Additional competition for an unaffordable-in-value new planting when nearby enemy-tree pursuit is unavailable."""
_TREE_LOCAL_FIRST=_tree_assess

def _tree_assess(obs,cfg,base,history):
    initial=_TREE_LOCAL_FIRST(obs,cfg,base,history)
    if initial:return initial
    seat=obs["player"];ps=obs["players"];own,enemy=ps[seat],ps[1-seat];n=cfg["size"]
    price=_TREE_LIB["quoted_price"](ps,seat,cfg);forbidden={v[0] for p in ps for v in p[3].values()} | {pos for p in ps for pos in p[1].values()}
    if own[0]<price or not own[2]:return None
    candidates=[]
    enemy_collectors=[w[0] for w in enemy[2].values() if w[2]=="COLLECTOR"]
    for uid,(pos,carry,role) in own[2].items():
        if role!="PLANTER" or pos in forbidden:continue
        occupied={_tree_dest(v[0],base.get(k,"NONE"),n) for k,v in own[2].items() if k!=uid}
        occupied.update(v for k,v in own[1].items() if k in base)
        if pos in occupied:continue
        x=_TREE_LIB["features"](obs,uid,cfg,history)
        z=[(v-m)/s for v,m,s in zip(x,_TREE_CASH["feature_mean"],_TREE_CASH["feature_scale"])]
        net,hold=[mu+sum(a*w[j] for a,w in zip(z,_TREE_CASH["coef"])) for j,mu in enumerate(_TREE_CASH["means"])]
        if net>=0:continue
        safe=[]
        for j,a in enumerate(_ACTIONS):
            q=_tree_dest(pos,a,n)
            if q in forbidden or q in occupied:continue
            d=min((_TREE_LIB["distance"](q,e,n) for e in enemy_collectors),default=n)
            safe.append((d,a==base.get(uid,"NONE"),-j,a))
        if not safe:continue
        chosen=max(safe)[-1]
        plant=base.copy();plant.pop(uid,None)
        wait=base.copy();wait[uid]=chosen
        risk=_TREE_LIB["predict"](x,_TREE_RISK)
        candidates.append((-net,risk,uid,dict(worker=uid,cell=pos,target=pos,age=None,route_distance=None,far_investment=True,
                           roots={"plant":plant,"defer":wait},est_net=net,est_hold=hold,failure=risk)))
    return max(candidates,key=lambda a:a[:3])[-1] if candidates else None
