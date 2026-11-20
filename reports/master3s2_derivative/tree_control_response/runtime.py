"""Offer competing native-search choices to one planter, without suppressing planting."""
_ACTIONS=("UP","RIGHT","DOWN","LEFT")


def _tree_dest(cell,action,n):
    return cell if action=="NONE" else _TREE_LIB["neighbors"](cell,n)[_ACTIONS.index(action)]


def _tree_assess(obs,cfg,base,history):
    ps,me,n=obs["players"],obs["player"],cfg["size"]; own,enemy=ps[me],ps[1-me]
    distance=_TREE_LIB["distance"]
    all_bases=set(own[1].values())|set(enemy[1].values())
    own_trees={r[0] for r in own[3].values()}
    enemy_trees={r[0]:r[1] for r in enemy[3].values() if r[1]<cfg["treeLifespan"]-1}
    all_trees={r[0] for p in ps for r in p[3].values()}
    plans=[]
    for uid,(pos,carried,kind) in own[2].items():
        if kind!="PLANTER" or pos in all_bases or pos in own_trees:continue
        held=base.get(uid,"NONE")
        occupied={_tree_dest(w[0],base.get(other,"NONE"),n) for other,w in own[2].items() if other!=uid}
        occupied.update(x for who,x in own[1].items() if who in base)
        targets=sorted((distance(pos,cell,n),cell,age) for cell,age in enemy_trees.items()
                       if distance(pos,cell,n)<=2 and cfg["treeLifespan"]-age>distance(pos,cell,n)+1)
        if not targets:continue
        target=targets[0];d,spot,age=target
        if own[0]<cfg["seizeCost"] or min(cfg["episodeSteps"]-1-obs["step"],cfg["treeLifespan"]-age)<=d:continue
        options={}
        def offer(label,action):
            if _tree_dest(pos,action,n) not in occupied:
                z=base.copy()
                if action=="NONE":z.pop(uid,None)
                else:z[uid]=action
                options[label]=z
        if d==0:
            offer("seize","NONE")
            if held=="NONE" or not options:continue
            plans.append((2,0,uid,dict(worker=uid,cell=pos,target=spot,age=age,route_distance=d,roots=options,est_net=None,est_hold=None,failure=None)))
            continue
        if pos in all_trees or own[0]<_TREE_LIB["quoted_price"](ps,me,cfg):continue
        x=_TREE_LIB["features"](obs,uid,cfg,history);risk=_TREE_LIB["predict"](x,_TREE_RISK)
        z=[(v-m)/s for v,m,s in zip(x,_TREE_CASH["feature_mean"],_TREE_CASH["feature_scale"])]
        net,hold=[mu+sum(a*w[j] for a,w in zip(z,_TREE_CASH["coef"])) for j,mu in enumerate(_TREE_CASH["means"])]
        offer("plant","NONE")
        directions=sorted((distance(_tree_dest(pos,a,n),spot,n),i,a) for i,a in enumerate(_ACTIONS)
                          if _tree_dest(pos,a,n) not in occupied and _tree_dest(pos,a,n) not in all_bases)
        forward=next((a for length,_,a in directions if length<d),None)
        if forward:offer("pursue",forward)
        # Prefer a quieter retreat occupied by neither allies' selected actions nor adjacent opponent collectors.
        enemy_collectors=[w[0] for w in enemy[2].values() if w[2]=="COLLECTOR"]
        delay=sorted((-min((distance(_tree_dest(pos,a,n),p,n) for p in enemy_collectors),default=n), i,a) for length,i,a in directions if a!=forward)
        if delay:offer("defer",delay[0][2])
        if "plant" not in options or "pursue" not in options:continue
        priority=risk-net/200-min(50,max(0,hold))/100
        plans.append((1,priority,uid,dict(worker=uid,cell=pos,target=spot,age=age,route_distance=d,roots=options,est_net=net,est_hold=hold,failure=risk)))
    if not plans:return None
    return max(plans,key=lambda p:(p[0],p[1],p[2]))[-1]


def _tree_menu_impl(obs,config,act0,cands,most):
    global _TREE_LAST
    step=obs["step"];seat=obs["player"]
    hist=_TREE_HIST.setdefault(seat,{})
    if step==0 or hist and max(hist)>=step:hist.clear()
    hx={k:hist[step-k] for k in (3,5) if step-k in hist}
    hist[step]=obs["players"]
    for old in list(hist):
        if old<step-5:del hist[old]
    _TREE_LAST=None
    if most<3 or not obs["players"][1-seat][3]:return
    plan=_tree_assess(obs,config,act0,hx)
    if not plan:return
    lookup=lambda a:tuple(sorted(a.items()))
    original=cands.copy()
    proposed=plan["roots"]
    labels=("seize",) if "seize" in proposed else (("pursue","plant","defer") if plan["est_net"]<=0 else ("plant","pursue","defer"))
    keep=[0]
    for label in labels:
        if label in proposed and proposed[label] in cands:keep.append(cands.index(proposed[label]))
    for label in labels:
        if label not in proposed:continue
        a=proposed[label]
        if a in cands:continue
        if len(cands)<most:
            cands.append(a);keep.append(len(cands)-1)
        else:
            ix=next((i for i in reversed(range(1,len(cands))) if i not in keep),None)
            if ix is None:continue
            cands[ix]=a;keep.append(ix)
    _TREE_LAST=dict(step=step,plan=plan,before=original,after=cands.copy(),new=[k for k,v in proposed.items() if v in cands and v not in original])


_TREE_HIST={}
_TREE_LAST=None
