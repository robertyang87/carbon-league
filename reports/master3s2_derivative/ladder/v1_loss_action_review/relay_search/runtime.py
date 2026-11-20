"""Finish the existing search first. At most one additional complete team order can be evaluated."""
_RELAY_ON = True
_RELAY_LAST = {}
_RELAY_PARENT_SEARCH = _search


def _relay_choice(players, me, step, order=None):
    """Return public geometric opportunities. With no order, compute only a necessary root screen."""
    remaining = LAST_TURN + 1 - step
    if not 2 <= remaining <= 5:
        return None
    team = players[me]
    bases = set(team[1].values())
    if not bases:
        return None
    workers = team[2]
    foes = players[1-me]
    home = lambda z: min(dist(z, base) for base in bases)
    best = None
    for donor, (pos, cargo, role) in workers.items():
        if role != "COLLECTOR" or cargo < 30 or home(pos) < remaining-1:
            continue
        if order is not None and home(move(pos, WIDX[order.get(donor,"NONE")])) <= remaining-1:
            continue
        for recipient, (rpos, rcargo, rrole) in workers.items():
            if rrole != "COLLECTOR" or rcargo >= cargo or dist(pos,rpos) > 2:
                continue
            rmoves = range(5) if order is None else [WIDX[order.get(recipient,"NONE")]]
            for rd in rmoves:
                target = move(rpos,rd)
                if target in bases or target in foes[1].values() or home(target) >= home(rpos) or home(target) > remaining-1:
                    continue
                if dist(pos,target)>1:
                    continue
                if any(w[2]=="COLLECTOR" and w[1]<=rcargo and dist(w[0],target)<=1 for w in foes[2].values()):
                    continue
                if order is None:
                    return True
                # Recruitment at a friendly base cannot enter this non-base rendezvous.
                # Keep every other order, including every base. Acceptance requires one living receiver at the root.
                if any(move(w[0],WIDX[order.get(uid,"NONE")]) == target for uid,w in workers.items() if uid not in (donor,recipient)):
                    continue
                direction = next(k for k in range(5) if move(pos,k)==target)
                if direction == WIDX[order.get(donor,"NONE")]:
                    continue
                rank = (-cargo, home(target), donor, recipient, direction)
                if best is None or rank < best[0]:
                    best = rank, donor,recipient,direction,target,cargo
    if best is None:
        return None
    _, donor, recipient, direction, target, cargo = best
    other = dict(order)
    if direction:
        other[donor]=WACT[direction]
    else:
        other.pop(donor,None)
    return dict(order=other, donor=donor, recipient=recipient, target=target, cargo=cargo)


def _relay_leaf(obs, cfg, est, prev, selected, alt, t0):
    """Compare to the official end using an identical, independent original opponent draw at the root."""
    step = obs["step"]; me = obs["player"]; enemy=1-me
    remaining=LAST_TURN+1-step
    deadline=t0+_S_ABORT
    if deadline-_time.perf_counter() < .025+.035*remaining:
        return None,"budget"
    root=dict(players=obs["players"],trees=obs["trees"],step=step,player=me,carbon=list(map(float,est)),
              remainingOverageTime=obs.get("remainingOverageTime",0))
    seed=int(1000003*(step+1)+17*me)%(2**31)
    levels=[1.]*N
    x=_quant(encode(root["players"],step,enemy,est,levels,prev))[None].astype(np.float32)
    w,b,_=_net()(x,_SID)
    w,b=_opponent_root_readout(x,w,b,root["players"],enemy)
    opp=_sample(root["players"],enemy,w[0],b[0],np.random.RandomState(seed+7),step)
    orders=[]
    for ours in (selected,alt["order"]):
        pair=[None,None];pair[me]=ours;pair[enemy]=opp;orders.append(pair)
    states=[root,root]
    def cost(name):
        return float(cfg.get(name,30) if hasattr(cfg,"get") else getattr(cfg,name,30))
    recruit=min(cost("recCollectorCost"),cost("recPlanterCost"))
    holder=alt["donor"]; banked=False
    for depth in range(remaining):
        if deadline-_time.perf_counter() < .010:
            return None,"budget"
        before=states; states=[]
        for i in (0,1):
            nxt=_turn(before[i],cfg,orders[i]);states.append(nxt)
            # Extinct reward includes the incoming wallet. Preserve the parent if this different event occurs.
            if any(not pl[2] and not pl[3] and (not pl[1] or pl[0]<recruit) for pl in nxt["players"]):
                return None,"simulated_extinction"
        if not banked:
            current=before[1]["players"][me][2]
            if holder not in current:
                return None,"cargo_untracked"
            dest=move(current[holder][0],WIDX[orders[1][me].get(holder,"NONE")])
            after=states[1]["players"][me]
            if dest in after[1].values():
                # A surviving own collector at own base with cargo cleared rules out an enemy interception.
                banked=any(w[0]==dest and w[2]=="COLLECTOR" and w[1]==0 for w in after[2].values())
                if not banked:
                    return None,"recipient_lost"
            else:
                at=[uid for uid,w in after[2].items() if w[0]==dest and w[2]=="COLLECTOR" and w[1]>0]
                if len(at)!=1 or (depth==0 and at[0]!=alt["recipient"]):
                    return None,"recipient_lost"
                holder=at[0]
        if depth+1==remaining:
            break
        if deadline-_time.perf_counter()<.032:
            return None,"budget"
        X=[]
        for i,st in enumerate(states):
            for side in (me,enemy):
                X.append(_quant(encode(st["players"],st["step"],side,st["carbon"],levels,before[i]["players"])))
        w,b,_=_RNET(np.asarray(X,dtype=np.float32),_SID)
        if _RNET is not _NET:
            w=w/_S_RTEMP; b=b/_S_RTEMP
        for i,st in enumerate(states):
            for j,side in enumerate((me,enemy)):
                rng=np.random.RandomState(seed+7919*(depth+1)+j+1)
                orders[i][side]=_sample(st["players"],side,w[2*i+j],b[2*i+j],rng,st["step"])
    if _time.perf_counter()>=deadline:
        return None,"budget"
    margins=[float(st["players"][me][0]-st["players"][enemy][0]) for st in states]
    return margins,"banked" if banked else "not_banked"


def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):
    global _RELAY_LAST
    selected=_RELAY_PARENT_SEARCH(obs,configuration,est,act0,lw0,lb0,prev,t0,home)
    _RELAY_LAST={}
    if not _RELAY_ON or not _SEARCH_OK or not 2<=LAST_TURN+1-obs["step"]<=5:
        return selected
    start=_time.perf_counter()
    try:
        alt=_relay_choice(obs["players"],obs["player"],obs["step"],selected)
        if alt is None:
            return selected
        values,reason=_relay_leaf(obs,configuration,est,prev,selected,alt,t0)
        emit=reason=="banked" and values is not None and values[1]>values[0]+.01
        _RELAY_LAST=dict(parent=selected, proposal=alt, condition=reason, projected_official_cash_margins=values,
                         elapsed=_time.perf_counter()-start,emitted=emit)
        return alt["order"] if emit else selected
    except Exception as error:
        _RELAY_LAST=dict(condition="error",error=type(error).__name__,elapsed=_time.perf_counter()-start,emitted=False)
        return selected
