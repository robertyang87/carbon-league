# Bound the tactical proposal to one changed worker in an otherwise complete native order.
_V3_AUDIT = None


def _v3_intercept(players, me, previous, step, parent, protected, logits, predicted):
    if previous is None or step > LAST_TURN-2:
        return None
    team, foes=players[me],players[1-me]
    if not team[2] or not foes[1]:
        return None
    ours=team[2]; enemies=foes[2]; bases=set(foes[1].values())
    ally_dest={u:move(w[0],WIDX[parent.get(u,"NONE")]) for u,w in ours.items()}
    occupied_bases={pos for uid,pos in team[1].items() if parent.get(uid) in ("RECCOLLECTOR","RECPLANTER")}
    predictions={u:move(w[0],WIDX[predicted.get(u,"NONE")]) for u,w in enemies.items()}
    probability={}
    def hits(uid, position, target):
        if dist(position,target)>1:
            return 0.
        if uid not in probability:
            probability[uid]=_probs(logits[position])
        direction=next(i for i in range(5) if move(position,i)==target)
        return float(probability[uid][direction])
    def km(a,b):
        return int(FEAT_DIST[a,b])
    best=None
    for enemy,(ep,load,kind) in enemies.items():
        if kind!="COLLECTOR" or load<100:
            continue
        homes=sorted((km(ep,b),b) for b in bases)
        distance,home=homes[0]
        if not 1<=distance<=6 or len(homes)>1 and homes[1][0]==distance:
            continue
        forecasts=[]
        for direction in range(1,5):
            nxt=move(ep,direction)
            if nxt not in bases and km(nxt,home)<distance:
                p=hits(enemy,ep,nxt)
                if nxt==predictions[enemy] or p>=.15:
                    forecasts.append(("direct",nxt,max(.18,p),1))
        old=previous[1-me][2].get(enemy)
        if old is not None and 2<=distance<=5 and km(old[0],home)>distance:
            entries=[move(home,d) for d in range(1,5) if km(ep,move(home,d))==distance-1]
            if len(entries)==1:
                toward=sum(hits(enemy,ep,move(ep,d)) for d in range(1,5) if km(move(ep,d),home)<distance)
                if toward>=.35 or km(predictions[enemy],home)<distance:
                    forecasts.append(("entry",entries[0],max(.18,toward),distance-1))
        for mode,target,chance,horizon in forecasts:
            for worker,(pos,cargo,role) in ours.items():
                if role!="COLLECTOR" or worker in protected or load<=cargo or load<1.25*cargo+15:
                    continue
                eta=km(pos,target)
                if eta>horizon or mode=="entry" and (cargo>35 or eta>2):
                    continue
                if parent.get(worker) and ally_dest[worker] in team[1].values() and cargo>0:
                    continue
                occupied=occupied_bases|{z for u,z in ally_dest.items() if u!=worker}
                for direction in range(5):
                    nxt=move(pos,direction)
                    if (nxt!=target if eta<=1 else km(nxt,target)!=eta-1):
                        continue
                    if nxt in bases or nxt in occupied or nxt==ally_dest[worker]:
                        continue
                    threats=[(u,w) for u,w in enemies.items() if w[2]=="COLLECTOR" and w[1]<=cargo and km(w[0],nxt)<=1]
                    if any(predictions[u]==nxt for u,_ in threats):
                        continue
                    risk=min(1.,sum(hits(u,w[0],nxt) for u,w in threats))
                    if risk>.4:
                        continue
                    if not team[1] or min(km(nxt,b) for b in team[1].values())>LAST_TURN-step:
                        continue
                    merit=chance*load/(1+max(0,horizon-1))-(cargo+30)*risk
                    rank=(-merit,eta,worker,enemy,direction)
                    if merit>0 and (best is None or rank<best[0]):
                        best=(rank,worker,enemy,direction,target,mode,chance,risk)
    if best is None:
        return None
    _,worker,enemy,direction,target,mode,chance,risk=best
    choice=dict(parent)
    if direction:
        choice[worker]=WACT[direction]
    else:
        choice.pop(worker,None)
    return choice,dict(worker=worker,enemy=enemy,direction=WACT[direction],target=target,
                       mode=mode,root_return_probability=chance,other_enemy_collision_risk=risk)
