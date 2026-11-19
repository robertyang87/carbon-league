# The victim has not yet seen our pending command. One forecast is shared by all hunter alternatives.
from collections import Counter

_PRESSURE_COUNTS=Counter()
_PRESSURE_LAST=None
_PRESSURE_HISTORY={}
_pressure_agent=agent


def agent(observation,configuration=None):
    global _PRESSURE_HISTORY,_PRESSURE_OBS,_PRESSURE_HOME,_PRESSURE_LAST
    seat=observation["player"];step=observation["step"]
    _PRESSURE_OBS={k:observation[k] for k in ("players","player","step")}
    own,other=observation["players"][seat],observation["players"][1-seat]
    if step==0 or getattr(agent,"seat",None)!=seat:
        _PRESSURE_COUNTS.clear();_PRESSURE_HISTORY={}
    agent.seat=seat
    _PRESSURE_HOME=_PRESSURE_DIST[:,list(other[1].values())].min(axis=1)
    updated={}
    for uid,(pos,quantity,role) in other[2].items():
        if role!="COLLECTOR":continue
        last_move=5;streak=0
        if uid in _PRESSURE_HISTORY:
            previous,_,old_streak=_PRESSURE_HISTORY[uid]
            legal=np.flatnonzero(_PRESSURE_DEST[previous]==pos)
            if len(legal):last_move=int(legal[0])
            if _PRESSURE_HOME[pos]<_PRESSURE_HOME[previous]:streak=min(3,old_streak+1)
        updated[uid]=(int(pos),last_move,streak)
    _PRESSURE_HISTORY=updated
    _PRESSURE_LAST=None
    return _pressure_agent(observation,configuration)


def _pressure_challenge(F,x,prefix,slot,normal,logits,legal,changes):
    if changes or slot>=10 or x["tok_kind"][slot]!=0:return normal
    seat=_PRESSURE_OBS["player"];players=_PRESSURE_OBS["players"]
    uid=_FEAT.own_uid[slot];pos,cargo,_=players[seat][2][uid]
    canonical_pos=int(x["tok_pos"][slot]);dst=F.DEST[canonical_pos]
    raw=F.CANON[dst] if seat else dst
    a=prefix.a; viable=legal & (prefix.reserved[dst]==0) & (prefix.undecided[dst]==0)
    viable &= ~a["en_base"][dst] & (a["en_min"][dst]>float(cargo))
    viable &= (logits[normal]-logits)<=1.1
    if not viable[normal] or viable.sum()<2:return normal
    if cargo>0 and a["own_base"][dst[normal]]:
        return normal
    bank=a["dob"][dst]; carry_cost=.20*float(cargo)*(bank-bank.min())
    bank_now=(a["own_base"][dst]*cargo).astype(float)
    harvest=np.array([float(_FEAT.est[pos])*.18 if k==0 and not a["own_base"][dst[k]] else 0.
                      for k in range(5)])
    own_value=bank_now+harvest-carry_cost
    gross=np.zeros(5);probs=np.zeros(5);victims=[None]*5
    for victim,(cell,last,streak) in _PRESSURE_HISTORY.items():
        load=players[1-seat][2][victim][1]
        if load<50 or load<=cargo or _PRESSURE_DIST[pos,cell]>2:continue
        feats=pressure_features(pos,cell,load,last,streak,_PRESSURE_OBS["step"],_PRESSURE_HOME)
        scores=feats@_PRESSURE_COEF;scores-=scores.max();p=np.exp(scores);p/=p.sum()
        crossing=_PRESSURE_DEST[cell]
        # No credit for a collision on the victim's base. Its own delivery window raises the value at risk.
        threat=load*(.42+.58/(1.+float(_PRESSURE_HOME[cell])/3.))
        for k,target in enumerate(raw):
            matching=np.flatnonzero(crossing==target)
            if not len(matching) or _PRESSURE_HOME[target]==0:continue
            probability=float(p[int(matching[0])]);benefit=threat*probability
            if benefit>gross[k]:
                gross[k],probs[k],victims[k]=benefit,probability,victim
    if not any(victims):return normal
    _PRESSURE_COUNTS["comparable_hunter_slots"]+=1
    total=gross+own_value
    alt=int(np.argmax(np.where(viable,total,-np.inf)))
    gain=float(total[alt]-total[normal])
    if alt==normal or gain<12. or probs[alt]<.16:return normal
    changes.append((slot,normal,alt,{"hunter":uid,"victim":victims[alt],"victim_probability":float(probs[alt]),
                                     "public_victim_load":float(players[1-seat][2][victims[alt]][1]),
                                     "predicted_enemy_delivery_at_risk_gain":float(gross[alt]-gross[normal]),
                                     "own_delivery_and_collection_change":float(own_value[alt]-own_value[normal]),
                                     "pressure_score_gain":gain,"policy_logit_gap":float(logits[normal]-logits[alt])}))
    return alt


def _mixed_actions(x,aux):
    global _PRESSURE_LAST
    encoded=_pressure_encode(x)
    before={r:v.copy() for r,v in _COUNTS.items()}
    changes=[];proposed=_pressure_decode(x,aux,encoded,changes)
    if not changes:return proposed
    after={r:v.copy() for r,v in _COUNTS.items()}
    _COUNTS.update(before)
    base=_pressure_decode(x,aux,encoded)
    slot,old,new,detail=changes[0]
    if np.count_nonzero(base!=proposed)!=1 or not _return_joint_ok(sys.modules["cnet2_feat"],x,aux,base,proposed,[(slot,old,new)]):
        _PRESSURE_COUNTS["joint_veto"]+=1
        return base
    _COUNTS.update(after)
    _PRESSURE_COUNTS["changed_roots"]+=1
    _PRESSURE_LAST=detail
    return proposed
