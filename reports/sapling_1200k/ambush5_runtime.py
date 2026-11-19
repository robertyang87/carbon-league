"""Build-time fragment after the parent actor and frozen forecast lookup."""

class _FrozenForecast:
    min_probability = .15
    min_expected_cargo = 25

    def crossings(self,obs,previous,uid):
        enemy=obs['players'][1-obs['player']]
        pos=enemy[2][uid][0]
        axes,coarse,detailed=context(obs,previous,uid)
        output={}
        for depth in (3,4,5):
            options=_DETAILED.get((depth,detailed),_COARSE.get((depth,coarse),_BASE[depth]))
            for offset,p in options:
                dest=project(pos,offset,axes)
                if dest not in enemy[1].values():
                    output[depth,dest]=p
        return output


_AMBUSH_MODEL=_FrozenForecast()
_PARENT_AGENT=agent
_AMBUSH_SEATS={}


def _next_intercept(obs,previous,base,plan):
    """Re-observe both workers and ally reservations before every additional order."""
    own,foe=obs['players'][obs['player']],obs['players'][1-obs['player']]
    a=own[2].get(plan['attacker'])
    f=foe[2].get(plan['target'])
    left=plan['end']-obs['step']
    if a is None or f is None or left<=0:
        return None
    if f[1]<50 or a[1]>=f[1] or DIST[f[0]][plan['destination']]>left:
        return None
    last=previous.get(plan['target'])
    if last is not None and last[1]>0 and f[1]+.01<last[1]:
        return None
    dest=plan['destination']
    obstacles={DEST[p][intent.MOVE_ID[base.get(wid) or 'NONE']] for wid,(p,_,_) in own[2].items() if wid!=plan['attacker']}
    obstacles.update(p for bid,p in own[1].items() if base.get(bid) in ('RECCOLLECTOR','RECPLANTER'))
    if a[0]==dest:
        return None if dest in obstacles else 'NONE'
    route=approach(a[0],dest,left,set(foe[1].values()),obstacles,base.get(plan['attacker']))
    return MOVES[route[0]] if route else None


def _ambush_update(obs,base):
    side,tick=obs['player'],obs['step']
    if side not in _AMBUSH_SEATS or tick!=_AMBUSH_SEATS[side]['step']+1:
        _AMBUSH_SEATS[side]=dict(step=tick-1,previous={},plan=None)
    state=_AMBUSH_SEATS[side]
    previous=state['previous']
    state['step']=tick
    phase='parent'
    action=None
    plan=state['plan']
    if plan is not None:
        action=_next_intercept(obs,previous,base,plan)
        if action is None:
            state['plan']=None
            phase='release'
        else:
            phase='continue'
    elif tick+5<=299:
        plan=choose(obs,previous,base,forecaster=_AMBUSH_MODEL)
        if plan:
            plan['end']=tick+plan['depth']
            state['plan']=plan
            action=plan['moves'][0]
            phase='start'
    state['previous']={uid:list(w) for uid,w in obs['players'][1-side][2].items()}
    orders=base.copy()
    if action is not None:
        if action=='NONE':
            orders.pop(plan['attacker'],None)
        else:
            orders[plan['attacker']]=action
    record={'phase':phase,'attacker':plan['attacker'] if plan else None,
            'target':plan['target'] if plan else None,'destination':plan['destination'] if plan else None,
            'changed':any((orders.get(k) or None)!=(base.get(k) or None) for k in set(orders)|set(base))}
    return orders,record


def agent(observation,configuration):
    base=_PARENT_AGENT(observation,configuration)
    orders,record=_ambush_update(observation,base)
    agent.last_baseline=base
    agent.last_decision=record
    return orders
