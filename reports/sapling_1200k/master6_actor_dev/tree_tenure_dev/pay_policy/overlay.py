"""Two chronological affordable planter opportunities. Appended to the frozen Sapling pack."""
from collections import Counter as _PayCounter

_PAY_BASE_AGENT = agent
_PAY_TRACE = []
_PAY_USED = set()
_PAY_VIEW = None


def _pay_select(x,aux,encoded,original,mode):
    view, cfg = _PAY_VIEW
    t = view['step']
    window = 0 if 45 <= t <= 125 else (1 if 150 <= t <= 235 else None)
    if window is None or window in _PAY_USED:
        return original
    s=view['player']; players=view['players']; own=players[s]
    F=sys.modules['cnet2_feat']; moves=F.MOVES
    base=_FEAT.to_actions(original)
    trees={pos:(who,age) for who,pl in enumerate(players) for pos,age in pl[3].values()}
    bases={p for pl in players for p in pl[1].values()}
    life=cfg['treeLifespan']; price=cfg['plantCost']
    if len(own[3])>cfg['playerTreeProtectiveNumber']:
        price+=cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**sum(age<life for _,age in trees.values())
    def quote(p):
        if p in bases:
            return None, None
        tree=trees.get(p)
        if tree and tree[1]<life:
            return (None,None) if tree[0]==s else ('seize',cfg['seizeCost'])
        return 'plant',price
    hires={}; requests={}; dest={}
    counts=_PayCounter(w[2] for pl in players for w in pl[2].values())
    available=own[0]; vacancies=max(0,cfg['workerLimit']-len(own[2]))
    for uid,cell in own[1].items():
        order=base.get(uid)
        if order not in ('RECCOLLECTOR','RECPLANTER'):
            continue
        typ='Collector' if order=='RECCOLLECTOR' else 'Planter'
        cost=cfg['rec'+typ+'Cost']+counts[typ.upper()]*cfg['rec'+typ+'IncreaseCost']
        if vacancies and available>=cost:
            hires[uid]=(cell,cost); available-=cost; vacancies-=1
    for uid,(p,_,role) in own[2].items():
        cmd=base.get(uid,'NONE')
        dest[uid]=int(F.DEST[p,moves.index(cmd)])
        if role=='PLANTER' and cmd=='NONE':
            _,price_worker=quote(p)
            if price_worker is not None:
                requests[uid]=price_worker
    before_conflicts=_return_conflicts(x,original)
    for uid in sorted(own[2]):
        p,_,role=own[2][uid]
        if role!='PLANTER':
            continue
        kind,cost=quote(p)
        if cost is None:
            continue
        others=sum(v for worker,v in requests.items() if worker!=uid)
        if available-others<cost:
            continue
        forbidden={v for worker,v in dest.items() if worker!=uid} | {v[0] for v in hires.values()}
        if p in forbidden:
            continue
        slot=next((i for i,u in enumerate(_FEAT.own_uid[:10]) if u==uid),None)
        if slot is None:
            continue
        old=base.get(uid,'NONE')
        for alternative in ([0] if old!='NONE' else [1,2,3,4]):
            if int(F.DEST[p,alternative]) in forbidden:
                continue
            label=int(F.CANON_MOVE[alternative]) if s else alternative
            other=_pay_forced(x,aux,encoded,mode,slot,label)
            if other is None or np.flatnonzero(other!=original).tolist()!=[slot]:
                continue
            conflicts=_return_conflicts(x,other)
            if conflicts-before_conflicts or any(slot in pair for pair in conflicts):
                continue
            rival=_FEAT.to_actions(other)
            if rival.get(uid,'NONE')!=moves[alternative] or {k for k in set(base)|set(rival) if base.get(k)!=rival.get(k)}!={uid}:
                raise ValueError('selected planter did not change in isolation')
            _PAY_USED.add(window)
            _PAY_TRACE.append(dict(t=int(t),seat=int(s),window=window,uid=uid,cell=int(p),kind=kind,quoted_cost=float(cost),
                                   own_cash=float(own[0]),other_planter_reserve=float(others),hire_reserve=float(own[0]-available),
                                   pre=old,changed=old!='NONE',base=dict(base)))
            return other if old!='NONE' else original
    return original


def agent(observation,configuration=None):
    global _PAY_VIEW
    if observation['step']==0:
        _PAY_USED.clear();_PAY_TRACE.clear()
    _PAY_VIEW=(observation,{**_PAY_CFG,**(configuration or {})})
    return _PAY_BASE_AGENT(observation,configuration)
