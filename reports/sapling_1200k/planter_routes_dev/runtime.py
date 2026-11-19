from collections import Counter

_ROUTE_COUNTS = Counter()
_ROUTE_LAST = None
_route_actor = agent


def agent(observation, configuration=None):
    global _route_state, _route_cooldown, _ROUTE_LAST
    step = observation['step']
    seat = observation['player']
    if step == 0 or getattr(agent,'seat',None) != seat:
        _route_state = PlanterRouteState()
        _route_cooldown = {}
        _ROUTE_COUNTS.clear()
    agent.seat = seat
    _ROUTE_LAST = None
    _route_state.observe(observation,seat,observation['carbon'],configuration or {},step)
    return _route_actor(observation,configuration)


def _route_decide(F,x,prefix,slot,normal,logits,legal,changes):
    st = _route_state
    if slot >= 10 or x['tok_kind'][slot] != 1 or normal == 0 or changes:
        return normal
    _ROUTE_COUNTS['parent_planter_moves_in_window'] += 1
    uid = _FEAT.own_uid[slot]
    if st.step - _route_cooldown.get(uid,-1000) <= 2:
        return normal
    pos = int(x['tok_pos'][slot]); dst = F.DEST[pos]
    physical = F.CANON[dst] if st.seat else dst
    if int(physical[0]) in st.own_trees and any(_ROUTE_DIST[int(physical[0]),e] <= 2 for e in st.opp_p):
        _ROUTE_COUNTS['protect_current_tree'] += 1
        return normal
    a = prefix.a
    viable = legal & (prefix.reserved[dst] == 0) & (prefix.undecided[dst] == 0)
    viable &= ~a['en_base'][dst] & ~np.isfinite(a['en_min'][dst]) & (a['en_pl'][dst] == 0)
    viable &= logits[normal]-logits <= .85
    viable[0] = False
    if not viable[normal] or viable.sum() < 2:
        return normal
    _ROUTE_COUNTS['near_policy_safe_pairs'] += 1
    reserve_cash = prefix.cash - prefix.plants*max(st.new_price,st.cfg.get('seizeCost',20))
    values = {}
    for k in np.flatnonzero(viable):
        feat = st.features(int(physical[k]))
        if feat is None or reserve_cash < feat[1] + 30.:continue
        values[int(k)] = (*route_value(_ROUTE_MODEL,feat[0]),feat[1])
    if normal not in values or len(values)<2:return normal
    _ROUTE_COUNTS['affordable_comparable_landing'] += 1
    alt = max(values,key=lambda k: values[k][0])
    gain = values[alt][0]-values[normal][0]
    if alt == normal or gain < 4.:return normal
    detail = dict(worker=uid,root=st.step,base_command=int(normal),alt_command=int(alt),source_cell=int(physical[0]),
                  proposed_cell=int(physical[alt]),parent_cell=int(physical[normal]),cash_available=float(reserve_cash),
                  expected_parent_net=values[normal][0],expected_chosen_net=values[alt][0],prob_pay=values[alt][1],
                  conditional_tenure_net=values[alt][2],quoted_price=values[alt][3],gain=float(gain),
                  logit_gap=float(logits[normal]-logits[alt]))
    changes.append((slot,normal,alt,detail))
    return int(alt)


def _mixed_actions(x,aux):
    global _ROUTE_LAST
    encoded = _route_encode(x)
    if not 45 <= _route_state.step <= 195:
        return _route_decode(x,aux,encoded)
    before = {role:d.copy() for role,d in _COUNTS.items()}
    changes = []
    proposed = _route_decode(x,aux,encoded,changes)
    if not changes:return proposed
    after = {role:d.copy() for role,d in _COUNTS.items()}
    _COUNTS.update(before)
    base = _route_decode(x,aux,encoded)
    slot,normal,alternative,detail = changes[0]
    if np.count_nonzero(base!=proposed)!=1 or not _return_joint_ok(sys.modules['cnet2_feat'],x,aux,base,proposed,[(slot,normal,alternative)]):
        _ROUTE_COUNTS['joint_veto']+=1
        return base
    _COUNTS.update(after)
    _ROUTE_LAST = detail
    _route_cooldown[detail['worker']] = _route_state.step
    _ROUTE_COUNTS['changed_roots'] += 1
    return proposed
