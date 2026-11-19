"""Append this module after the unchanged Sapling carrier and the public simulation helpers."""
_FRONT_ENABLED = True
_FRONT_ROOT = None
_FRONT_PAST = None
_FRONT_AUDIT = None
_front_old_agent = agent
_front_old_mix = _mixed_actions


def _mixed_actions(x, aux):
    global _FRONT_ROOT
    labels = _front_old_mix(x, aux)
    _FRONT_ROOT = x, aux, labels
    return labels


def _front_decode(x, aux, encoded, slot, action):
    forced = (slot, action)
    base = _front_forced_decode(x, aux, encoded, False, forced)
    if base is None or not np.any(_RETURN_THETA):
        return base
    adjusted = _front_forced_decode(x, aux, encoded, True, forced)
    if adjusted is None:
        return base
    protected = (np.arange(len(base)) >= 10) | (x['tok_kind'][:len(base)] == 1)
    if np.any(base[protected] != adjusted[protected]):
        return base
    return base if _return_conflicts(x, adjusted) - _return_conflicts(x, base) else adjusted


def _front_project(obs, past, cfg, options):
    q = _QUEUE_POLICY
    seat, t = obs['player'], obs['step']
    state = q._public_sim(obs, cfg)
    reset = state.get_state()
    enemy = [q.orders(state.obs(1-seat), past, cfg, k) for k in range(3)]
    projections = []
    for cmd in options:
        scores = []
        for k in range(3):
            state.set_state(reset)
            joint = [None, None]
            joint[seat], joint[1-seat] = cmd, enemy[k]
            history = {t: {'step': t, 'players': obs['players']}}
            while True:
                state.step(*joint)
                if q._extinct(state):
                    scores.append(None)
                    break
                if state.step_count >= cfg['episodeSteps']-1:
                    scores.append(state.cash[seat]-state.cash[1-seat])
                    break
                views = [state.obs(p) for p in (0, 1)]
                joint = [q.orders(view, history, cfg, k) for view in views]
                history = {state.step_count: {'step': state.step_count, 'players': views[seat]['players']}}
        projections.append(scores)
    return projections


def _front_metadata(obs, command, cfg):
    q = _QUEUE_POLICY
    n, seat = cfg['size'], obs['player']
    own, opp = obs['players'][seat], obs['players'][1-seat]
    homes, banks = set(own[1].values()), set(own[1].values()) | set(opp[1].values())
    trees = {pos: (side, age) for side, p in enumerate(obs['players']) for pos, age in p[3].values()}
    living = sum(age < cfg['treeLifespan'] for _, age in trees.values())
    price = cfg['plantCost'] if len(own[3]) <= cfg['playerTreeProtectiveNumber'] else (
        cfg['plantCost'] + cfg['plantCostInflationRatio'] * cfg['plantCostInflationBase']**living)
    routes, inbox = {}, {}
    for uid, (pos, load, kind) in own[2].items():
        if kind != 'COLLECTOR' or load <= 0:
            continue
        cells = q.moves(pos, n)
        expected = cells[command.get(uid, 'NONE')]
        if expected in homes:
            inbox[uid] = [expected, load]
        d = q.home_dist(pos, homes, n)
        if 0 < d <= 2:
            for cell in cells.values():
                if q.home_dist(cell, homes, n) < d:
                    routes.setdefault(cell, []).append([uid, load])
    risks = []
    for uid in _FEAT.own_uid[:10]:
        if uid is None:
            continue
        pos, _, kind = own[2][uid]
        if kind != 'PLANTER':
            continue
        cmd = command.get(uid, 'NONE')
        dest = q.moves(pos, n)[cmd]
        owner, age = trees.get(pos, (None, None))
        fee = cfg['seizeCost'] if owner == 1-seat else price
        pay = cmd == 'NONE' and pos not in banks and owner != seat and own[0] >= fee
        if pay or dest in routes:
            risks.append({'uid': uid, 'pos': pos, 'pay_intent': pay, 'fee_before_joint': fee if pay else 0,
                          'tree_owner': owner, 'tree_age': age, 'cargo_corridor': routes.get(dest, []),
                          'enemy_cargo_within_two': sum(w[1] for w in opp[2].values()
                                                       if w[2]=='COLLECTOR' and q.dist(w[0],pos,n)<=2)})
    return risks, inbox


def _front_choose(obs, base, cfg, past):
    me = obs['player']
    x, aux, old = _FRONT_ROOT
    risk, inbox = _front_metadata(obs, base, cfg)
    audit = {'roots':risk,'inbound':inbox,'arms':[],'chosen':None}
    if not risk:
        return base, audit
    slot = _FEAT.own_uid.index(risk[0]['uid'])
    uid = risk[0]['uid']
    enc = []
    for policy in _NETS:
        tok, ctx = policy.encode(x, _HYBRID['style'])
        enc.append((policy, tok, ctx, policy.lin(ctx, 'h0')))
    choices = [base]
    q = _QUEUE_POLICY
    protected = (np.arange(len(old)) >= 10) | (x['tok_kind'][:len(old)] == 1)
    protected[slot] = False
    original = _return_conflicts(x, old)
    n = cfg['size']
    # Head direction is canonical. Command IDs and locations below are physical for this seat.
    for action in (1, 2, 3, 4):
        if old[slot] == action:
            continue
        alt = _front_decode(x, aux, enc, slot, action)
        if alt is None or np.any(alt[protected] != old[protected]) or _return_conflicts(x,alt)-original:
            continue
        cmd = _FEAT.to_actions(alt)
        if any(q.moves(obs['players'][me][2][u][0],n)[cmd.get(u,'NONE')] != dest for u,(dest,_) in inbox.items()):
            continue
        choices.append(cmd)
    if len(choices)==1:
        return base,audit
    projections = _front_project(obs, past, cfg, choices)
    best = None
    for i, (cmd, scores) in enumerate(zip(choices, projections)):
        gain = None if None in scores or None in projections[0] else [v-b for v,b in zip(scores,projections[0])]
        audit['arms'].append({'command':cmd,'projected_terminal_margins':scores,'gain':gain})
        if i and gain is not None and min(gain)>.01 and sum(gain)/3>=5:
            key=(min(gain),sum(gain)/3,-i)
            if best is None or key>best[0]:
                best=(key,i)
    if best is not None:
        audit['chosen'] = best[1]
        return choices[best[1]], audit
    return base, audit


def agent(observation, configuration=None):
    global _FRONT_PAST,_FRONT_AUDIT
    own = _front_old_agent(observation, configuration)
    t,seat = observation['step'],observation['player']
    cfg = _QUEUE_SIM.DEFAULT_CONFIG if configuration is None else {
        k:configuration.get(k,v) for k,v in _QUEUE_SIM.DEFAULT_CONFIG.items()}
    result,trace = own,None
    if _FRONT_ENABLED and 1 <= cfg['episodeSteps']-1-t <= 20:
        prev = _FRONT_PAST
        past = {t-1:prev} if prev and prev['step']==t-1 and prev['player']==seat else {}
        result,trace = _front_choose(observation, own, cfg, past)
    _FRONT_AUDIT=(t,seat,own,trace)
    _FRONT_PAST={'step':t,'player':seat,'players':[(None,None,{u:tuple(w) for u,w in p[2].items()})
                                                    for p in observation['players']]}
    return result
