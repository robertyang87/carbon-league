"""One root proposal from the frozen rent model, outside the live policy path in isolation."""


def _tree_sites(players, step, carbon, known, me, act0, config):
    if not 20 <= step <= 92:
        return None
    own = players[me]
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({k: config[k] for k in DEFAULT_CONFIG if k in config})
    bases = {p for pl in players for p in pl[1].values()}
    occupied_trees = {v[0] for pl in players for v in pl[3].values()}
    if len(own[3]) <= cfg['playerTreeProtectiveNumber']:
        price = cfg['plantCost']
    else:
        price = cfg['plantCost'] + cfg['plantCostInflationRatio'] * (
            cfg['plantCostInflationBase'] ** len(occupied_trees))
    available = own[0] - sum(k in act0 for k in own[1]) * 30
    if available < price:
        return None
    own_destinations = {
        wid: move(unit[0], WIDX.get(act0.get(wid, 'NONE'), 0))
        for wid, unit in own[2].items()
    }
    planned_bases = {p for k, p in own[1].items() if k in act0}
    proposals = []
    sites = [list(p) for p in players]
    sites[me][3] = own[3].copy()
    marker = '_rent_proposal_only'
    weights = _RENT_FIT

    def quote(cell):
        if cell in bases or cell in occupied_trees:
            return None
        sites[me][3][marker] = [cell, 0]
        values = next(x for k, _, x in _existing_tree_features(sites, step, carbon, known) if k == marker)
        x = np.asarray(values)
        return max(0., float(weights['intercept'] + ((x-weights['mean'])/weights['scale']) @ weights['coef']))

    quotes = {}
    for uid, (cell, _, kind) in own[2].items():
        if kind != 'PLANTER':
            continue
        current = own_destinations[uid]
        if current not in quotes:
            quotes[current] = quote(current)
        reference = quotes[current]
        if reference is None:
            # Do not compete with a planned defense, seizure or move onto a base.
            continue
        blocked = planned_bases | {p for k, p in own_destinations.items() if k != uid}
        for direction in range(5):
            dest = move(cell, direction)
            if dest == current or dest in blocked:
                continue
            if dest not in quotes:
                quotes[dest] = quote(dest)
            value = quotes[dest]
            if value is None:
                continue
            difference = value-reference - (2.0 if direction else 0.0)
            if difference > 4.0:
                proposals.append((difference, uid, direction))
    if not proposals:
        return None
    _, uid, direction = max(proposals)
    candidate = act0.copy()
    if direction:
        candidate[uid] = WACT[direction]
    else:
        candidate.pop(uid, None)
    return candidate
