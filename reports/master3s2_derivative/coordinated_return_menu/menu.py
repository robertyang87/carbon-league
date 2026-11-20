def _coordinated_return(players, me, logits, base_order, fixed):
    """Offer one joint move that gives a loaded worker an occupied return lane."""
    workers, _, bases, _ = parse(players)
    mine = {wid: w for wid, w in workers.items() if w[3] == me}
    home = [pos for _, pos in bases[me]]
    if not home or len(mine) < 2:
        return None
    locked = fixed or {}
    destinations = {}
    indices = {}
    for wid, worker in mine.items():
        index = WIDX.get(base_order.get(wid, 'NONE'), 0)
        indices[wid] = index
        destinations.setdefault(move(worker[0], index), []).append(wid)
    recruits = {pos for bid, pos in bases[me] if bid in base_order}
    best = None
    for wid, (position, cargo, role, _) in mine.items():
        if role != 'C' or cargo < 100 or wid in locked:
            continue
        current = min(dist(position, b) for b in home)
        prior = min(dist(move(position, indices[wid]), b) for b in home)
        if current == 0 or prior < current:
            continue
        for action in range(1, 5):
            target = move(position, action)
            occupants = destinations.get(target, ())
            if target in recruits or len(occupants) != 1 or min(dist(target, b) for b in home) >= current:
                continue
            blocker = occupants[0]
            if blocker == wid or blocker in locked:
                continue
            blocker_pos = mine[blocker][0]
            vacated = move(position, indices[wid])
            for alternative in np.argsort(-logits[blocker_pos], kind='stable'):
                destination = move(blocker_pos, alternative)
                occupied = destinations.get(destination, ())
                if destination == target or destination in recruits or (occupied and not (destination == vacated and occupied == [wid])):
                    continue
                priority = (
                    cargo,
                    float(logits[position, action] - logits[position, indices[wid]]
                          + logits[blocker_pos, alternative] - logits[blocker_pos, indices[blocker]]),
                )
                if best is None or priority > best[0]:
                    best = (priority, wid, action, blocker, int(alternative))
                break
    if best is None:
        return None
    _, wid, action, blocker, alternative = best
    order = dict(base_order)
    order[wid] = WACT[action]
    if alternative:
        order[blocker] = WACT[alternative]
    else:
        order.pop(blocker, None)
    return order
