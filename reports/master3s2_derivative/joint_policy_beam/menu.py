def _joint_policy_beam(players, seat, logits, baseline, fixed, existing, limit):
    """Enumerate diverse legal team changes under the frozen policy prior."""
    workers, _, bases, _ = parse(players)
    mine = {uid: w for uid, w in workers.items() if w[3] == seat}
    locked = fixed or {}
    occupied = {pos for bid, pos in bases[seat] if bid in baseline}
    for uid, action in locked.items():
        if uid in mine:
            occupied.add(move(mine[uid][0], WIDX.get(action or 'NONE', 0)))
    queue = []
    for uid, worker in mine.items():
        if uid in locked:
            continue
        pos = worker[0]
        delta = logits[pos] - logits[pos].max()
        ranked = np.argsort(-delta, kind='stable')
        queue.append((float(delta[ranked[1]]), uid, pos, delta, ranked))
    queue.sort(key=lambda row: row[0])
    if not queue:
        return []
    width = 8
    beam = [(0.0, {}, occupied)]
    for _, uid, pos, delta, ranked in queue:
        expanded = []
        for score, order, taken in beam:
            choices = [int(k) for k in ranked if move(pos, int(k)) not in taken][:3]
            for k in choices:
                dest = move(pos, k)
                branch = order.copy()
                if k:
                    branch[uid] = WACT[k]
                expanded.append((score+float(delta[k]), branch, taken|{dest}))
        if not expanded:
            return []
        beam = sorted(expanded,key=lambda row: -row[0])[:width]
    static_order = {uid: action for uid, action in baseline.items() if uid not in mine or uid in locked}
    result = []
    for _, moves, _ in beam:
        actions = static_order | moves
        key = tuple(sorted(actions.items()))
        if key not in existing:
            result.append(actions)
        if len(result) == limit:
            break
    return result
