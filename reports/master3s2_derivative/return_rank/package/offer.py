_RETURN_ENABLED = True
_RETURN_TRACE = None


def _return_offer(players, me, step, act0, home, carbon):
    """Construct at most one new feasible joint action from the root public rosters."""
    if _RETURN_TRACE is not None:
        _RETURN_TRACE.clear()
        _RETURN_TRACE.update(step=step, decision="no_eligible_worker")
    if step > LAST_TURN - 5:
        return None
    mine = players[me]
    enemy = players[1 - me]
    bases = tuple(mine[1].values())
    eligible = []
    for uid, (position, cargo, role) in mine[2].items():
        if role == "COLLECTOR" and cargo >= 100 and uid not in home:
            distance = min(dist(position, base) for base in bases)
            if 1 <= distance <= 5:
                eligible.append((-cargo, uid, position, distance))
    if not eligible:
        return None
    neg_cargo, uid, position, distance = min(eligible)
    baseline = WIDX.get(act0.get(uid, "NONE"), 0)
    original_end = move(position, baseline)
    max_distance = min(distance, min(dist(original_end, base) for base in bases))
    reserved = {move(w[0], WIDX.get(act0.get(wid, "NONE"), 0))
                for wid, w in mine[2].items() if wid != uid}
    reserved.update(cell for name, cell in mine[1].items()
                    if act0.get(name) in ("RECCOLLECTOR", "RECPLANTER"))
    reserved.update(enemy[1].values())
    threats = [w[0] for w in enemy[2].values() if w[2] == "COLLECTOR" and w[1] <= -neg_cargo]
    legal = []
    for action in range(5):
        target = move(position, action)
        if action == baseline or target in reserved or any(dist(target, pos) <= 1 for pos in threats):
            continue
        if min(dist(target, base) for base in bases) <= max_distance:
            legal.append(action)
    if _RETURN_TRACE is not None:
        _RETURN_TRACE.update(considered=uid, native_action=WACT[baseline],
                             allowed=[WACT[a] for a in legal], decision="all_alternatives_blocked")
    if not legal:
        return None
    features = _RETURN_FEATURES(players, me, uid, step, carbon=carbon)
    scores = np.asarray(_RETURN_SCORE(features, _RETURN_WEIGHTS), dtype=np.float64)
    if scores.shape != (5,) or not np.isfinite(scores).all():
        return None
    action = max(legal, key=lambda k: (float(scores[k]), -k))
    gain = float(scores[action] - scores[baseline])
    if _RETURN_TRACE is not None:
        _RETURN_TRACE.update(decision="insufficient_gain", scores=scores.tolist(), gain=gain)
    if gain < 0.05:
        return None
    order = act0.copy()
    if action:
        order[uid] = WACT[action]
    else:
        order.pop(uid, None)
    if _RETURN_TRACE is not None:
        _RETURN_TRACE.update(pivot=uid, offer_action=WACT[action], cargo=-neg_cargo, decision="proposed")
    return order


def _return_menu(cands, players, me, step, home, most, t0, configuration, carbon):
    # The original search and its A0/zero-depth fallback remain usable when
    # an offer is unavailable. Candidate generation consumes none of its RNG.
    if (_time.perf_counter() - t0 >= 0.50 or len(cands) < 2 or most < 2 or
            configuration.get("size", 15) != 15 or configuration.get("moveCost", 0) != 0):
        return
    try:
        order = _return_offer(players, me, step, cands[0], home, carbon)
    except (ValueError, TypeError, KeyError, ArithmeticError, IndexError):
        if _RETURN_TRACE is not None:
            _RETURN_TRACE['invalid_input'] = True
        return
    if order is None or _time.perf_counter() - t0 >= 0.50:
        return
    if order in cands:
        if _RETURN_TRACE is not None:
            _RETURN_TRACE.update(existing=True, offer_index=cands.index(order))
        return
    cands[-1] = order
    if _RETURN_TRACE is not None:
        _RETURN_TRACE.update(existing=False, offer_index=len(cands) - 1)
