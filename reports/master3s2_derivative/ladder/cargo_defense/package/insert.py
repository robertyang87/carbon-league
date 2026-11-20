# Runtime switch and lightweight observational hook for local matched public requests.
_CARGO_DEFENSE = True
_CARGO_TRACE = None


def _cargo_postsearch(obs, previous, parent, t0):
    clock = _time.perf_counter
    start = clock()
    trace = _CARGO_TRACE
    if trace is not None:
        trace.update(before=dict(parent), status="idle", revisions=[], checked=[])
    if t0 is None or start-t0 >= _S_ABORT-.012:
        if trace is not None:
            trace["status"] = "insufficient_time"
        return parent
    roster, side = obs["players"], obs["player"]
    cash, bases, workers, _ = roster[side]
    heavy = [(uid,w) for uid,w in workers.items() if w[2] == "COLLECTOR" and w[1] >= 200]
    if not heavy:
        return parent
    rival = roster[1-side]
    # Isolated rival base recruit probability is below the registered risk threshold.
    # A closer/equal rival worker is a necessary condition for a score >= .25.
    opposing = [(w[0],w[1]) for w in rival[2].values() if w[2] == "COLLECTOR"]
    heavy = [(uid,w) for uid,w in heavy if any(load <= w[1] and FEAT_DIST[pos,w[0]] <= 2 for pos,load in opposing)]
    if not heavy:
        return parent
    state = _cargo_prepare(roster,side,obs["step"],previous)
    candidate = dict(parent)
    taken = np.zeros(225, dtype=np.int16)
    count = len(workers)
    for uid,cell in bases.items():
        if candidate.get(uid) in ("RECCOLLECTOR","RECPLANTER") and count < 10 and cash >= 30:
            taken[cell] += 1
            count += 1
            cash -= 30
    for uid,w in workers.items():
        taken[_OPP_STEPS[w[0],WIDX[candidate.get(uid) or "NONE"]]] += 1
    changed = []
    for uid,w in sorted(heavy,key=lambda item:(-item[1][1],item[0])):
        if clock()-t0 >= _S_ABORT-.006:
            if trace is not None:
                trace["status"] = "discarded_for_time"
            return parent
        a = WIDX[candidate.get(uid) or "NONE"]
        f = _cargo_forecast(state,uid,_CARGO_W,_CARGO_RECRUIT)
        risk = f["risk"]
        if trace is not None:
            trace["checked"].append(dict(uid=uid, old=a, risk=risk, home=f["home"], friend=f["friend"]))
        if risk[a] < .25:
            continue
        dest = _OPP_STEPS[w[0]]
        alternatives = [k for k in range(5) if k != a and not f["friend"][k]
                        and taken[dest[k]] == 0 and risk[a]-risk[k] >= .10-1e-12]
        if not alternatives:
            continue
        k = min(alternatives,key=lambda i:(risk[i],f["home"][i],i))
        taken[dest[a]] -= 1
        if taken[dest[a]] < 0:
            raise RuntimeError("Invalid own order ledger")
        taken[dest[k]] += 1
        if k:
            candidate[uid] = WACT[k]
        else:
            candidate.pop(uid,None)
        changed.append((uid,a,k))
    if clock()-t0 >= _S_ABORT-.002:
        if trace is not None:
            trace["status"] = "discarded_for_time"
        return parent
    if trace is not None:
        trace.update(status="changed" if changed else "unchanged",revisions=changed)
    return candidate if changed else parent
