"""Appended to the source-frozen v1 actor and exact immutable prefix validator."""
_VALUE_PARENT = agent
_VALUE_MIX = _mixed_actions
_VALUE_CONTEXT = None
_VALUE_ENABLED = True
_VALUE_SEAT = None
_VALUE_LAST_STEP = None
_VALUE_USED = False
_VALUE_TRACE = None


def _mixed_actions(x, aux):
    global _VALUE_CONTEXT
    labels = _VALUE_MIX(x, aux)
    _VALUE_CONTEXT = x, aux
    return labels


def _value_choices(view, cfg, parent, x, aux):
    # First legal alternative of each kind in the exact deterministic proposal ordering.
    result = []
    seen = set()
    for p in _VALUE_ACTIONS.proposals(view,cfg,parent):
        kind = p["kind"]
        if kind in seen:
            continue
        orders, reason = _local_edit(view,cfg,parent,{p["uid"]:p["action"]},x,aux,_FEAT)
        if reason=="accepted":
            seen.add(kind)
            result.append((p,orders))
    return result


def agent(observation, configuration=None):
    global _VALUE_CONTEXT, _VALUE_SEAT, _VALUE_LAST_STEP, _VALUE_USED, _VALUE_TRACE
    seat = observation["player"]
    tick = observation["step"]
    if tick==0 or _VALUE_SEAT!=seat or (_VALUE_LAST_STEP is not None and tick<_VALUE_LAST_STEP):
        _VALUE_USED = False
    _VALUE_SEAT, _VALUE_LAST_STEP = seat, tick
    _VALUE_CONTEXT = None
    # The frozen featurizer also supports an offline-only reveal hook. Project the
    # official public observation before the first network or local scorer sees it.
    view = {k:observation[k] for k in ("step","player","players","trees","carbon")}
    parent = _VALUE_PARENT(view, configuration)
    result = parent
    choices = []
    chosen = None
    if not _VALUE_ENABLED or not _VALUE_PARAMS["enabled"]:
        reason = "disabled"
    elif not any(lo<=tick<=hi for lo,hi in _VALUE_ACTIONS.BANDS):
        reason = "outside_window"
    elif _VALUE_USED:
        reason = "spent"
    else:
        cfg = {k:(configuration or {}).get(k,v) for k,v in _VALUE_DEFAULT_CFG.items()}
        options = _value_choices(view,cfg,parent,*_VALUE_CONTEXT)
        choices = [p for p,_ in options]
        chosen = _VALUE_POLICY.choose(choices,_VALUE_PARAMS)
        if chosen is None:
            reason = "no_eligible" if choices else "no_opportunity"
        else:
            for p, orders in options:
                if p is chosen:
                    result = orders
                    _VALUE_USED = True
                    break
            else:
                raise ValueError("value learner selected outside the validated menu")
            reason = "accepted"
    _VALUE_TRACE = {"seat":seat,"step":tick,"parent":parent,"reason":reason,"used":_VALUE_USED,
                    "offered":[(p["uid"],p["action"],p["kind"]) for p in choices],
                    "chosen":(chosen["uid"],chosen["action"],chosen["kind"]) if chosen is not None else None}
    _VALUE_CONTEXT = None
    return result
