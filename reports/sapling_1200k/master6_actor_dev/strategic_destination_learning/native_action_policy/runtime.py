"""Deployment addition to the hash-pinned incumbent. All head inputs precede both seats' actions."""

_NATIVE_PARENT_MIXED = _mixed_actions
_NATIVE_RNG = np.random.default_rng()
_LAST_POLICY = None


def _native_features(x, unit, moves, logits, legal):
    values = np.asarray(logits, dtype=np.float32)
    legal_values = values[legal]
    centered = np.clip(values - legal_values.mean(), -16, 16)
    gap = np.clip(values - legal_values.max(), -16, 0)
    repeated = np.broadcast_to(np.r_[unit, x["scal"]], (5, len(unit) + len(x["scal"])))
    return np.concatenate((moves, repeated, np.eye(5, dtype=np.float32),
                           centered[:, None], gap[:, None]), axis=1).astype(np.float32)


def _native_guard(x, baseline, adjusted):
    own = len(baseline)
    protected = (np.arange(own) >= 10) | (x["tok_kind"][:own] == 1)
    if np.any(adjusted[protected] != baseline[protected]):
        return baseline, False
    if np.any(adjusted != baseline) and _return_conflicts(x, adjusted) - _return_conflicts(x, baseline):
        return baseline, False
    return adjusted, True


def _native_softmax(values):
    exponent = np.exp(np.asarray(values, np.float64) - np.max(values))
    return exponent / float(exponent.sum())


def _native_reachable(option, pivot, traces):
    # Cash and recruitment slots are prefix-dependent on each guard route.
    return all(bool(route[pivot][2][option]) for route in traces)


def _native_menu(x, aux, step):
    encoded = []
    for net in _NETS:
        tokens, context = net.encode(x, _HYBRID["style"])
        encoded.append((net, tokens, context, net.lin(context, "h0")))
    routes, traces = [], []
    for residual in (False, True):
        trace = {}
        routes.append(_native_decode(x, aux, encoded, residual, trace=trace))
        traces.append(trace)
    base, rerouted = _native_guard(x, *routes)
    trace = traces[int(rerouted)]
    preferred = (step // 6 - 1) % 3
    slots = [s for s in trace if trace[s][2].sum() >= 2]
    if not slots:
        return base, None
    choices = [s for s in slots if x["tok_kind"][s] == preferred] or slots

    def entropy(s):
        prob = _native_softmax(trace[s][3][trace[s][2]])
        return float(-np.dot(prob, np.log(prob)))

    pivot = max(choices, key=entropy)
    unit, moves, legal, logits = trace[pivot]
    actions = np.flatnonzero(legal)
    teams = []
    base_conflicts = _return_conflicts(x, base)
    for action in actions:
        if action == base[pivot] or not _native_reachable(action, pivot, traces):
            teams.append(base)
            continue
        proposals = [_native_decode(x, aux, encoded, residual, forced=(pivot, action))
                     for residual in (False, True)]
        full, _ = _native_guard(x, *proposals)
        if _return_conflicts(x, full) - base_conflicts:
            full = base
        teams.append(full)
    features = _native_features(x, unit, moves, logits, legal)
    return base, (pivot, actions, np.stack(teams), features, logits)


def _native_distribution(base, menu):
    if menu is None:
        return [base], np.array([1.0])
    _, actions, teams, features, logits = menu
    hidden = np.tanh(features @ _NATIVE_W0 + _NATIVE_B0)
    residual = (hidden @ _NATIVE_W1).ravel()
    move_p = _native_softmax(logits[actions] + residual[actions])
    masses, distinct = {}, {}
    for team, weight in zip([base, *teams], [0.5, *(0.5 * move_p)]):
        key = tuple(sorted(_FEAT.to_actions(team).items()))
        distinct[key] = team
        masses[key] = masses.get(key, 0.) + float(weight)
    return list(distinct.values()), np.asarray(list(masses.values()), np.float64)


def _mixed_actions(x, aux):
    global _LAST_POLICY
    step = int(299 - x["left"])
    menu = None
    if 0 < step <= 294 and step % 6 == 0:
        base, menu = _native_menu(x, aux, step)
    else:
        base = _NATIVE_PARENT_MIXED(x, aux)
    teams, probabilities = _native_distribution(base, menu)
    draw = int(_NATIVE_RNG.choice(len(teams), p=probabilities)) if len(teams) > 1 else 0
    commands = [_FEAT.to_actions(team) for team in teams]
    base_cmd = _FEAT.to_actions(base)
    pivot = None if menu is None else int(menu[0])
    role = None if pivot is None else ("collector", "planter", "base")[int(x["tok_kind"][pivot])]
    _LAST_POLICY = {
        "step": step, "seat": int(_FEAT.seat), "pivot_slot": pivot, "pivot_role": role,
        "pivot_uid": None if pivot is None else _FEAT.own_uid[pivot],
        "orders": [{"action": cmd, "probability": float(p)} for cmd,p in zip(commands, probabilities)],
        "parent": base_cmd, "parent_probability": float(probabilities[commands.index(base_cmd)]),
        "issued": commands[draw], "logp": float(np.log(probabilities[draw]))
    }
    return teams[draw]
