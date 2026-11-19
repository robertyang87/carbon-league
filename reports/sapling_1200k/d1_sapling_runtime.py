# Appended to frozen D1 by d1_sapling_build.py.
_WEIGHTS = None
_MODELS = None
_AG = None
_WITNESSES = {}
_LAST = {}


def _prepare_root(x):
    states = []
    for model in _MODELS:
        tokens, context = model.encode(x, 2)
        states.append((model, tokens, context, model.lin(context, "h0")))
    return states


def _decode_joint(x, feat, encoded, forced=None):
    F = _P["cnet2_feat"]
    prefix = F.Prefix(x, feat.aux)
    states = [s[3] for s in encoded]
    previous = 8
    labels = np.full(F.OWN, -1, np.int64)
    logits = np.full((F.OWN, 5), -np.inf, np.float32)
    evidence = []
    for slot in range(F.OWN):
        if not x["tok_mask"][slot]:
            continue
        unit, move, legal = prefix.feats(slot)
        scores = []
        for idx, (model, tokens, context, _) in enumerate(encoded):
            features = np.concatenate([tokens[slot], context, unit, move.reshape(-1)])
            hidden = model.gru(model.lin(features, "din") + model.w["aemb.weight"][previous], states[idx])
            states[idx] = hidden
            if slot < 10:
                base = model.lin(hidden, "whead")
            else:
                base = np.concatenate([model.lin(hidden, "bhead"), np.zeros(2, np.float32)])
            scores.append(base + model.lin(move, "V") @ model.lin(hidden, "U"))
        combined = 0.75 * scores[0] + 0.25 * scores[1]
        logits[slot] = np.where(legal, combined, -np.inf)
        forced_here = forced is not None and forced[slot] >= 0
        action = int(forced[slot] if forced_here else np.argmax(logits[slot]))
        if not legal[action]:
            return None, None
        expert = int(np.argmax(np.where(legal, scores[0], -np.inf)))
        if not forced_here and action != expert:
            evidence.append((feat.own_uid[slot], action, expert))
        labels[slot] = action
        prefix.advance(slot, action)
        previous = action + (0 if slot < 10 else 5)
    key = tuple(sorted(feat.to_actions(labels).items()))
    _WITNESSES.setdefault(key, evidence)
    if forced is None:
        _LAST["base"] = dict(key)
    else:
        _LAST["legal_branches"] += 1
    return labels, logits


def agent(observation, configuration=None):
    global _AG, _WEIGHTS, _MODELS
    if _WEIGHTS is None:
        def unpack(blob, qmeta, meta):
            weights = _dequantize(zlib.decompress(base64.b64decode(blob)), qmeta)
            weights["meta"] = np.array(meta)
            return weights
        _WEIGHTS = [unpack(_B1, _Q1, _M1), unpack(_B2, _Q2, _M2)]
        _MODELS = [_P["cnet2_np"].Policy(weights) for weights in _WEIGHTS]
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(dict(CONF, MIXED_ENCODE=_prepare_root, MIXED_DECODE=_decode_joint),
                                   weights=_WEIGHTS[0])
    _WITNESSES.clear()
    _LAST.clear()
    _LAST["legal_branches"] = 0
    # Strip only out-of-band diagnostic inputs; legitimate observations are identical to the parent.
    keys = ("step", "player", "players", "trees", "carbon", "remainingOverageTime")
    public = {k:observation[k] for k in keys if k in observation}
    actions = _AG(public, configuration)
    key = tuple(sorted(actions.items()))
    if key not in _WITNESSES:
        raise RuntimeError("candidate executed an action without a validated decoder prefix")
    _LAST["selected_unforced_differs_from_expert"] = _WITNESSES[key]
    _LAST["changed"] = actions != _LAST["base"]
    agent.last_stats = _AG.stats
    return actions
