# Appended after the trained intent functions and action-selection code.
_GEOMETRY = sys.modules["cnet2_feat"]
_INTENT = types.SimpleNamespace(DEST=DEST, DIST=DIST, MOVE_ID={name:i for i,name in enumerate(MOVES)}, predict=predict)
_NET = None
_SEATS = {}


def agent(observation, configuration):
    global _NET
    if _NET is None:
        w = _dequantize(zlib.decompress(base64.b64decode(_W64)), _QMETA)
        w["meta"] = np.array(_WMETA)
        _NET = sys.modules["cnet2_np"].Policy(w)
    seat, step = observation["player"], observation["step"]
    if seat not in _SEATS or step <= _SEATS[seat]["step"]:
        _SEATS[seat] = dict(feat=_GEOMETRY.Feat(seat), previous=None, step=-1,
                            stats=dict(calls=0, preserved_roots=0, changed_roots=0))
    state = _SEATS[seat]
    x = state["feat"].observe(observation)
    base, action, changes = choose_actions(_NET, state["feat"], x, observation, state["previous"],
                                           configuration, _INTENT_WEIGHTS, _INTENT)
    state["previous"] = {"player": seat, "step": step, "players": [
        [None, None, {wid:list(unit) for wid,unit in player[2].items()}] for player in observation["players"]]}
    state["step"] = step
    stats = state["stats"]
    stats["calls"] += 1
    stats["changed_roots"] += int(base != action)
    stats["preserved_roots"] += int(base == action)
    agent.last_stats = stats
    agent.last_baseline = base
    agent.last_interventions = changes
    return action
