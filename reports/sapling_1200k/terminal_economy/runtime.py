# This observation contract contains no map seed, opponent label or hidden carbon.
_ECON_HISTORY = None
_ECON_COVER = {}


def _update_economy(observation):
    global _ECON_HISTORY, _ECON_COVER
    me = int(observation["player"])
    players = observation["players"]
    trees = observation.get("trees", {})
    incomes = [sum(float(trees.get(tid, (None, 0.))[1]) for tid in player[3]) for player in players]
    rival_advantage = incomes[1-me] - incomes[me]
    if _ECON_HISTORY is None or observation["step"] == 0 or _ECON_HISTORY["seat"] != me:
        _ECON_HISTORY = {"seat":me,"rival_tree_income":rival_advantage}
        _ECON_COVER = {"planter_slots":0,"threatened_collector_slots":0}
    _ECON_HISTORY["rival_tree_income"] = .85*_ECON_HISTORY["rival_tree_income"] + .15*rival_advantage
    _ECON_HISTORY["weaker_tree_economy"] = (_ECON_HISTORY["rival_tree_income"] > 2.
                                           and len(players[1-me][3]) >= 2)


def _student_fraction(f, x, aux, slot):
    if slot >= 10:
        return .25
    cell = int(x["tok_pos"][slot])
    if int(x["tok_kind"][slot]) == 1:
        if (_ECON_HISTORY["weaker_tree_economy"]
                and np.any(aux["en_tree"] & (f.DIST[cell] <= 4))):
            _ECON_COVER["planter_slots"] += 1
            return _ECON_FRACTIONS[0]
    else:
        cargo = float(x["tok_cargo"][slot])
        if (cargo >= 45 and 1 <= aux["dob"][cell] <= 6
                and np.any(aux["en_min"][f.DEST[cell]] <= cargo)):
            _ECON_COVER["threatened_collector_slots"] += 1
            return _ECON_FRACTIONS[1]
    return .25
