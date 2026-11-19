# Runtime payload appended by build_it50_v3.py to the Sapling int8 pack.
_HYBRID = {"expert": "ppou2_m_it50", "student": "Sapling cash_v3",
           "expert_weight": 0.75, "student_weight": 0.25, "style": 2,
           "guard": False, "search": False}
_NETS = None
_FEAT = None
_COUNTS = {}


def _mixed_actions(x, aux):
    F = sys.modules["cnet2_feat"]
    states = []
    for policy in _NETS:
        tokens, context = policy.encode(x, _HYBRID["style"])
        states.append([policy, tokens, context, policy.lin(context, "h0")])
    prefix = F.Prefix(x, aux)
    previous = 8
    labels = np.full(F.OWN, -1, np.int64)
    for slot in range(F.OWN):
        if not x["tok_mask"][slot]:
            continue
        unit, move, legal = prefix.feats(slot)
        scores = []
        for state in states:
            policy, tokens, context, hidden = state
            features = np.concatenate([tokens[slot], context, unit, move.reshape(-1)])
            hidden = policy.gru(policy.lin(features, "din") + policy.w["aemb.weight"][previous], hidden)
            state[3] = hidden
            if slot < 10:
                base = policy.lin(hidden, "whead")
            else:
                base = np.concatenate([policy.lin(hidden, "bhead"), np.zeros(2, np.float32)])
            scores.append(base + policy.lin(move, "V") @ policy.lin(hidden, "U"))
        combined = _HYBRID["expert_weight"] * scores[0] + _HYBRID["student_weight"] * scores[1]
        action = int(np.argmax(np.where(legal, combined, -np.inf)))
        expert, student = [int(np.argmax(np.where(legal, score, -np.inf))) for score in scores]
        role = "base" if slot >= 10 else ("planter" if int(x["tok_kind"][slot]) == 1 else "collector")
        detail = _COUNTS[role]
        detail["valid_slots"] += 1
        detail["disagree"] += int(expert != student)
        detail["selected_student_in_disagreement"] += int(expert != student and action == student)
        detail["selected_third_in_disagreement"] += int(expert != student and action not in (expert, student))
        labels[slot] = action
        prefix.advance(slot, action)
        previous = action + (0 if slot < 10 else 5)
    return labels


def agent(observation, configuration=None):
    global _NETS, _FEAT, _COUNTS
    if _NETS is None:
        policy = sys.modules["cnet2_np"].Policy
        _NETS = []
        for blob, qmeta, wmeta in ((_B1, _Q1, _M1), (_W64, _QMETA, _WMETA)):
            weights = _dequantize(zlib.decompress(base64.b64decode(blob)), qmeta)
            weights["meta"] = np.array(wmeta)
            _NETS.append(policy(weights))
    if _FEAT is None or observation["step"] == 0 or _FEAT.seat != observation["player"]:
        _FEAT = sys.modules["cnet2_feat"].Feat(observation["player"])
        _COUNTS = {role: {"valid_slots": 0, "disagree": 0, "selected_student_in_disagreement": 0,
                          "selected_third_in_disagreement": 0}
                   for role in ("base", "collector", "planter")}
    x = _FEAT.observe(observation)
    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))
