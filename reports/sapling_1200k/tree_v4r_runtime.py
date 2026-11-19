# Carbon League's Sapling V4r. Both int8 actors are authors' own Sapling checkpoints.
_POLICY = {"collector": "cash_v3", "planter": "tree_v4", "base": "cash_v3",
           "style": 2, "minimum_existing_collectors_for_extra_planter": 2,
           "extra_planter_count": 1, "reserve": "one plant + one collector"}
_NETS = None
_FEATURES = {}
_LAST_BUDGET = {}
_F = sys.modules["cnet2_feat"]
_NP = sys.modules["cnet2_np"]


def _budget(obs, cfg):
    players = obs["players"]
    own = players[obs["player"]]
    counts = [sum(w[2] == "COLLECTOR" for w in p[2].values()) for p in players]
    all_workers = sum(len(p[2]) for p in players)
    own_collectors = counts[obs["player"]]
    all_collectors = sum(counts)
    cost_c = cfg.get("recCollectorCost", 30) + all_collectors * cfg.get("recCollectorIncreaseCost", 0)
    cost_p = cfg.get("recPlanterCost", 30) + (all_workers - all_collectors) * cfg.get("recPlanterIncreaseCost", 0)
    plant = float(cfg.get("plantCost", 20))
    if len(own[3]) > cfg.get("playerTreeProtectiveNumber", 0):
        alive = sum(tree[1] < cfg.get("treeLifespan", 50) for p in players for tree in p[3].values())
        plant += cfg.get("plantCostInflationRatio", 5) * cfg.get("plantCostInflationBase", 1.235) ** alive
    canonical = _F.CANON if obs["player"] else _F.GRID
    lifespan = cfg.get("treeLifespan", 50)
    live_owners = {int(canonical[tree[0]]): p for p, player in enumerate(players)
                   for tree in player[3].values() if tree[1] < lifespan}
    bases = {int(canonical[pos]) for player in players for pos in player[1].values()}
    return {"cash": float(own[0]), "plant": plant, "seize": cfg.get("seizeCost", 20),
            "owners": live_owners, "bases": bases, "seat": obs["player"], "held_cells": set(),
            "held_tree_cost": 0., "redirected": 0, "unavoidable": 0, "existing_collectors": own_collectors,
            "planters": len(own[2]) - own_collectors, "extra": False,
            "remaining_slots": int(cfg.get("workerLimit", 10)) - len(own[2]),
            "cost": (0., cost_c, cost_p), "reserve": cost_c + plant}


def _choose_base(budget, scores3, scores4, decoded_legal):
    legal = decoded_legal.copy()
    for a in (1, 2):
        legal[a] &= budget["remaining_slots"] > 0 and budget["cash"] >= budget["cost"][a]
        if budget["held_tree_cost"] > 0:
            legal[a] &= budget["cash"] - budget["cost"][a] >= budget["cost"][1]
        if budget["extra"]:
            legal[a] &= a == 1 and budget["cash"] - budget["cost"][a] >= budget["reserve"]
    action = int(np.argmax(np.where(legal, scores3, -np.inf)))
    v4_action = int(np.argmax(np.where(legal, scores4, -np.inf)))
    if (action != 2 and v4_action == 2 and budget["existing_collectors"] >= 2
            and budget["planters"] == 0 and budget["cash"] - budget["cost"][2] >= budget["reserve"]):
        action = 2
        budget["extra"] = True
    if action:
        budget["cash"] -= budget["cost"][action]
        budget["remaining_slots"] -= 1
        budget["planters"] += int(action == 2)
    return action


def _planter_action(budget, pos, v3, v4, legal):
    action = int(np.argmax(np.where(legal, v4, -np.inf)))
    if action or pos in budget["bases"] or pos in budget["held_cells"]:
        return action
    owner = budget["owners"].get(pos)
    if owner == budget["seat"]:
        return action
    fee = budget["plant"] if owner is None else budget["seize"]
    if budget["cash"] - fee < budget["cost"][1]:
        if legal[1:].any():
            budget["redirected"] += 1
            return 1 + int(np.argmax(np.where(legal[1:], v3[1:], -np.inf)))
        budget["unavoidable"] += 1
    budget["held_cells"].add(pos)
    budget["held_tree_cost"] += fee
    budget["cash"] -= fee
    return 0


def _choose(x, aux, budget):
    states = []
    for policy in _NETS:
        tokens, context = policy.encode(x, _POLICY["style"])
        states.append([policy, tokens, context, policy.lin(context, "h0")])
    prefix = _F.Prefix(x, aux)
    previous = 8
    labels = np.full(_F.OWN, -1, np.int64)
    for s in np.flatnonzero(x["tok_mask"][:_F.OWN]):
        unit, moves, legal = prefix.feats(s)
        logits = []
        for state in states:
            net, tokens, context, h = state
            inp = net.lin(np.concatenate([tokens[s], context, unit, moves.reshape(-1)]), "din")
            h = net.gru(inp + net.w["aemb.weight"][previous], h)
            state[3] = h
            head = net.lin(h, "whead") if s < 10 else np.concatenate([net.lin(h, "bhead"), np.zeros(2, np.float32)])
            logits.append(head + net.lin(moves, "V") @ net.lin(h, "U"))
        if s < 10:
            if int(x["tok_kind"][s]) == 1:
                action = _planter_action(budget, int(x["tok_pos"][s]), logits[0], logits[1], legal)
            else:
                action = int(np.argmax(np.where(legal, logits[0], -np.inf)))
        else:
            action = _choose_base(budget, logits[0], logits[1], legal)
        prefix.advance(s, action)
        labels[s] = action
        previous = action if s < 10 else action + 5
    return labels


def agent(observation, configuration=None):
    global _NETS
    if _NETS is None:
        _NETS = []
        for blob, quant, meta in ((_W64, _QMETA, _WMETA), (_V4_W64, _V4_QMETA, _V4_WMETA)):
            w = _dequantize(zlib.decompress(base64.b64decode(blob)), quant)
            w["meta"] = np.array(meta)
            _NETS.append(_NP.Policy(w))
    seat = observation["player"]
    if seat not in _FEATURES or observation["step"] <= _FEATURES[seat].step:
        _FEATURES[seat] = _F.Feat(seat)
    feat = _FEATURES[seat]
    x = feat.observe(observation)
    budget = _budget(observation, configuration or {})
    labels = _choose(x, feat.aux, budget)
    _LAST_BUDGET[seat] = {k: budget[k] for k in ("extra", "held_tree_cost", "redirected", "unavoidable")}
    return feat.to_actions(labels)
