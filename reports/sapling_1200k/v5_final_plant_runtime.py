"""V5 final-step cash comparison. All inputs are from the acting seat's public state."""
_F = sys.modules["cnet2_feat"]
_NP = sys.modules["cnet2_np"]
_POLICY = None
_FEATURES = {}


def _final_choice(pol, feat, x, y, logits, obs, cfg):
    if obs["step"] != cfg.get("episodeSteps", 300) - 2:
        return y, "outside_final_step"
    seat = obs["player"]
    players = obs["players"]
    own, enemy = players[seat], players[1-seat]
    bases = {p for side in players for p in side[1].values()}
    life = cfg.get("treeLifespan", 50)
    live_trees = {p: owner for owner,side in enumerate(players) for p,age in side[3].values() if age < life}
    live_slots = np.flatnonzero(x["tok_mask"][:10])
    charges = []
    for s in live_slots:
        pos, _, role = own[2][feat.own_uid[s]]
        if role == "PLANTER" and y[s] == 0 and pos not in bases and live_trees.get(pos) != seat:
            charges.append((s,pos))
    if len(charges) != 1:
        return y, "no_unique_payable_planter"
    slot, pos = charges[0]
    if pos in live_trees:
        return y, "preserve_seizure"
    fee = float(cfg.get("plantCost",20))
    if len(own[3]) > cfg.get("playerTreeProtectiveNumber",0):
        fee += cfg.get("plantCostInflationRatio",5) * cfg.get("plantCostInflationBase",1.235) ** len(live_trees)
    kinds = [w[2] for p in players for w in p[2].values()]
    reserve = float(own[0])
    for s in (10,11):
        if y[s] not in (1,2):
            continue
        role = "Collector" if y[s] == 1 else "Planter"
        reserve -= cfg.get("rec"+role+"Cost",30) + kinds.count(role.upper()) * cfg.get("rec"+role+"IncreaseCost",0)
    if fee <= 0 or reserve < fee:
        return y, "insufficient_confirmed_cash"
    # Public positions are unrotated; logits and labels use the seat-canonical board.
    def destination(s, action):
        cell = own[2][feat.own_uid[s]][0]
        move = _F.CANON_MOVE[action] if seat else action
        return int(_F.DEST[cell, move])
    unsafe = {int(v) for p,_,_ in enemy[2].values() for v in _F.DEST[p]}
    # A newborn enemy can stay on its base. Keep both bases and known
    # original recruitment choices fixed, plus conservative enemy reach.
    if pos in unsafe:
        return y, "original_cell_contested"
    occupied = {destination(s, y[s]) for s in live_slots if s!=slot}
    if pos in occupied:
        return y, "original_friendly_collision"
    for move in np.argsort(-logits[slot],kind="stable"):
        if not move or logits[slot,move] <= _NP.NEG or destination(slot,move) in (unsafe | occupied | bases):
            continue
        def pivot(s, raw_logits, prefix, selected, pivot_slot=slot, alternate=int(move)):
            return alternate if s==pivot_slot else None
        alt=pol.act(x,feat.aux,teacher=2,hook=pivot)[0]
        if np.flatnonzero(y!=alt).tolist()==[int(slot)]:
            return alt, "avoid_new_plant"
    return y, "no_safe_prefix_preserving_alternative"


def agent(observation, configuration=None):
    global _POLICY
    if _POLICY is None:
        weights=_dequantize(zlib.decompress(base64.b64decode(_W64)),_QMETA)
        weights["meta"]=np.array(_WMETA)
        _POLICY=_NP.Policy(weights)
    seat=observation["player"]
    if seat not in _FEATURES or observation["step"]<=_FEATURES[seat].step:
        _FEATURES[seat]=_F.Feat(seat)
    feat=_FEATURES[seat]
    x=feat.observe(observation)
    y,logits,_=_POLICY.act(x,feat.aux,teacher=2)
    alt,reason=_final_choice(_POLICY,feat,x,y,logits,observation,configuration or {})
    agent.last_stats={"final_plant_reason":reason,"changed_slots":int(np.count_nonzero(y!=alt))}
    return feat.to_actions(alt)
