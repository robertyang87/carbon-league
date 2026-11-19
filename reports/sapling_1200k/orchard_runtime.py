# Public basis, caller-supplied configuration and current planter observations only.
from collections import Counter

_ORCHARD_COUNTS = Counter()
_ORCHARD_LAST = ()
_orchard_actor = agent


def agent(observation, configuration=None):
    global _ORCHARD_OBS, _ORCHARD_CFG
    _ORCHARD_OBS = {k: observation[k] for k in ("players","player","carbon","step","trees")}
    _ORCHARD_CFG = configuration or {}
    if observation["step"] == 0:
        _ORCHARD_COUNTS.clear()
    return _orchard_actor(observation, configuration)


def _orchard_decide(F, x, prefix, slot, normal, logits, legal, changes):
    if slot >= 10 or x["tok_kind"][slot] != 1 or changes:
        return normal
    pos = int(x["tok_pos"][slot])
    seat = _ORCHARD_OBS["player"]
    raw_pos = int(F.CANON[pos]) if seat else pos
    opportunity = orchard_features(_ORCHARD_OBS, _ORCHARD_OBS["carbon"], seat, raw_pos,
                                   _ORCHARD_OBS["step"], _ORCHARD_CFG)
    if opportunity is None:
        return normal
    kind, cost, features = opportunity
    if prefix.cash < cost:
        return normal
    net = orchard_predict(_ORCHARD_MODELS[kind], features) - cost
    safe = legal & (prefix.reserved[F.DEST[pos]] == 0) & (prefix.undecided[F.DEST[pos]] == 0)
    safe &= ~np.isfinite(prefix.a["en_min"][F.DEST[pos]]) & (prefix.a["en_pl"][F.DEST[pos]] == 0)
    _ORCHARD_COUNTS["eligible"] += 1
    action = normal
    if normal == 0 and net < -25.:
        safe[0] = False
        if safe.any():
            alternative = int(np.argmax(np.where(safe,logits,-np.inf)))
            if logits[0]-logits[alternative] < min(3., -net/65.):
                action = alternative
    elif normal != 0 and safe[0] and prefix.cash >= cost+90. and net > 50.:
        if logits[normal]-logits[0] < min(2.,net/100.):
            action = 0
    if action != normal:
        changes.append((slot,normal,action))
    return action


def _mixed_actions(x,aux):
    global _ORCHARD_LAST
    _ORCHARD_LAST = ()
    encoded = _orchard_encode(x)
    if not np.any(x["tok_mask"][:10] & (x["tok_kind"][:10] == 1)):
        return _orchard_decode(x,aux,encoded)
    before = {role:d.copy() for role,d in _COUNTS.items()}
    changes = []
    proposed = _orchard_decode(x,aux,encoded,changes)
    if not changes:
        return proposed
    after = {role:d.copy() for role,d in _COUNTS.items()}
    _COUNTS.update(before)
    base = _orchard_decode(x,aux,encoded)
    if not _return_joint_ok(sys.modules["cnet2_feat"],x,aux,base,proposed,changes):
        _ORCHARD_COUNTS["veto"] += 1
        return base
    _COUNTS.update(after)
    _ORCHARD_LAST = tuple(changes)
    _ORCHARD_COUNTS["encouraged_purchase" if changes[0][2]==0 else "avoided_purchase"] += 1
    return proposed
