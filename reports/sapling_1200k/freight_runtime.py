# Cargo routing supplement for the deployed 75:25 neural actor.
from collections import Counter

_FREIGHT_COUNTS = Counter()
_FREIGHT_LAST = ()
_FREIGHT_ENABLED = True


def _freight_values(F, x, prefix, slot):
    """Discounted net banked cargo per first action, using only current public observations."""
    aux = prefix.a
    pos = int(x["tok_pos"][slot])
    cargo = float(x["tok_cargo"][slot])
    home = aux["own_base"]
    danger = aux["en_min"] <= float(np.nextafter(np.float16(cargo), np.float16(np.inf)))
    # Several trees may simultaneously absorb ten percent of the same starting
    # cargo. Own-base deposit resolves before any tree absorbs the load.
    toll = np.minimum(1., .1 * (aux["toll"] + aux["en_tree"]))
    horizon = min(int(x["left"]), int(aux["dob"][pos]) + 3)
    # Per-carbon net receipts for paths that reach either home in time. Each
    # absorption reduces the delivered load and also credits the rival. Future
    # threat reach fades; opponents choose simultaneously and will move.
    continuation = np.full(F.N, -np.inf)
    for t in range(horizon, 0, -1):
        survival = 1. - danger * (.48 if t == 1 else (.20 if t == 2 else .08 if t == 3 else .03))
        retention = (1. - toll) * survival
        value = np.full(F.N, -np.inf)
        viable = np.isfinite(continuation) & (retention > 0.)
        value[viable] = .92 * retention[viable] * continuation[viable] - toll[viable] * survival[viable]
        value[home] = survival[home]
        value[aux["en_base"]] = -np.inf
        if t > 1:
            continuation = value[F.DEST[:, 1:]].max(axis=1)
    destinations = F.DEST[pos]
    values = value[destinations] * cargo
    if not (aux["own_tree"][pos] or aux["en_tree"][pos] or home[pos]):
        values[0] = value[pos] * (cargo + 30. * aux["est"][pos])
    return values, danger[destinations]


def _freight_decide(F, x, prefix, slot, normal, logits, legal, changes):
    if slot >= 10 or int(x["tok_kind"][slot]) != 0:
        return normal
    pos, cargo = int(x["tok_pos"][slot]), float(x["tok_cargo"][slot])
    distance = int(prefix.a["dob"][pos])
    if not 1 <= distance <= 6 or distance > x["left"] or cargo < (24 if x["left"] <= distance + 2 else 55):
        return normal
    _FREIGHT_COUNTS["eligible_slots"] += 1
    values, danger = _freight_values(F, x, prefix, slot)
    dest = F.DEST[pos]
    # A frightened collector mustn't choose a new interception square, collide
    # with an earlier teammate or count on a later teammate vacating its tile.
    allowed = legal & ~danger & (prefix.reserved[dest] == 0) & (prefix.undecided[dest] == 0)
    feasible = allowed & np.isfinite(values) & (values > 0.)
    if not feasible.any():
        return normal
    # Compare in banked-carbon units. The shared-prefix neural margin can veto
    # a modest geometric advantage and preserves ordering for equivalent routes.
    # An unreachable parent route at the real deadline has zero expected deposit.
    normal_value = max(0., float(values[normal]))
    edge = np.where(feasible, values - normal_value, -np.inf)
    adjusted = edge - 9. * np.maximum(0., logits[normal] - logits)
    choice = int(np.argmax(adjusted))
    if choice == normal or adjusted[choice] <= max(4., .035 * cargo):
        return normal
    changes.append((slot, normal, choice))
    return choice


def _mixed_actions(x, aux):
    global _FREIGHT_LAST
    _FREIGHT_LAST = ()
    if x["left"] == 299:
        _FREIGHT_COUNTS.clear()
    encoded = _freight_encode(x)
    if not _FREIGHT_ENABLED or not 1 <= x["left"] <= 299:
        return _freight_decode(x, aux, encoded)
    before = {role: d.copy() for role, d in _COUNTS.items()}
    changes = []
    proposed = _freight_decode(x, aux, encoded, changes)
    if not changes:
        return proposed
    after = {role: d.copy() for role, d in _COUNTS.items()}
    _COUNTS.update(before)
    base = _freight_decode(x, aux, encoded)
    if not _return_joint_ok(sys.modules["cnet2_feat"], x, aux, base, proposed, changes):
        _FREIGHT_COUNTS["joint_veto"] += 1
        return base
    _COUNTS.update(after)
    _FREIGHT_LAST = tuple(changes)
    _FREIGHT_COUNTS["changed_roots"] += 1
    _FREIGHT_COUNTS["changed_slots"] += len(changes)
    return proposed
