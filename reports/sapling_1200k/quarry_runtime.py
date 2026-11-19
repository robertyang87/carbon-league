"""A single uncontested, still-productive worksite can retain a lightly loaded worker."""
def quarry_decide(F, x, prefix, slot, normal, logits, legal, changes):
    if changes or slot >= 10 or normal == 0 or x["tok_kind"][slot] != 0:
        return normal
    pos, cargo = int(x["tok_pos"][slot]), float(x["tok_cargo"][slot])
    if (cargo > 75. or x["cash"] < 60. or x["left"] < 18 or prefix.reserved[pos] or
            prefix.a["thr"][F.DEST[pos]].any() or prefix.a["est"][pos] < .55):
        return normal
    # An immediate bank and high-priority territorial assignments remain with
    # the network. Require working capital for simultaneous recruits/seizures.
    if (prefix.a["own_base"][F.DEST[pos,normal]] and cargo >= 35.) or prefix.cash < 60.:
        return normal
    freshness = (299 - x["left"]) - _FEAT.last_step[F.CANON if _FEAT.seat else F.GRID]
    values = quarry_values(F, x, prefix.a, slot, freshness)
    if values is None or not np.isfinite(values[0]) or values[0] < 35.:
        return normal
    next_site = max(0., float(np.max(values[1:])))
    lead = values[0] - next_site
    if lead < 12. or logits[normal] - logits[0] > min(1.4, (lead - 7.) / 12.):
        return normal
    changes.append((slot,normal,0))
    return 0
