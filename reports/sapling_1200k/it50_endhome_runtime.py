# Additional runtime on top of the frozen 75:25 shared-prefix decoder.
from collections import Counter, deque

_RETURN_ENABLED = True
_RETURN_COUNTS = Counter()
_RETURN_LAST = ()


def _return_distance(F, aux, cargo):
    # Distances under currently visible threats. Each subsequent turn replans.
    # Upper edge of the float16 cargo bin also treats near-ties conservatively.
    cargo_hi = float(np.nextafter(np.float16(cargo), np.float16(np.inf)))
    safe = (aux["en_min"] > cargo_hi) & ~aux["en_base"]
    roots = np.flatnonzero(aux["own_base"] & safe)
    dist = np.full(F.N, F.N + 1, dtype=int)
    dist[roots] = 0
    todo = deque(roots.tolist())
    while todo:
        pos = todo.popleft()
        for dest in F.DEST[pos, 1:]:
            if safe[dest] and dist[dest] > dist[pos] + 1:
                dist[dest] = dist[pos] + 1
                todo.append(dest)
    return dist


def _return_decision(F, x, prefix, slot, normal, scores, legal, changes):
    if slot >= 10 or x["tok_kind"][slot] != 0 or x["tok_cargo"][slot] < 30:
        return normal
    left, pos = x["left"], int(x["tok_pos"][slot])
    aux = prefix.a
    d = aux["dob"][pos]
    if not 0 < d <= left <= d + 1:
        return normal
    _RETURN_COUNTS["tight_loaded_slots"] += 1
    dests = F.DEST[pos]
    if aux["dob"][dests[normal]] <= left - 1:
        _RETURN_COUNTS["policy_still_reachable"] += 1
        return normal
    _RETURN_COUNTS["policy_misses_deadline"] += 1
    dist = _return_distance(F, aux, float(x["tok_cargo"][slot]))
    options = legal & (dist[dests] <= left - 1) & (prefix.reserved[dests] == 0)
    if not options.any():
        _RETURN_COUNTS["no_visible_route"] += 1
        return normal
    # Preserve the mixed network's ordering among timely safe first moves.
    chosen = int(np.argmax(np.where(options, scores, -np.inf)))
    changes.append((slot, normal, chosen))
    return chosen


def _return_destinations(F, x, labels):
    dests = {}
    for slot, action in enumerate(labels):
        if action < 0:
            continue
        pos = int(x["tok_pos"][slot])
        if slot < 10:
            dests[slot] = int(F.DEST[pos, action])
        elif action > 0:
            dests[slot] = pos
    return dests


def _return_joint_ok(F, x, aux, base, proposed, changes):
    parent_dests = _return_destinations(F, x, base)
    new_dests = _return_destinations(F, x, proposed)
    # Include later friendly units and new recruits. A collision involving
    # even one new participant is a new risk, including at our own base.
    old_pairs = {(a, b) for a in parent_dests for b in parent_dests
                 if a < b and parent_dests[a] == parent_dests[b]}
    new_pairs = {(a, b) for a in new_dests for b in new_dests
                 if a < b and new_dests[a] == new_dests[b]}
    changed_slots = {s for s, _, _ in changes}
    if (new_pairs - old_pairs) or any(a in changed_slots or b in changed_slots for a, b in new_pairs):
        return False
    for s, dest in new_dests.items():
        if s >= 10 or parent_dests[s] == dest or x["tok_kind"][s] != 0:
            continue
        cargo = float(np.nextafter(np.float16(x["tok_cargo"][s]), np.float16(np.inf)))
        if aux["en_min"][dest] <= cargo < aux["en_min"][parent_dests[s]]:
            return False
        if aux["en_base"][dest] and not aux["en_base"][parent_dests[s]]:
            return False
    return True


def _mixed_actions(x, aux):
    global _RETURN_LAST
    _RETURN_LAST = ()
    if x["left"] == 299:
        _RETURN_COUNTS.clear()
    if not _RETURN_ENABLED or not 1 <= x["left"] <= 15:
        return _return_decode(x, aux)
    _RETURN_COUNTS["late_roots"] += 1
    changes = []
    saved = {role: detail.copy() for role, detail in _COUNTS.items()}
    proposed = _return_decode(x, aux, changes)
    if not changes:
        return proposed
    counts_after = {role: detail.copy() for role, detail in _COUNTS.items()}
    _COUNTS.update(saved)
    base = _return_decode(x, aux)
    F = sys.modules["cnet2_feat"]
    if not _return_joint_ok(F, x, aux, base, proposed, changes):
        _RETURN_COUNTS["rejected_joint_roots"] += 1
        return base
    _COUNTS.update(counts_after)
    _RETURN_COUNTS["changed_roots"] += 1
    _RETURN_COUNTS["overridden_slots"] += len(changes)
    _RETURN_LAST = tuple(changes)
    return proposed
