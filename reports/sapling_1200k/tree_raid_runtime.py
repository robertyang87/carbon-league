# Inference uses only the public root and the same-step mixed Sapling logits.
_RAID_COUNTS = {}
_RAID_TARGETS = {}
_RAID_APPROACH = {}
_RAID_LAST = ()


def _raid_bonus(F, x, prefix, slot, normal, logits, legal, changes):
    if slot >= 10 or x["tok_kind"][slot] != 0:
        return normal
    pos = int(x["tok_pos"][slot])
    cargo, dest = float(x["tok_cargo"][slot]), F.DEST[pos]
    adjusted = logits.copy()
    attributed = {}
    for a, to in enumerate(dest):
        if not legal[a] or prefix.reserved[to]:
            continue
        hi = float(np.nextafter(np.float16(cargo), np.float16(np.inf)))
        if prefix.a["en_min"][to] <= hi or prefix.a["en_base"][to]:
            continue
        alternatives = []
        if to in _RAID_TARGETS:
            p, life = _RAID_TARGETS[to]
            alternatives.append((p * life, "occupied", int(to)))
        for tree, (entry, reduction, life) in _RAID_APPROACH.items():
            # Get within reach before the opponent has a full turn to seize. A
            # committed earlier unit can already provide this coverage.
            if F.DIST[pos, tree] == 2 and F.DIST[to, tree] <= 1 and not prefix.reserved[F.DEST[tree]].any():
                alternatives.append((entry * reduction * life, "approach", tree))
        if alternatives:
            benefit, stage, target = max(alternatives)
            bank_now = prefix.a["own_base"][dest[normal]] and cargo > 0
            bonus = 2.5 * benefit / (1 + (cargo / 30 if bank_now else cargo / 150))
            adjusted[a] += bonus
            attributed[a] = (stage, int(target), float(bonus))
    proposed = int(np.argmax(np.where(legal, adjusted, -np.inf)))
    if proposed != normal:
        changes.append((slot, normal, proposed, int(dest[proposed]), *attributed[proposed]))
    return proposed


def _raid_collisions(F, x, labels):
    arrivals = {}
    for s, a in enumerate(labels):
        if a < 0:
            continue
        pos = int(x["tok_pos"][s])
        if s >= 10 and a == 0:
            continue
        to = int(F.DEST[pos, a]) if s < 10 else pos
        arrivals.setdefault(to, []).append(s)
    return {(to,min(a,b),max(a,b)) for to, slots in arrivals.items()
            for i, a in enumerate(slots) for b in slots[i+1:]}


def _mixed_actions(x, aux):
    if not (_RAID_TARGETS or _RAID_APPROACH):
        return _raid_decode(x, aux)
    changes = []
    before = {role:counts.copy() for role, counts in _COUNTS.items()}
    proposed = _raid_decode(x, aux, changes)
    if not changes:
        return proposed
    after = {role:counts.copy() for role, counts in _COUNTS.items()}
    _COUNTS.update(before)
    original = _raid_decode(x, aux)
    F = sys.modules['cnet2_feat']
    conflicts = _raid_collisions(F,x,proposed)
    screened = {change[3] for change in changes}
    if conflicts - _raid_collisions(F,x,original) or any(cell in screened for cell,_,_ in conflicts):
        _RAID_COUNTS["joint_conflict_veto"] += 1
        return original
    _COUNTS.update(after)
    global _RAID_LAST
    _RAID_LAST = tuple(changes)
    _RAID_COUNTS["changed_roots"] += 1
    return proposed


def _raid_sigmoid(x):
    return 1. / (1. + math.exp(-max(-25., min(float(x),25.))))


def agent(observation, configuration=None):
    global _RAID_TARGETS, _RAID_APPROACH, _RAID_LAST
    _RAID_TARGETS, _RAID_APPROACH, _RAID_LAST = {}, {}, ()
    F = sys.modules['cnet2_feat']
    if observation['step'] == 0 or not _RAID_COUNTS:
        _RAID_COUNTS.update(eligible_roots=0, changed_roots=0, joint_conflict_veto=0)
    if 50 <= observation["step"] < 250:
        cfg = configuration or {}
        me = observation['player']
        cash = observation['players'][1-me][0]
        if cash >= cfg.get('seizeCost', 20):
            for uid, cell, life, features in tree_raid_rows(observation, F, cfg):
                z = float(np.dot(features, _RAID_WEIGHTS))
                p = 1. / (1. + math.exp(-max(-25., min(z, 25.))))
                canonical = int(F.CANON[cell]) if me else cell
                _RAID_TARGETS[canonical] = (p, life / cfg.get('treeLifespan', 50))
            for uid, cell, life, features in tree_raid_rows(observation, F, cfg, approaches=True):
                # An existing collector in range already covers this tree.
                if features[4] or features[7] or life <= 1:
                    continue
                entry = _raid_sigmoid(np.dot(features, _RAID_ENTRY_WEIGHTS))
                unguarded = _raid_sigmoid(np.dot(features, _RAID_WEIGHTS))
                guarded = list(features)
                guarded[4] = 1
                # A conservative hypothetical single defender; no low-cargo
                # bonus. The stay difference is observational and not causal.
                reduction = max(0., unguarded - _raid_sigmoid(np.dot(guarded, _RAID_WEIGHTS)))
                canonical = int(F.CANON[cell]) if me else cell
                old = _RAID_APPROACH.get(canonical)
                if old is None or entry > old[0]:
                    _RAID_APPROACH[canonical] = (entry,reduction,life/cfg.get('treeLifespan',50))
    if _RAID_TARGETS or _RAID_APPROACH:
        _RAID_COUNTS['eligible_roots'] += 1
    return _raid_agent(observation, configuration)
