"""CARBON baseline v1: independent exploration, collection and delivery.

Uses only the current partial observation plus within-episode memory.
No external strategy code, model, files, or network access.
"""

_STATE = {}
_ACTIONS = (
    ("UP", -1, 0),
    ("RIGHT", 0, 1),
    ("DOWN", 1, 0),
    ("LEFT", 0, -1),
)


def _distance(a, b, size):
    ar, ac = divmod(a, size)
    br, bc = divmod(b, size)
    dr = abs(ar - br)
    dc = abs(ac - bc)
    return min(dr, size - dr) + min(dc, size - dc)


def _move(position, action, size):
    if action is None:
        return position
    row, col = divmod(position, size)
    for name, dr, dc in _ACTIONS:
        if name == action:
            return ((row + dr) % size) * size + (col + dc) % size
    return position


def agent(observation, configuration):
    global _STATE

    size = configuration["size"]
    step = observation["step"]
    episode_steps = configuration["episodeSteps"]
    if step == 0 or _STATE.get("size") != size:
        _STATE = {"size": size, "seen": {}, "targets": {}, "homes": {}}

    seen = _STATE["seen"]
    targets = _STATE["targets"]
    homes = _STATE["homes"]
    player_id = observation["player"]
    me = observation["players"][player_id]
    cash, centers, workers, _ = me
    collectors = {
        worker_id: data
        for worker_id, data in workers.items()
        if data[2] == "COLLECTOR"
    }

    # Nonzero entries are necessarily visible. Zero entries are only recorded
    # at our collectors' positions, so invisible cells are never treated as empty.
    carbon_map = observation["carbon"]
    for position, value in enumerate(carbon_map):
        if value > 0:
            seen[position] = (float(value), step)
    for position, _, _ in collectors.values():
        seen[position] = (float(carbon_map[position]), step)

    alive_ids = set(collectors)
    targets = {key: value for key, value in targets.items() if key in alive_ids}
    homes = {key: value for key, value in homes.items() if key in alive_ids}
    _STATE["targets"] = targets
    _STATE["homes"] = homes

    own_bases = set(centers.values())
    opponent_bases = set()
    opponent_workers = {}
    opponent_trees = set()
    all_worker_positions = set()
    for other_id, player in enumerate(observation["players"]):
        for worker_id, data in player[2].items():
            position = data[0]
            all_worker_positions.add(position)
            if other_id != player_id:
                opponent_workers[position] = data
        if other_id != player_id:
            opponent_bases.update(player[1].values())
            opponent_trees.update(data[0] for data in player[3].values())

    blocked_targets = own_bases | opponent_bases | opponent_trees | all_worker_positions
    own_positions = {data[0] for data in collectors.values()}
    for worker_id, (position, _, _) in collectors.items():
        if worker_id not in homes and own_bases:
            homes[worker_id] = min(own_bases, key=lambda base: _distance(position, base, size))

    danger = set()
    for tree in opponent_trees:
        row, col = divmod(tree, size)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                danger.add(((row + dr) % size) * size + (col + dc) % size)

    actions = {}
    reserved_destinations = set()
    assigned_targets = set(targets.values())

    def choose_target(worker_id, position):
        previous = targets.get(worker_id)
        if previous is not None and previous != position and previous not in blocked_targets:
            if previous not in seen or seen[previous][0] >= 1.0:
                return previous

        other_targets = assigned_targets - {previous}
        known = []
        for candidate, (value, seen_step) in seen.items():
            age = step - seen_step
            if (
                candidate != position
                and candidate not in blocked_targets
                and candidate not in other_targets
                and candidate not in danger
                and value >= 4.0
                and age <= 80
            ):
                distance = _distance(position, candidate, size)
                score = value / (distance + 3.0) / (1.0 + 0.01 * age)
                known.append((-score, candidate))
        if known:
            known.sort()
            return known[0][1]

        unknown = []
        home = homes.get(worker_id, position)
        for candidate in range(size * size):
            if (
                candidate not in seen
                and candidate not in blocked_targets
                and candidate not in other_targets
                and candidate not in danger
            ):
                distance = _distance(position, candidate, size)
                spread = _distance(home, candidate, size)
                unknown.append((distance * 4 + spread, candidate))
        if unknown:
            unknown.sort()
            return unknown[0][1]

        fallback = [
            candidate
            for candidate, (value, _) in seen.items()
            if value >= 1.0 and candidate not in blocked_targets and candidate not in other_targets
        ]
        if fallback:
            return min(fallback, key=lambda p: (_distance(position, p, size), p))
        return None

    # Loaded collectors choose paths first; empty explorers fill the remaining space.
    ordered = sorted(collectors.items(), key=lambda item: (-item[1][1], item[0]))
    for worker_id, data in ordered:
        position, cargo, _ = data
        cargo = float(cargo)
        current_carbon = float(carbon_map[position])
        remaining_steps = episode_steps - step
        nearest_base = None
        base_distance = size * 2
        if own_bases:
            nearest_base = min(own_bases, key=lambda base: (_distance(position, base, size), base))
            base_distance = _distance(position, nearest_base, size)

        must_return = cargo > 0 and remaining_steps <= base_distance + 2
        returning = must_return or cargo >= 25.0 or (cargo >= 12.0 and current_carbon < 1.0)
        if returning and nearest_base is not None and position != nearest_base:
            target = nearest_base
            targets.pop(worker_id, None)
        elif (
            position not in own_bases
            and position not in opponent_trees
            and current_carbon >= 1.0
            and cargo < 25.0
            and not must_return
        ):
            # Omitting an action makes a collector stay and collect this turn.
            reserved_destinations.add(position)
            continue
        else:
            target = choose_target(worker_id, position)

        if target is None:
            reserved_destinations.add(position)
            targets.pop(worker_id, None)
            continue
        if target == position:
            targets.pop(worker_id, None)
            reserved_destinations.add(position)
            continue
        if not returning:
            targets[worker_id] = target
            assigned_targets.add(target)

        candidates = []
        for rank, (name, _, _) in enumerate(_ACTIONS):
            destination = _move(position, name, size)
            if destination in opponent_bases:
                continue
            if destination in own_positions or destination in reserved_destinations:
                continue
            score = float(_distance(destination, target, size))
            if destination in opponent_workers:
                score += 12.0 + cargo * 0.1
            if destination in danger:
                score += (5.0 + cargo * 0.15) if cargo > 0 else 0.5
            tie_break = (rank - (sum(map(ord, worker_id)) % 4)) % 4
            candidates.append((score + tie_break * 0.001, name, destination))

        stay_score = float(_distance(position, target, size)) + 0.4
        if position in danger and cargo > 0:
            stay_score += 5.0 + cargo * 0.15
        candidates.append((stay_score, None, position))
        candidates.sort(key=lambda item: item[0])
        _, chosen, destination = candidates[0]
        if chosen is not None:
            actions[worker_id] = chosen
        reserved_destinations.add(destination)

    # Build a small, replaceable workforce. Stop early enough for a new collector
    # to earn back its recruitment cost before the final scoring step.
    desired_collectors = 4
    recruit_cost = configuration["recCollectorCost"]
    available_cash = float(cash)
    if step < episode_steps - 80:
        for center_id, base in sorted(centers.items()):
            if len(collectors) >= desired_collectors:
                break
            if available_cash < recruit_cost:
                break
            if base in all_worker_positions:
                continue
            actions[center_id] = "RECCOLLECTOR"
            available_cash -= recruit_cost
            collectors[f"pending-{center_id}"] = [base, 0, "COLLECTOR"]

    return actions
