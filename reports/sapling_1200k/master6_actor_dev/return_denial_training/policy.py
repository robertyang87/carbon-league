"""Public enemy return routes and single-collector alternatives in physical board coordinates."""
from collections import Counter
from math import isfinite


MOVES = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")
_DELTAS = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))
_SIZE = 15
_MAX_DISTANCE = 14
_DEST = tuple(tuple((p % _SIZE + dx) % _SIZE + _SIZE * ((p // _SIZE + dy) % _SIZE)
                    for dx, dy in _DELTAS) for p in range(_SIZE * _SIZE))


def _distance(p, q):
    dx, dy = abs(p % _SIZE - q % _SIZE), abs(p // _SIZE - q // _SIZE)
    return min(dx, _SIZE-dx) + min(dy, _SIZE-dy)


_DIST = tuple(tuple(_distance(p,q) for q in range(_SIZE*_SIZE)) for p in range(_SIZE*_SIZE))
NAMES = tuple("action_" + move.lower() for move in MOVES) + (
    "distance_closer_to_own_base", "homeward_when_carrying", "repeats_last_direction",
    "extends_two_aligned_moves", "reverses_last_direction", "changes_lateral_direction",
    "nearest_hostile_lighter_or_equal_collector", "nearest_hostile_planter",
    "nearest_supporting_collector", "nearest_supporting_planter", "own_teammate_on_cell",
    "hostile_committed_on_cell", "hostile_committed_lighter_or_equal_on_cell",
    "hostile_committed_lighter_or_equal_adjacent", "own_base_on_cell", "hostile_base_on_cell",
    "own_tree_on_cell", "hostile_tree_on_cell", "nearby_hostile_tree_density",
)


def _nearest(cell, positions):
    return min((_DIST[cell][p] for p in positions), default=_MAX_DISTANCE)


def _direction(a, b):
    if a is None or b is None:
        return None
    try:
        return _DEST[a].index(b)
    except ValueError:
        # No attempt to infer a move across a disappeared worker or a missing frame.
        return None


def _public_history(history, step, seat):
    previous = {}
    for h in history:
        t = h["step"]
        if h["player"] != seat or t > step or t in previous:
            raise ValueError("mixed seats, duplicate turns or future observations")
        if t < step:
            previous[t] = h["players"][1-seat]
    return previous


def _commitments(own, enemy, orders, cfg):
    movers = {}
    blocks = []
    fuel = cfg["moveCost"]
    for uid, (origin, cargo, role) in own[2].items():
        command = orders.get(uid) or "NONE"
        if command not in MOVES:
            raise ValueError("unsupported worker command")
        idx = MOVES.index(command)
        movers[uid] = (origin, float(cargo), role, idx)
        blocks.append((uid, _DEST[origin][idx], role, cargo * (1-fuel) if idx else cargo))
    counts = Counter(w[2] for side in (own, enemy) for w in side[2].values())
    cash, slots = own[0], cfg["workerLimit"]-len(own[2])
    for uid, cell in own[1].items():
        command = orders.get(uid)
        if command not in ("RECCOLLECTOR", "RECPLANTER"):
            continue
        role = command[3:]
        name = role.title()
        price = cfg["rec"+name+"Cost"] + counts[role]*cfg["rec"+name+"IncreaseCost"]
        if cash >= price and slots > 0:
            blocks.append((uid,cell,role,0.0))
            cash -= price
            slots -= 1
    return movers, blocks


def _open_cells(cells, loads, commitments):
    # A tied lightest collector denies safe passage even when neither side gets paid.
    return [cell for cell,load in zip(cells, loads)
            if not any(pos==cell and role=="COLLECTOR" and cargo<=load
                       for _,pos,role,cargo in commitments)]


def _action_features(target, own, enemy, commits, loads, previous, t, life):
    origin, cargo, _ = enemy[2][target]
    homes = tuple(enemy[1].values())
    enemy_team = [(p,c,k) for uid,(p,c,k) in enemy[2].items() if uid!=target]
    base_dist = _nearest(origin,homes)
    locations = []
    for k in (t-2,t-1):
        old = previous.get(k)
        record = old[2].get(target) if old is not None else None
        locations.append(record[0] if record is not None else None)
    older = _direction(*locations)
    last = _direction(locations[1],origin)
    own_trees = {p for p,age in enemy[3].values() if age < life}
    hostile_trees = {p for p,age in own[3].values() if age < life}
    hostile_bases = set(own[1].values())
    features = []
    for idx, (cell, load) in enumerate(zip(_DEST[origin],loads)):
        change = base_dist - _nearest(cell, homes)
        risk = [p for p,c,k in own[2].values() if k=="COLLECTOR" and c<=load]
        pressure = [p for _,p,k,c in commits if k=="COLLECTOR" and c<=load]
        supporting_collectors = [p for p,_,k in enemy_team if k=="COLLECTOR"]
        supporting_planters = [p for p,_,k in enemy_team if k=="PLANTER"]
        lateral = idx not in (0,last) and last not in (None,0)
        reverse = lateral and (idx-last)%4==2
        x = [float(idx==k) for k in range(5)] + [
            float(change), change * min(cargo,800.0)/800.0, float(last==idx),
            float(older==last==idx and idx!=0), float(reverse), float(lateral and not reverse),
            _nearest(cell,risk)/_MAX_DISTANCE,
            _nearest(cell,[p for p,_,k in own[2].values() if k=="PLANTER"])/_MAX_DISTANCE,
            _nearest(cell,supporting_collectors)/_MAX_DISTANCE,
            _nearest(cell,supporting_planters)/_MAX_DISTANCE,
            float(any(p==cell for p,_,_ in enemy_team)),
            min(3,sum(p==cell for _,p,_,_ in commits))/3.0,
            float(cell in pressure), min(3,sum(_DIST[cell][p]==1 for p in pressure))/3.0,
            float(cell in homes), float(cell in hostile_bases), float(cell in own_trees),
            float(cell in hostile_trees),
            min(9,sum(min(abs(cell%15-p%15),15-abs(cell%15-p%15))<=1 and
                      min(abs(cell//15-p//15),15-abs(cell//15-p//15))<=1 for p in hostile_trees))/9.0,
        ]
        if len(x)!=len(NAMES) or not all(isfinite(v) for v in x):
            raise ValueError("invalid pre-action features")
        features.append(x)
    return features


def _options(cells, loads, shortest, movers, commits, fuel):
    old_open = _open_cells(cells,loads,commits)
    short_cells = {cell for cell,short in zip(cells,shortest) if short}
    options = []
    for uid in sorted(movers):
        origin, cargo, role, before = movers[uid]
        if role!="COLLECTOR":
            continue
        others = [row for row in commits if row[0]!=uid]
        occupied = {row[1] for row in others}
        for idx,cell in enumerate(_DEST[origin]):
            if idx==before or cell not in short_cells or cell in occupied:
                continue
            after_load = cargo*(1-fuel) if idx else cargo
            enemy_load = loads[cells.index(cell)]
            if after_load>=enemy_load:
                continue
            new_open = _open_cells(cells,loads,[*others,(uid,cell,role,after_load)])
            options.append(dict(own_uid=uid,old=MOVES[before],order=MOVES[idx],
                                old_cell=_DEST[origin][before],cell=cell,own_origin=origin,own_cargo=cargo,
                                own_collision_cargo=after_load,target_collision_cargo=enemy_load,
                                open_old=old_open.copy(),open_new=new_open,
                                open_shortest_old=[c for c in old_open if c in short_cells],
                                open_shortest_new=[c for c in new_open if c in short_cells]))
    return options


def public_turn(obs, history, own_commands, cfg):
    """All incoming carriers, including those with zero available alternative.

    All decisions and feature columns use physical board coordinates. A history element
    is a public observation of the same seat; the inclusion of the current frame is
    harmless. Command input belongs solely to the frozen friendly baseline.
    """
    t, seat = obs["step"], obs["player"]
    if cfg["size"]!=_SIZE or seat not in (0,1):
        raise ValueError("unsupported geometry or acting seat")
    if t<3 or t>=cfg["episodeSteps"]-1:
        return []
    previous = _public_history(history,t,seat)
    if t-3 not in previous:
        return []
    own, enemy = obs["players"][seat], obs["players"][1-seat]
    old_enemy = previous[t-3]
    homes = tuple(enemy[1].values())
    if not homes or not old_enemy[1]:
        raise ValueError("missing enemy home")
    movers, commits = _commitments(own,enemy,own_commands,cfg)
    fuel = cfg["moveCost"]
    population = []
    for target in sorted(enemy[2]):
        origin,cargo,role = enemy[2][target]
        if role!="COLLECTOR" or cargo<100:
            continue
        dist = _nearest(origin,homes)
        earlier = old_enemy[2].get(target)
        if dist<3 or earlier is None or earlier[2]!="COLLECTOR" or _nearest(earlier[0],old_enemy[1].values())<=dist:
            continue
        cells = _DEST[origin]
        shortest = [_nearest(c,homes)==dist-1 for c in cells]
        loads = [float(cargo)]+[cargo*(1-fuel)]*4
        population.append(dict(target=target,origin=origin,cargo=float(cargo),return_distance=dist,
                               action_cells=list(cells),shortest=shortest,
                               features=_action_features(target,own,enemy,commits,loads,previous,t,cfg["treeLifespan"]),
                               options=_options(cells,loads,shortest,movers,commits,fuel)))
    return population
