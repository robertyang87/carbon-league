"""Conditional location of a visible loaded enemy at t+3 or t+5, in physical coordinates.

The last class (-1) combines death, intervening unloading, and final load below 100.
Input contains only the observer's public root and up to three past physical movements.
Each horizon has its own shared coefficient vector. Rows fit into [-1,1].
"""
from math import exp, isfinite


NAMES = (
    'stay', 'travel_over_h', 'travel_squared', 'reduction_in_home_distance',
    'on_shortest_home_route', 'position_is_target_home', 'position_is_observer_home',
    'project_last_direction', 'project_last_three_mean', 'distance_to_recent_velocity',
    'return_when_approaching', 'return_with_heavy_load', 'return_late_in_game',
    'nearest_lighter_opponent', 'lighter_opponent_at_cell', 'lighter_opponent_within_two',
    'teammate_at_cell', 'teammates_within_two', 'enemy_planter_at_cell',
    'friendly_tree', 'enemy_tree', 'other', 'other_home_reachable', 'other_home_distance',
    'other_lighter_pressure', 'other_approach', 'other_heavy_load',
    'other_phase', 'other_idle_history',
)
_SIDE = 15
_RADIUS = (3, 5)
def _offset(a, b):
    return (b - a + 7) % _SIDE - 7


def _vector(a, b):
    return _offset(a % _SIDE, b % _SIDE), _offset(a // _SIDE, b // _SIDE)


_DIST = [[sum(map(abs, _vector(i, j))) for j in range(225)] for i in range(225)]
_BALLS = {h: [[c for c, d in enumerate(ds) if d <= h] for ds in _DIST] for h in _RADIUS}


def target_cells(obs, history, target_uid, h, cfg):
    """Return [physical cells, OTHER] and row-aligned features. Input seat is the observer.

    History runs oldest to newest. A gap or disappearance ends the velocity trace.
    An optional duplicate current frame is ignored; future/foreign-seat observations fail.
    Positions, distances and threats use root observations, with no projected joint orders.
    """
    if type(h) is not int or h not in _RADIUS or cfg['size'] != _SIDE:
        raise ValueError('expected a 15x15 board and horizon 3 or 5')
    seat, now = obs['player'], obs['step']
    if seat not in (0, 1):
        raise ValueError('invalid observer')
    own, rival = obs['players'][seat], obs['players'][1-seat]
    worker = rival[2].get(target_uid)
    if worker is None or worker[2] != 'COLLECTOR':
        raise ValueError('expected a visible enemy collector')
    origin, load = worker[:2]
    homes, foe_homes = set(rival[1].values()), set(own[1].values())
    if not homes or not foe_homes:
        raise ValueError('missing public home locations')
    seen = {}
    last_t = -1
    for frame in history:
        t = frame['step']
        if frame['player'] != seat or t > now or t < last_t or t in seen:
            raise ValueError('expected chronological same-seat public history')
        last_t = t
        if now-3 <= t < now:
            seen[t] = frame['players'][1-seat][2].get(target_uid)
    home_d = lambda cell: min(_DIST[cell][b] for b in homes)
    positions = [origin]
    for t in range(now-1, now-4, -1):
        w = seen.get(t)
        if w is None or w[2] != 'COLLECTOR' or _DIST[w[0]][positions[-1]] > 1:
            break
        positions.append(w[0])
    diffs = [_vector(b, a) for a, b in zip(positions, positions[1:])]
    n = len(diffs)
    vx, vy = diffs[0] if diffs else (0, 0)
    mx = sum(d[0] for d in diffs) / max(1, n)
    my = sum(d[1] for d in diffs) / max(1, n)
    root_home = home_d(origin)
    approach = (home_d(positions[-1])-root_home) / max(1, n)
    phase = min(1.0, now / max(1, cfg.get('episodeSteps', 300)-1))
    heavy = min(1.0, float(load) / 500.0)
    # A lighter collector now is an observable threat proxy; future post-move cargo is unknown.
    threat = [p for p, cash, role in own[2].values() if role == 'COLLECTOR' and cash < load]
    enemy_planters = {p for p, _, role in own[2].values() if role == 'PLANTER'}
    mates = [p for uid, (p, _, _) in rival[2].items() if uid != target_uid]
    trees, enemy_trees = ({p for p, age in side[3].values() if age < cfg.get('treeLifespan', 50)}
                          for side in (rival, own))
    cells = _BALLS[h][origin]
    rows = []
    for cell in cells:
        ds = _DIST[cell]
        radial = ds[origin] / h
        dh = home_d(cell)
        progress = (root_home - dh) / h
        dx, dy = _vector(origin, cell)
        near_threat = min((ds[p] for p in threat), default=15)
        rows.append([
            float(cell == origin), radial, radial * radial, progress,
            float(ds[origin] + dh == root_home), float(cell in homes), float(cell in foe_homes),
            (dx * vx + dy * vy) / h, (dx * mx + dy * my) / h,
            (abs(dx/h - mx) + abs(dy/h - my)) / 2,
            progress * approach, progress * heavy, progress * phase,
            min(near_threat, 5) / 5, float(near_threat == 0),
            min(3, sum(ds[p] <= 2 for p in threat)) / 3,
            float(cell in mates), min(3, sum(ds[p] <= 2 for p in mates)) / 3,
            float(cell in enemy_planters), float(cell in trees), float(cell in enemy_trees),
            0., 0., 0., 0., 0., 0., 0., 0.,
        ])
    root_threat = min((_DIST[origin][p] for p in threat), default=15)
    other = [0.] * (len(NAMES) - 8) + [
        1., float(root_home <= h), min(1., root_home / 10), 1-min(5, root_threat)/5,
        approach, heavy, phase, sum(d == (0, 0) for d in diffs)/max(n, 1),
    ]
    return cells + [-1], rows + [other]


def predict(weights, obs, history, target_uid, h, cfg):
    if tuple(weights['features']) != NAMES:
        raise ValueError('forecast feature order differs')
    beta = weights['horizons'][str(h)]
    if len(beta) != len(NAMES) or any(not isfinite(b) for b in beta):
        raise ValueError('forecast coefficients malformed')
    cells, rows = target_cells(obs, history, target_uid, h, cfg)
    logits = [sum(a*b for a, b in zip(row, beta)) for row in rows]
    if not all(isfinite(v) for v in logits):
        raise ValueError('nonfinite forecast score')
    high = max(logits)
    mass = [exp(v-high) for v in logits]
    total = sum(mass)
    return {cell: p/total for cell, p in zip(cells, mass)}
