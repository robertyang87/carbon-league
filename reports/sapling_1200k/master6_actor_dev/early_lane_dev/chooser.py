"""Physical torus, two forecast dates, one early staging move. No opponent orders or future inputs."""

from math import isfinite

MOVES = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")
SIDE = 15
DEST = [
    [(i % 15 + dx) % 15 + 15 * ((i // 15 + dy) % 15) for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))]
    for i in range(225)
]
DIST = [
    [
        min(abs(i % 15 - j % 15), 15 - abs(i % 15 - j % 15))
        + min(abs(i // 15 - j // 15), 15 - abs(i // 15 - j // 15))
        for j in range(225)
    ]
    for i in range(225)
]


def by_cell(forecast, horizon):
    raw = forecast.get(horizon, forecast.get(str(horizon), {}))
    masses = {}
    total = 0.0
    for cell, p in raw.items():
        p = float(p)
        if not isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid probability")
        total += p
        if str(cell).upper() == "OTHER" or str(cell) == "-1":
            continue
        cell = int(cell)
        if cell not in range(225) or cell in masses:
            raise ValueError("invalid forecast cell")
        masses[cell] = p
    if total > 1 + 1e-6:
        raise ValueError("forecast mass exceeds one")
    return masses


def own_commits(own, enemy, cmd, cfg):
    occupied = {uid: DEST[pos][MOVES.index(cmd.get(uid) or "NONE")] for uid, (pos, _, _) in own[2].items()}
    cash, free = own[0], cfg["workerLimit"] - len(own[2])
    for uid, cell in own[1].items():
        order = cmd.get(uid)
        if order in ("RECCOLLECTOR", "RECPLANTER"):
            role = order[3:]
            n = sum(w[2] == role for side in (own, enemy) for w in side[2].values())
            price = cfg["rec" + role.title() + "Cost"] + n * cfg["rec" + role.title() + "IncreaseCost"]
            if free > 0 and cash >= price:
                occupied[uid] = cell
                cash -= price
                free -= 1
    return occupied


def coverage(cell, x, y, p, q):
    # The runner can reach x by t+2, wait or proceed, and reach its neighbor by t+4.
    first = DIST[cell][x] <= 1
    second = DIST[cell][y] <= 3
    return 0.5 * (p * first + q * second), [c for (c, ok) in ((x, first), (y, second)) if ok]


def choose(obs, own_commands, cfg, forecasts):
    """Forecasts map each enemy worker ID to {3:{cell:p},5:{cell:p}}, retaining OTHER mass."""
    empty = dict(candidate=None, triggered=False, commands=dict(own_commands))
    s = obs["player"]
    if cfg["size"] != 15 or s not in (0, 1):
        raise ValueError("wrong physical observation")
    if obs["step"] + 5 > cfg["episodeSteps"] - 1:
        return empty
    own, enemy = obs["players"][s], obs["players"][1 - s]
    home = set(own[1].values())
    away = set(enemy[1].values())
    if not home or not away:
        return empty
    blocks = own_commits(own, enemy, own_commands, cfg)
    tree_cells = {pos for pl in (own, enemy) for pos, age in pl[3].values() if age < cfg["treeLifespan"]}
    rate = max(
        0.0,
        cfg["initialCollectRate"]
        - sum(w[2] == "COLLECTOR" for w in own[2].values()) * cfg["collectDecreaseRate"],
    )
    best = None
    for target, (origin, cargo, role) in sorted(enemy[2].items()):
        if role != "COLLECTOR" or cargo < 120 or target not in forecasts:
            continue
        d_home = min(DIST[origin][b] for b in away)
        if d_home < 3:
            continue
        closest = [b for b in away if DIST[origin][b] == d_home]
        p3, p5 = (by_cell(forecasts[target], h) for h in (3, 5))
        edges = []
        for x, p in p3.items():
            if p < 0.05 or x in home | away:
                continue
            for y in DEST[x][1:]:
                q = p5.get(y, 0.0)
                if q < 0.05 or y in home | away:
                    continue
                if any(
                    DIST[origin][x] + DIST[x][b] == d_home
                    and DIST[x][b] == DIST[y][b] + 1
                    and DIST[origin][y] + DIST[y][b] == d_home
                    for b in closest
                ):
                    edges.append((x, y, p, q))
        if not edges:
            continue
        for uid, (start, load, job) in sorted(own[2].items()):
            before = own_commands.get(uid) or "NONE"
            anchor = blocks[uid]
            if job != "COLLECTOR" or load > 35 or (anchor in home and load > 0):
                continue
            occupied = {pos for key, pos in blocks.items() if key != uid}
            other_light = [
                pos
                for other, (pos, c, k) in enemy[2].items()
                if other != target and k == "COLLECTOR" and c * (1 - cfg["moveCost"]) <= load
            ]
            old_edge = max(edges, key=lambda edge: coverage(anchor, *edge)[0])
            old_best, old_reach_best = coverage(anchor, *old_edge)
            for move, cell in zip(MOVES, DEST[start]):
                if (
                    move == before
                    or cell in occupied | home | away
                    or any(DIST[cell][pos] <= 1 for pos in other_light)
                ):
                    continue
                # No unobserved carbon is used. Abstain when the actual harvest forgone is unknown.
                if before == "NONE" and start not in tree_cells | home | away:
                    carbon = obs["carbon"][start]
                    if carbon is None or carbon < 0:
                        continue
                    harvest = carbon * rate
                else:
                    harvest = 0.0
                extra = max(0, min(DIST[cell][b] for b in home) - min(DIST[anchor][b] for b in home))
                cost = harvest + extra * (15 + 0.15 * load) + 2
                for x, y, p, q in edges:
                    score_alt, reach = coverage(cell, x, y, p, q)
                    if len(reach) != 2:
                        continue
                    _, old_reach = coverage(anchor, x, y, p, q)
                    delta = score_alt - old_best
                    value = cargo * delta - cost
                    row = dict(
                        target_uid=target,
                        own_uid=uid,
                        old=before,
                        order=move,
                        h=[3, 5],
                        cells=[x, y],
                        reachable_v1=old_reach,
                        reachable_alt=reach,
                        own_origin=start,
                        own_cargo=load,
                        own_home_distance=min(DIST[start][b] for b in home),
                        target_origin=origin,
                        target_cargo=cargo,
                        target_home_distance=d_home,
                        old_cell=anchor,
                        cell=cell,
                        latest_arrival=[2, 4],
                        earliest_arrival=[1 + DIST[cell][x], 2 + DIST[cell][x]],
                        p_route=[p, q],
                        score_alt=score_alt,
                        score_v1=old_best,
                        old_best_cells=list(old_edge[:2]) if old_reach_best else [],
                        old_best_reachable=old_reach_best,
                        p_old_best=list(old_edge[2:]) if old_reach_best else [],
                        coverage_gain=delta,
                        harvest_opportunity_cost=harvest,
                        extra_home_steps=extra,
                        estimated_cost=cost,
                        score=value,
                        triggered=delta >= 0.05 and value >= 8,
                    )
                    rank = (row["triggered"], value, delta)
                    if best is None or rank > best[0]:
                        best = rank, row
    if best is None:
        return empty
    row = best[1]
    orders = dict(own_commands)
    if row["triggered"]:
        if row["order"] == "NONE":
            orders.pop(row["own_uid"], None)
        else:
            orders[row["own_uid"]] = row["order"]
    return dict(candidate=row, triggered=row["triggered"], commands=orders)
