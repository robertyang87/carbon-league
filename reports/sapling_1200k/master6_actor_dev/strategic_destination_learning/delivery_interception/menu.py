"""Public pre-action offers for a bounded training-only comparison of two physical team orders."""
from functools import lru_cache

ACTIONS = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")


@lru_cache(maxsize=2)
def board(n):
    move, dist = [], []
    for pos in range(n*n):
        y, x = divmod(pos, n)
        move.append(dict(zip(ACTIONS, [pos, ((y-1)%n)*n+x, y*n+(x+1)%n, ((y+1)%n)*n+x, y*n+(x-1)%n])))
        row = []
        for q in range(n*n):
            qy, qx = divmod(q, n)
            dx, dy = abs(x-qx), abs(y-qy)
            row.append(min(dx, n-dx) + min(dy, n-dy))
        dist.append(row)
    return move, dist


def threats(obs):
    """All large loaded enemies within ten moves of their closest home."""
    me = obs["player"]
    _, bases, workers, _ = obs["players"][1-me]
    n = int(len(obs["carbon"])**.5)
    if n*n != len(obs["carbon"]):
        raise ValueError("Non-square public board")
    move, dist = board(n)
    result = []
    for uid, (pos, cargo, role) in workers.items():
        if role != "COLLECTOR" or cargo < 100:
            continue
        distance = min(dist[pos][v] for v in bases.values())
        if not 2 <= distance <= 10:
            continue
        home = sorted(v for v in bases.values() if dist[pos][v] == distance)
        lane = sorted(q for q in move[pos].values() if min(dist[q][v] for v in home) == distance-1)
        result.append(dict(target_uid=uid, target_pos=pos, target_cargo=cargo, target_home_distance=distance,
                           nearest_bases=home, next_shortest_cells=lane))
    return sorted(result, key=lambda x: (-x["target_cargo"]/(1+x["target_home_distance"]), x["target_uid"]))


def menu_for(obs, parent_order, target):
    me = obs["player"]
    _, bases, own, own_trees = obs["players"][me]
    _, enemy_bases, enemy, _ = obs["players"][1-me]
    move, dist = board(int(len(obs["carbon"])**.5))
    lane = target["next_shortest_cells"]
    base_cells = set(bases.values()) | set(enemy_bases.values())
    tree_cells = {p for p, age in own_trees.values()}
    reserved = {uid: move[pos][parent_order.get(uid, "NONE")] for uid, (pos, _, _) in own.items()}
    recruited = {pos for uid, pos in bases.items() if parent_order.get(uid) in ("RECCOLLECTOR", "RECPLANTER")}
    offers = []
    for uid, (pos, cargo, role) in own.items():
        if role != "COLLECTOR" or cargo >= target["target_cargo"]:
            continue
        original = parent_order.get(uid, "NONE")
        occupied = {p for other, p in reserved.items() if other != uid} | recruited
        # Neither arm may create a friendly collision. Own original commands for all other workers remain intact.
        if reserved[uid] in occupied:
            continue
        legal = {k: dest for k, dest in move[pos].items() if dest not in occupied}
        intercepts = {k: q for k, q in legal.items() if q in lane and q not in base_cells}
        avoids = {k: q for k, q in legal.items() if q not in lane and q not in enemy_bases.values()}
        if not intercepts or not avoids:
            continue
        if original in intercepts:
            ia = "A"
            changed = min(avoids, key=lambda k: (min(dist[avoids[k]][b] for b in bases.values()),
                                                  k != "NONE", ACTIONS.index(k)))
            cell = reserved[uid]
        elif original in avoids:
            ia = "B"
            changed = min(intercepts, key=ACTIONS.index)
            cell = legal[changed]
        else:
            continue
        third = sorted(wid for wid, (p, c, r) in enemy.items() if wid != target["target_uid"] and
                       r == "COLLECTOR" and c <= cargo and cell in move[p].values())
        options = [dict(parent_order), dict(parent_order)]
        if changed == "NONE":
            options[1].pop(uid, None)
        else:
            options[1][uid] = changed
        offers.append(dict(**target, focal_uid=uid, focal_pos=pos, focal_cargo=cargo, intercept_arm=ia,
                           cell=cell, complete_next_shortest_coverage=len(lane)==1, competing_enemy_uids=third,
                           parent_action=original, alternative_action=changed,
                           holding_own_tree=pos in tree_cells, focal_home_distance=min(dist[pos][v] for v in bases.values()),
                           options=options))
    return min(offers, key=lambda v: (bool(v["competing_enemy_uids"]), v["focal_cargo"], v["focal_uid"])) if offers else None


def select(obs, parent_order):
    """Return earliest caller-selected state's highest-ranked actionable threat and the full A/B commands."""
    for target in threats(obs):
        offer = menu_for(obs, parent_order, target)
        if offer:
            return offer
    return None
