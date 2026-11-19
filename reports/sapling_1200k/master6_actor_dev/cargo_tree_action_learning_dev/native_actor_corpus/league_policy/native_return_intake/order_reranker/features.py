"""Shared public, pre-action features for two complete physical Sapling proposals.

This file is also embedded intact in the NumPy deployment. Destinations describe
own intents prior to the simultaneous transition, not predicted next observations.
"""
import math
import numpy as np


def _order_dist(a, b, size):
    dy, dx = abs(a // size - b // size), abs(a % size - b % size)
    return min(dy, size - dy) + min(dx, size - dx)


def _order_dest(position, command, size):
    row, col = divmod(position, size)
    dy, dx = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1),
              "NONE": (0, 0)}[command]
    return ((row + dy) % size) * size + (col + dx) % size


def _order_rows(observation, focus, commands, original_logits):
    size = math.isqrt(len(observation["carbon"]))
    if size * size != len(observation["carbon"]):
        raise ValueError("Invalid square board")
    cap = max(2 * (size // 2), 1)
    distance = lambda p, q: _order_dist(p, q, size)
    nearest = lambda p, targets: min((distance(p, q) for q in targets), default=cap) / cap
    clip = lambda value, unit: min(max(float(value) / unit, 0.0), 1.0)
    own, enemy = (observation["players"][(observation["player"] + s) % 2] for s in (0, 1))
    workers, foes = own[2], enemy[2]
    home, away = list(own[1].values()), list(enemy[1].values())
    self_trees, enemy_trees = ({val[0] for val in player[3].values()} for player in (own, enemy))
    own_cargo = sum(val[1] for val in workers.values())
    enemy_cargo = sum(val[1] for val in foes.values())
    pos, cargo, role = workers[focus]
    enemy_pos = [v[0] for v in foes.values()]
    loaded_pos = [v[0] for v in foes.values() if v[1] > 0]
    allies = [v[0] for uid, v in workers.items() if uid != focus]
    context = [clip(observation["step"], 299), clip(own[0], 2000), clip(enemy[0], 2000),
               clip(own_cargo, 500), clip(enemy_cargo, 500), clip(len(workers), 10),
               clip(len(foes), 10), clip(len(self_trees), 20), clip(len(enemy_trees), 20),
               float(role == "PLANTER"), clip(cargo, 200), nearest(pos, home), nearest(pos, away)]
    rows = []
    for index, order in enumerate(commands):
        target = {uid: _order_dest(worker[0], order.get(uid, "NONE"), size)
                  for uid, worker in workers.items()}
        dest = target[focus]
        close_foes = [v for v in foes.values() if distance(dest, v[0]) <= 1]
        dangerous = sum(v[2] == "COLLECTOR" and v[1] < cargo for v in close_foes)
        n = max(len(workers), 1)
        candidate = [float(np.clip(original_logits[index] - original_logits[1 - index], -1, 1)),
                     float(dest == pos), nearest(dest, home), nearest(dest, away),
                     nearest(dest, enemy_pos), nearest(dest, loaded_pos), nearest(dest, allies),
                     clip(observation["carbon"][dest], 100), float(dest in self_trees),
                     float(dest in enemy_trees), clip(len(close_foes), 5), clip(dangerous, 5)]
        deposited = sum(workers[uid][1] for uid, dst in target.items() if dst in home)
        cargo_distance = sum(workers[uid][1] * nearest(dst, home) for uid, dst in target.items())
        team = [(len(target) - len(set(target.values()))) / n, clip(deposited, 500),
                cargo_distance / max(own_cargo, 1e-9),
                sum(workers[u][2] == "PLANTER" and d == workers[u][0] and d in enemy_trees
                    for u, d in target.items()) / n,
                sum(workers[u][1] > 0 and any(distance(d, e[0]) <= 1 for e in foes.values())
                    for u, d in target.items()) / n,
                sum(d != workers[u][0] for u, d in target.items()) / n,
                sum(d in away for d in target.values()) / n]
        rows.append(context + candidate + team)
    return np.asarray(rows, dtype=np.float64)


def _order_logit_delta(rows, weights):
    hidden = np.tanh(rows @ weights["hidden_w"] + weights["hidden_b"])
    return 0.25 * float(np.tanh((hidden[0] - hidden[1]) @ weights["readout"]))
