"""Intercept an enemy planter already on an owned tree that it can seize this turn."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "intercept_response_dev"))
from forecast import _D, _DIRECTION, _TO


def intervention(obs, commands, cfg):
    seat = obs["player"]
    own, enemy = obs["players"][seat], obs["players"][1 - seat]
    if enemy[0] < cfg["seizeCost"]:
        return None
    destinations = {uid: int(_TO[w[0], _DIRECTION.index(commands.get(uid))])
                    for uid, w in own[2].items()}
    targeted = set(destinations.values())
    trees = {pos: (uid, age) for uid, (pos, age) in own[3].items()}
    options = []
    for foe, (cell, _, kind) in enemy[2].items():
        if kind != "PLANTER" or cell not in trees or cell in targeted:
            continue
        tree, age = trees[cell]
        if age >= cfg["treeLifespan"]:
            continue
        for uid, (pos, cargo, role) in own[2].items():
            if role != "COLLECTOR" or _D[pos,cell] != 1:
                continue
            if any(c == "COLLECTOR" and load <= cargo and _D[enemy_pos,cell] <= 1
                   for enemy_pos, load, c in enemy[2].values()):
                continue
            if destinations[uid] in own[1].values() and cargo > cfg["smeltCost"]:
                continue
            action = _DIRECTION[list(_TO[pos]).index(cell)]
            options.append((age, cargo, uid, foe, tree, action))
    if not options:
        return None
    age, cargo, uid, foe, tree, action = min(options)
    return dict(tree=tree, age=age, cargo=cargo, uid=uid, enemy=foe, before=commands.get(uid), after=action)
