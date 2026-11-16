"""opp_collision_guard -- a base pack plus a one-step "do not walk into a 1-for-1 trade" filter (diagnostic).

Ladder forensics: 92% of our worker deaths are mutual annihilations of tie-class units (planter vs
planter, empty collector vs empty collector). For each own tie-class unit, if the destination of
the base pack's action is a cell an enemy unit of the same tie class could also enter this step
(its own cell or a 4-neighbour), the action is replaced by the first safe alternative among
[stay, the other moves] -- "safe" = not enterable by such an enemy and not the destination of
another own unit. With probability 1-GUARD_P the base action is kept (GUARD_P default 1.0).
GUARD_ROLES: "both" (default), "planter", "collector". GUARD_BASE: base pack relative to code/.
"""
import importlib.util
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_N = 15
_BASE = None
_RNG = {}
_MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
STATS = {"units": 0, "threatened": 0, "changed": 0}


def _load():
    global _BASE
    if _BASE is None:
        path = os.path.join(_HERE, "..", os.environ.get("GUARD_BASE", "packs/echo4_int8.py"))
        spec = importlib.util.spec_from_file_location(f"guard_base_{os.getpid()}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BASE = mod.agent
    return _BASE


def _mv(pos, act):
    if act not in _MOVES:
        return pos
    r, c = divmod(pos, _N)
    dr, dc = _MOVES[act]
    return ((r + dr) % _N) * _N + (c + dc) % _N


def agent(obs, config):
    acts = dict(_load()(obs, config) or {})
    me = obs["player"]
    if obs["step"] == 0 or me not in _RNG:
        _RNG[me] = random.Random(7919 * me + 23)
    rng = _RNG[me]
    p_use = float(os.environ.get("GUARD_P", "1.0")); roles = os.environ.get("GUARD_ROLES", "both")
    mine = obs["players"][me][2]; theirs = obs["players"][1 - me][2]
    reach = {"PLANTER": set(), "COLLECTOR": set()}
    for pos, carbon, role in theirs.values():
        if role == "PLANTER" or carbon == 0:
            reach[role].add(pos)
            for a in _MOVES:
                reach[role].add(_mv(pos, a))
    taken = {}
    for uid, (pos, carbon, role) in mine.items():
        taken[uid] = _mv(pos, acts.get(uid))
    for uid, (pos, carbon, role) in mine.items():
        if role == "COLLECTOR" and carbon != 0:
            continue
        if roles != "both" and roles.upper() != role:
            continue
        STATS["units"] += 1
        if taken[uid] not in reach[role]:
            continue
        STATS["threatened"] += 1
        if rng.random() >= p_use:
            continue
        others = {d for u, d in taken.items() if u != uid}
        for alt in [None] + [a for a in _MOVES if a != acts.get(uid)]:
            d = _mv(pos, alt)
            if d not in reach[role] and d not in others:
                if alt is None:
                    acts.pop(uid, None)
                else:
                    acts[uid] = alt
                taken[uid] = d
                STATS["changed"] += 1
                break
    return acts
