"""Estimate the next three collection turns at free local worksites on visible information."""
import numpy as np


def quarry_values(F, x, aux, slot, freshness):
    pos = int(x["tok_pos"][slot])
    cargo = float(x["tok_cargo"][slot])
    dist = int(aux["dob"][pos])
    if (int(x["tok_kind"][slot]) != 0 or cargo > 100 or x["left"] < dist+9 or
            x["cash"] < 45 or dist > 7 or not x["tok_mask"][slot]):
        return None
    dest = F.DEST[pos]
    mine = np.array(x["tok_pos"][:10])[x["tok_mask"][:10]]
    other = mine[mine != pos]
    # Never ask another worker to clear a tile. For unoccupied targets contested
    # by our own collectors at the same range, preserve the actor's allocation.
    collectors = np.array(x["tok_pos"][:10])[x["tok_mask"][:10] & (x["tok_kind"][:10] == 0)]
    other_c = collectors[collectors != pos]
    shared = (F.DIST[dest][:,other_c].min(axis=1) <= 1 if len(other_c) else np.zeros(5, bool))
    shared[0] = False
    occupied = np.isin(dest,other)
    rich = aux["est"][dest] * 100.
    # An enemy one step from a neighbour could intercept the second collection
    # turn. A friendly base can bank cargo immediately but cannot be mined.
    light_foe = aux["en_min"] <= cargo + 55.
    unsafe = light_foe[F.DEST[dest]].any(axis=1)
    harvestable = (~aux["own_tree"][dest] & ~aux["en_tree"][dest] & ~aux["own_base"][dest] &
                   ~aux["en_base"][dest] & ~unsafe & (aux["toll"][dest] == 0) & ~occupied & ~shared)
    # Alternative mines require a recent direct sighting, not inferred symmetry.
    # Current carbon is visible to our collector on every call.
    harvestable &= freshness[dest] <= 2
    if not harvestable.any():
        return None
    # Equal three-turn time end point. Stay yields c*(1-.7**3), move then
    # mine twice yields c*(1-.7**2). Only the first command is committed.
    value = rich * np.array([.657,.51,.51,.51,.51])
    # A step away from home costs both current load and the new haul time.
    value -= .055 * (cargo+value) * (aux["dob"][dest] - dist)
    value[~harvestable] = -np.inf
    return value
