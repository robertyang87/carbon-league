"""Predict an enemy planter's next STAY when already standing on a live friendly tree.

Uses the public board before either side acts. Features and fitted coefficients are shared
between the replay trainer and the candidate. Geometry is supplied by the caller.
"""
import math

RAID_FEATURES = ("bias", "life_left", "bank", "cash_short", "defender_in_reach",
                 "other_defender_in_reach", "light_defender", "friendly_planter_in_reach",
                 "intruder_escort", "stay_is_only_safe_cell", "near_intruder_home",
                 "near_defender_home", "game_left")


def tree_raid_rows(obs, geometry, config, *, approaches=False):
    me = obs["player"]
    mine, enemy = obs["players"][me], obs["players"][1-me]
    if not mine[3] or not enemy[2]:
        return []
    ages = {int(pos): age for pos, age in mine[3].values()}
    lifespan = config.get("treeLifespan", 50)
    rows = []
    for uid, (pos, _, kind) in enemy[2].items():
        if kind != "PLANTER":
            continue
        targets = geometry.DEST[pos, 1:] if approaches else (pos,)
        for pos in targets:
            pos = int(pos)
            if pos not in ages or ages[pos] >= lifespan:
                continue
            life = min(lifespan - ages[pos], config.get("episodeSteps", 300)-1-obs["step"])
            if life <= 0:
                continue
            arrivals = set(geometry.DEST[pos].tolist())
            nearby = [cargo for cell, cargo, role in mine[2].values()
                      if role == "COLLECTOR" and cell in arrivals]
            covered = set()
            for cell, _, role in mine[2].values():
                if role == "COLLECTOR" and geometry.DIST[pos, cell] <= 2:
                    covered.update(geometry.DEST[cell].tolist())
            features = (1, life / lifespan, math.log1p(max(0, enemy[0])) / 8,
                        float(enemy[0] < config.get("seizeCost", 20)), min(len(nearby),1),
                        min(max(len(nearby)-1,0),1), any(c <= 5 for c in nearby),
                        any(role == "PLANTER" and cell in arrivals for cell, _, role in mine[2].values()),
                        any(role == "COLLECTOR" and cell in arrivals for cell, _, role in enemy[2].values()),
                        all(cell in covered for cell in arrivals if cell != pos),
                        min(geometry.DIST[pos, cell] for cell in enemy[1].values()) <= 2,
                        min(geometry.DIST[pos, cell] for cell in mine[1].values()) <= 2,
                        (config.get("episodeSteps", 300)-1-obs["step"])/299)
            rows.append((uid, pos, life, features))
    return rows
