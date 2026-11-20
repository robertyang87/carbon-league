"""Read a planting decision from the planter's own public observation."""

import math

REFERENCE = ("quoted_price", "neighbor_carbon", "remaining_life", "nearest_enemy_planter")
FEATURES = REFERENCE + (
    "weakest_neighbor", "richest_neighbor", "competing_trees_per_neighbor",
    "free_richness", "second_enemy_planter", "enemy_planters_within_five",
    "nearest_enemy_collector", "nearby_enemy_collectors", "nearest_own_other_planter",
    "nearest_own_collector", "nearby_own_collectors", "nearby_other_own_trees",
    "nearby_other_enemy_trees", "own_base_distance", "enemy_base_distance",
    "enemy_closing_three", "enemy_seen_three", "enemy_closing_five", "enemy_seen_five",
)


def distance(a, b, size):
    ay, ax = divmod(a, size)
    by, bx = divmod(b, size)
    dx, dy = abs(ax - bx), abs(ay - by)
    return min(dx, size - dx) + min(dy, size - dy)


def neighbors(cell, n):
    row, col = divmod(cell, n)
    return (
        n * ((row - 1) % n) + col, n * row + (col + 1) % n,
        n * ((row + 1) % n) + col, n * row + (col - 1) % n,
    )


def quoted_price(players, seat, config):
    starting = config["plantCost"]
    if len(players[seat][3]) <= config["playerTreeProtectiveNumber"]:
        return starting
    alive = sum(v[1] < config["treeLifespan"] for p in players for v in p[3].values())
    return starting + config["plantCostInflationRatio"] * config["plantCostInflationBase"] ** alive


def features(public, planter, config, historical_players):
    """historical_players maps positive lag to that time's public board rosters."""
    seat, step = public["player"], public["step"]
    ps, size = public["players"], config["size"]
    own, enemy = ps[seat], ps[1-seat]
    spot, _, role = own[2][planter]
    if role != "PLANTER":
        raise ValueError("The planting worker must already be present in its sending seat")
    around = neighbors(spot, size)
    # Every cardinal neighbor of this planter is visible in its own observation.
    carbon = [float(public["carbon"][cell]) for cell in around]
    if any(c < 0 or not math.isfinite(c) for c in carbon):
        raise ValueError("Unavailable neighbor carbon")
    all_trees = {v[0] for p in ps for v in p[3].values()}
    competitors = [sum(q in all_trees for q in neighbors(cell,size)) for cell in around]
    def ordered(player, kind):
        return sorted((distance(spot,w[0],size), uid) for uid,w in player[2].items()
                      if w[2] == kind and uid != planter)
    def nearest(ordered_workers, index=0):
        return ordered_workers[index][0] if index < len(ordered_workers) else size
    def close_count(workers, radius):
        return sum(d <= radius for d, _ in workers)
    def bases(player):
        return min((distance(spot,pos,size) for pos in player[1].values()), default=size)
    ep, ec = ordered(enemy,"PLANTER"), ordered(enemy,"COLLECTOR")
    fp, fc = ordered(own,"PLANTER"), ordered(own,"COLLECTOR")
    life=min(config["treeLifespan"],max(0, config["episodeSteps"]-1-step))
    out = [
        math.log1p(quoted_price(ps,seat,config)), sum(carbon)/100, life/50, nearest(ep)/size,
        min(carbon)/100, max(carbon)/100, sum(competitors)/4,
        sum(c/(1+n) for n,c in zip(competitors,carbon))/100,
        nearest(ep,1)/size, close_count(ep,5), nearest(ec)/size, close_count(ec,3),
        nearest(fp)/size, nearest(fc)/size, close_count(fc,3),
        sum(distance(spot,v[0],size)<=4 for v in own[3].values()),
        sum(distance(spot,v[0],size)<=4 for v in enemy[3].values()),
        bases(own)/size,bases(enemy)/size,
    ]
    for lag in (3,5):
        # Follow the same nearest enemy planter, including if another was closer previously.
        past = historical_players.get(lag)
        tracked = past[1-seat][2].get(ep[0][1]) if past is not None and ep else None
        out.extend([(distance(spot,tracked[0],size) - nearest(ep))/lag if tracked else 0.,
                    float(tracked is not None)])
    if len(out)!=len(FEATURES) or not all(math.isfinite(x) for x in out):
        raise ValueError("Invalid physical input")
    return out


def predict(feature_row, head):
    z = head["intercept"] + sum(
        w*(feature_row[j]-mu)/sd for j,(w,mu,sd) in
        enumerate(zip(head["weights"],head["feature_mean"],head["feature_scale"]))
    )
    return 1/(1+math.exp(-max(-50.,min(50.,z))))
