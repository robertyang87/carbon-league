"""Public-input collector stay classifier and conservative single-command overlay."""
from math import exp, log1p

MOVES = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")
FEATURES = ("bias", "patch", "load", "load2", "distance_home", "time",
            "time_to_bank", "lighter_distance", "enemy_distance", "tax_trees",
            "friendly_near", "visible_neighbor_max", "seen_neighbors",
            "last_stayed", "last_load_gain", "own_cash", "team_size",
            "patch_times_load", "patch_times_return_slack")
THRESHOLD = .85


def dest(pos, cmd, n):
    r, c = divmod(pos, n)
    dr, dc = {"NONE": (0, 0), "UP": (-1, 0), "RIGHT": (0, 1),
              "DOWN": (1, 0), "LEFT": (0, -1)}[cmd or "NONE"]
    return (r + dr) % n * n + (c + dc) % n


def dist(a, b, n):
    ar, ac = divmod(a, n)
    br, bc = divmod(b, n)
    x, y = abs(ar - br), abs(ac - bc)
    return min(x, n - x) + min(y, n - y)


def opportunities(obs, previous, cfg):
    seat, tick, n = obs["player"], obs["step"], cfg["size"]
    own, opp = obs["players"][seat], obs["players"][1-seat]
    homes = set(own[1].values())
    other_homes = set(opp[1].values())
    trees = {p for side in (own, opp) for p, _ in side[3].values()}
    visible = set()
    for pos, _, role in own[2].values():
        if role == "COLLECTOR":
            visible.add(pos)
        else:
            visible.update(dest(pos, cmd, n) for cmd in MOVES[1:])
    earlier = previous["players"][seat][2] if previous else {}
    for uid, (pos, cargo, role) in own[2].items():
        if role != "COLLECTOR" or pos in trees or pos in (homes | other_homes):
            continue
        carbon = obs["carbon"][pos]
        home = min(dist(pos, base, n) for base in homes)
        left = cfg["episodeSteps"] - 1 - tick
        if carbon < 30 or left < home + 2 or any(dist(pos, b, n) <= 1 for b in other_homes):
            continue
        light = min((dist(pos, p, n) for p, v, k in opp[2].values()
                     if k == "COLLECTOR" and v <= cargo), default=15)
        if light <= 1:
            continue
        enemy = min((dist(pos, p, n) for p, _, _ in opp[2].values()), default=15)
        # Trees levy cargo in their eight-neighbour footprint.
        r, c = divmod(pos, n)
        ny = {((r + dr) % n) * n + ((c + dc) % n) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        tax = sum(p in ny and age < cfg["treeLifespan"] for p, age in opp[3].values())
        near = sum(other != uid and dist(pos, p, n) <= 1 for other, (p, _, _) in own[2].items())
        adj = {dest(pos, cmd, n) for cmd in MOVES[1:]} & visible
        old = earlier.get(uid)
        load = min(log1p(cargo) / 5, 1.4)
        patch = carbon / 100
        slack = min(left - home, 30) / 30
        x = (1, patch, load, load*load, home/7, tick/299, slack,
             min(light, 7)/7, min(enemy, 7)/7, tax/4, near/3,
             max((obs["carbon"][q] for q in adj), default=0)/100, len(adj)/4,
             float(bool(old and old[0] == pos)),
             max(-1, min(1, (cargo-old[1])/30)) if old else 0,
             log1p(max(0, own[0]))/9, len(own[2])/10, patch*load, patch*slack)
        yield uid, x


def score(weights, x):
    z = max(-50, min(50, sum(a*b for a,b in zip(weights, x))))
    return 1 / (1 + exp(-z))


def choose(obs, previous, cfg, base, weights):
    own = obs["players"][obs["player"]]
    homes = set(own[1].values())
    n = cfg["size"]
    options = []
    for uid, x in opportunities(obs, previous, cfg):
        cmd = base.get(uid) or "NONE"
        if cmd == "NONE":
            continue
        cell, cargo, _ = own[2][uid]
        if dest(cell, cmd, n) in homes and cargo > 0:
            continue
        if any(other != uid and dest(p, base.get(other), n) == cell for other,(p,_,_) in own[2].items()):
            continue
        probability = score(weights, x)
        if probability >= THRESHOLD:
            options.append((probability, uid))
    return max(options, default=None)
