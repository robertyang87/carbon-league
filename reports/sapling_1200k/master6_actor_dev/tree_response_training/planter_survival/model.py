"""One-transition risk for a focal planter. The online API accepts public state and our proposed orders only."""
import math

MOVES = {"NONE": (0, 0), "UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
ROLE_FIELDS = ("on", "adjacent", "distance_two", "heading_to_cell", "closing_last", "closing_three", "low_cargo_reach")
SHARED = ("stay", "root_threat", "other_friend_reaching", "friend_adjacent", "enemy_base", "near_enemy_base",
          "own_base", "own_tree", "enemy_tree", "from_own_tree", "from_enemy_tree", "late", "own_return", "enemy_return")
NAMES = [f"{kind}_{field}" for kind in ("collector", "planter") for field in ROLE_FIELDS] + list(SHARED)


def destination(cell, cmd, size=15):
    dr, dc = MOVES[cmd or "NONE"]
    return ((cell // size + dr) % size) * size + (cell % size + dc) % size


def distance(first, second, n=15):
    dr = abs(first // n - second // n)
    dc = abs(first % n - second % n)
    return min(dr, n - dr) + min(dc, n - dc)


def features(obs, history, own_orders, uid, cmd):
    """history is ordered oldest to newest, ending at t-1. No replay or opposing orders are arguments."""
    if "full_carbon" in obs:
        raise ValueError("supply only an acting-seat observation")
    seat = obs["player"]
    if any(h["player"] != seat or h["step"] != obs["step"] - len(history) + i for i, h in enumerate(history)):
        raise ValueError("nonconsecutive or cross-seat observations")
    own, enemy = obs["players"][seat], obs["players"][1-seat]
    origin, _, role = own[2][uid]
    if role != "PLANTER":
        raise ValueError("a focal planter is required")
    dest = destination(origin, cmd)
    profile = []
    root_threat = False
    any_enemy_in_reach = False
    for role in ("COLLECTOR", "PLANTER"):
        counts = [0] * len(ROLE_FIELDS)
        for worker, (pos, cargo, kind) in enemy[2].items():
            if kind != role:
                continue
            d = distance(pos, dest)
            if d < 3:
                counts[d] += 1
            root_threat |= distance(pos, origin) <= 1
            if d > 1:
                continue
            any_enemy_in_reach = True
            counts[-1] += cargo <= 1
            if history:
                prior = history[-1]["players"][1-seat][2].get(worker)
                if prior is not None:
                    counts[4] += distance(prior[0], dest) > d
                    last = next((a for a in MOVES if destination(prior[0], a) == pos), None)
                    counts[3] += last is not None and destination(pos, last) == dest
            if len(history) >= 3:
                prior = history[-3]["players"][1-seat][2].get(worker)
                counts[5] += prior is not None and distance(prior[0], dest) > d
        profile.extend(min(3, v) for v in counts)
    own_bases = set(own[1].values())
    enemy_bases = set(enemy[1].values())
    friends = [(r[0], destination(r[0], own_orders.get(w))) for w,r in own[2].items() if w != uid]
    conflicts = sum(next_pos == dest for _,next_pos in friends)
    conflicts += any(p == dest and own_orders.get(k) in ("RECCOLLECTOR", "RECPLANTER") for k,p in own[1].items())
    own_trees = {p for p,_ in own[3].values()}
    enemy_trees = {p for p,_ in enemy[3].values()}
    own_dist = min((distance(dest,p) for p in own_bases), default=15)
    enemy_dist = min((distance(dest,p) for p in enemy_bases), default=15)
    profile += [cmd == "NONE", root_threat, min(2,conflicts), min(3,sum(distance(p,dest)<=1 for p,_ in friends)),
                dest in enemy_bases, enemy_dist <= 2, dest in own_bases, dest in own_trees, dest in enemy_trees,
                origin in own_trees, origin in enemy_trees, obs["step"] / 299., own_dist/15., enemy_dist/15.]
    return [float(v) for v in profile], bool(any_enemy_in_reach or dest in enemy_bases or conflicts)


def score(model, vector, envelope):
    if not envelope:
        return 0.
    if model["features"] != NAMES or len(model["weights"]) != len(vector):
        raise ValueError("risk model feature schema mismatch")
    logit = model["intercept"] + sum(w*x for w,x in zip(model["weights"], vector))
    return 1. / (1. + math.exp(-max(-60., min(60., logit))))


def death_risk(model, public_obs, history, own_orders, uid, cmd):
    return score(model, *features(public_obs, history, own_orders, uid, cmd))
