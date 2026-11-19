"""Localized single-collector alternatives from the acting seat's current observation.

Physical commands and positions are used throughout. Static threat envelopes and
projected harvest are hypotheses for the value learner, not predicted enemy commands.
An alternative must also pass the frozen v1 prefix validator before simulation or execution.
"""
from math import isfinite

MOVES = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")
OFFSETS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))
KINDS = ("mine_hold", "bank_route", "evade", "ore_route")
BANDS = ((255, 270), (271, 285), (286, 295))
FEATURES = ("bias", "cargo", "remaining", "home_slack", "mine_now_delta",
            "mine_after_approach_delta", "home_progress_value_delta", "direct_deposit_delta",
            "bank_reach_delta", "safe_route_delta", "immediate_loss_risk_delta",
            "next_pressure_delta", "tree_exposure_delta", "return_slack_delta")


def step(pos, action, size):
    y, x = divmod(pos, size)
    dy, dx = OFFSETS[MOVES.index(action)]
    return ((y + dy) % size) * size + (x + dx) % size


def distance(a, b, n):
    ay, ax = divmod(a, n)
    by, bx = divmod(b, n)
    dy, dx = abs(ay - by), abs(ax - bx)
    return min(dy, n - dy) + min(dx, n - dx)


def adjacent_tree(a, b, n):
    ay, ax = divmod(a, n)
    by, bx = divmod(b, n)
    return min(abs(ay-by), n-abs(ay-by)) <= 1 and min(abs(ax-bx), n-abs(ax-bx)) <= 1


def visible_cells(own_workers, n):
    cells = set()
    for pos, _, role in own_workers.values():
        if role == "COLLECTOR":
            cells.add(pos)
        elif role == "PLANTER":
            cells.update(step(pos, a, n) for a in MOVES[1:])
    return cells


def proposals(obs, cfg, parent_actions):
    cfg = cfg or {}
    n = cfg.get("size", 15)
    tick = obs["step"]
    left = cfg.get("episodeSteps", 300) - 1 - tick
    if n != 15 or not any(lo <= tick <= hi for lo, hi in BANDS) or left < 3:
        return []
    seat = obs["player"]
    own, enemy = obs["players"][seat], obs["players"][1-seat]
    bases, hostile_bases = set(own[1].values()), set(enemy[1].values())
    if not bases:
        return []
    mine = own[2]
    own_moves = {u: parent_actions.get(u, "NONE") for u in mine}
    if any(cmd not in MOVES for cmd in own_moves.values()):
        return []
    ends = {u:step(mine[u][0], cmd, n) for u,cmd in own_moves.items()}
    hiring = {p for b,p in own[1].items() if parent_actions.get(b) in ("RECCOLLECTOR","RECPLANTER")}
    trees = {p for player in (own, enemy) for p,_ in player[3].values()}
    covered = visible_cells(mine, n)
    enemy_trees = [p for p,age in enemy[3].values() if age < cfg.get("treeLifespan",50)]
    enemy_collectors = [(p, c) for p,c,r in enemy[2].values() if r == "COLLECTOR"]
    move_fraction = 1 - cfg.get("moveCost", 0.)
    collectors = sum(r == "COLLECTOR" for _,_,r in mine.values())
    # Post-collision collector count is unknown when we submit orders. Approximate it
    # with the public pre-action count plus this joint order's already staged recruits.
    recruits = sum(parent_actions.get(b) == "RECCOLLECTOR" for b in own[1])
    rate = max(0., cfg.get("initialCollectRate",.3) - (collectors + recruits)*cfg.get("collectDecreaseRate",0.))
    ore = {p:max(0.,obs["carbon"][p])*rate for p in covered-trees-bases-hostile_bases}
    result = []
    for uid,(pos,cargo,role) in sorted(mine.items()):
        if role != "COLLECTOR":
            continue
        original = own_moves[uid]
        blocked = hiring | {p for other,p in ends.items() if other!=uid}
        d0 = min(distance(pos,b,n) for b in bases)
        cargo_scale = min(cargo/100., 4.)
        exposure_scale = .1 + cargo_scale

        def pressure(dest, amount):
            now = False
            close = False
            for en, carrying in enemy_collectors:
                d = distance(en,dest,n)
                # Include ties, which destroy both collectors. A mover pays before collision.
                their = carrying if d == 0 else carrying * move_fraction
                now |= d <= 1 and their <= amount
                close |= d <= 2 and min(their, their*move_fraction) <= amount
            return int(now), int(close)

        def attributes(action):
            dest = step(pos,action,n)
            d = min(distance(dest,b,n) for b in bases)
            amount = cargo * (1. if action=="NONE" else move_fraction)
            risk, close = pressure(dest,amount)
            # Check a safe forward continuation along an actual shortest return path.
            # Rival positions are the common pre-action positions. They are not fixed future orders.
            downhill = [step(dest,m,n) for m in MOVES[1:]]
            downhill = [p for p in downhill if p not in hostile_bases and any(distance(p,b,n)<d for b in bases)]
            safe = sum(not pressure(p,amount*move_fraction)[1] for p in downhill)
            route = 1. if d==0 else safe/max(len(downhill),1)
            reach = float(d<=left-1)
            # Moving to ore takes one turn, mining there one more; reserve d turns for banking.
            now_ore = ore.get(dest,0.) if action=="NONE" and d<=left-1 else 0.
            next_ore = ore.get(dest,0.) if d<=left-2 else 0.
            tree_count = sum(adjacent_tree(dest,p,n) for p in enemy_trees)
            slack = max(-4,min(4,left-1-d))/4.
            feat = (now_ore/30., next_ore/30., (d0-d)*cargo_scale,
                    float(d==0)*cargo_scale, reach*cargo_scale, route*reach*exposure_scale,
                    risk*exposure_scale, close*exposure_scale, min(3,tree_count)*cargo_scale,
                    slack*exposure_scale)
            return {"dest":dest,"distance":d,"risk":risk,"close":close,"route":route,
                    "now_ore":now_ore,"next_ore":next_ore,"feat":feat}

        old = attributes(original)
        for action in MOVES:
            if action==original:
                continue
            dest = step(pos,action,n)
            if dest in blocked or dest in hostile_bases:
                continue
            new = attributes(action)
            if cargo>0 and ((old["risk"] and not new["risk"]) or
                            (not new["risk"] and old["close"] and not new["close"])):
                kind = "evade"
            elif action=="NONE" and new["now_ore"]>0:
                kind = "mine_hold"
            elif (action!="NONE" and cargo>0 and new["distance"]<d0 and new["distance"]<=left-1 and
                  (new["distance"]<old["distance"] or new["route"]>old["route"])):
                kind = "bank_route"
            elif (action!="NONE" and new["next_ore"]>max(0.,old["next_ore"]) and
                  not new["risk"]):
                kind = "ore_route"
            else:
                continue
            delta = [a-b for a,b in zip(new["feat"],old["feat"])]
            features = [1.,cargo_scale,min(left,45)/45.,max(-8,min(8,left-d0))/8.,*delta]
            now,after,progress,direct,reach,route,risk,close,tax,slack = delta
            priority = (1.15*now + .55*after + .9*progress + 1.5*direct + 1.7*reach +
                        .32*route - 1.4*risk - .48*close - .17*tax + .25*slack)
            if len(features)!=len(FEATURES) or not all(isfinite(x) for x in (*features,priority)):
                raise ValueError("nonfinite public action features")
            result.append({"uid":uid,"original":original,"action":action,"features":features,
                           "kind":kind,"priority":priority})
    return sorted(result,key=lambda p:(-p["priority"],p["uid"],MOVES.index(p["action"])))


def validated_proposals(obs,cfg,parent_actions,x,aux,features,validate):
    """Use the exact validator from late_local_residual_dev/deploy.py with original production context."""
    kept = []
    for p in proposals(obs,cfg,parent_actions):
        outcome, reason = validate(obs,cfg,parent_actions,{p["uid"]:p["action"]},x,aux,features)
        if reason == "accepted":
            if {k for k in parent_actions.keys()|outcome.keys() if parent_actions.get(k)!=outcome.get(k)} != {p["uid"]}:
                raise ValueError("validator changed another parent slot")
            kept.append(p)
    return kept
