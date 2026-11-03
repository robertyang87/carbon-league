"""Carbon League strategy v3: mirror-symmetry mapping, route exploration,
sustainable harvesting, late-game trees, ram safety.

v20b: interception-aware escape routing. v18 called a cell "unsafe" only when a
lighter enemy collector was already within one step of it, i.e. it assumed the
enemy stood still, so a loaded collector could still be cut off two cells from
home. Here every lighter enemy is projected 1..3 steps along its greedy chase
line and the cells it can occupy before I can are weighted - more when they sit
on my remaining path home, less when the projection is far down the line.
"""
SIZE = 15
N = 225
OFF4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
OFF8 = ((-1, 1), (-1, 0), (-1, -1), (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1))
ACT = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
ACT_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")

# direction B: how far a lighter enemy is projected along its chase line
INT_STEPS = 1
INTERCEPT = True

P = {
    "leave_at": 14.0,
    "deliver_at": 22.0,
    "plant_sum": 140.0,
    "plant_step": 6,
    "plant_stop": 280,
    "back_penalty": 1.5,
    "col_target": 7,
    "pla_target": 3,
    "hunt_range": 4,
    "hunt_min_cargo": 12,
    "harvest_step": 60,
    "early_leave": 15.0,
}

ST = {}


def tor_dist(a, b):
    r1, c1 = divmod(a, SIZE)
    r2, c2 = divmod(b, SIZE)
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    return min(dr, SIZE - dr) + min(dc, SIZE - dc)


def nb(pos, off):
    r, c = divmod(pos, SIZE)
    return ((r + off[0]) % SIZE) * SIZE + (c + off[1]) % SIZE


def mirror(q):
    r, c = divmod(q, SIZE)
    return [q,
            r * SIZE + (SIZE - 1 - c),
            (SIZE - 1 - r) * SIZE + c,
            (SIZE - 1 - r) * SIZE + (SIZE - 1 - c)]


def cfg_get(c, key, default):
    try:
        if isinstance(c, dict):
            return c.get(key, default)
        return getattr(c, key, default)
    except Exception:
        return default


def build_route(me):
    cols = range(0, SIZE // 2 + 1) if me == 0 else range(SIZE // 2, SIZE)
    out = []
    for r in range(SIZE):
        cs = list(cols)
        if r % 2:
            cs.reverse()
        for c in cs:
            out.append(r * SIZE + c)
    return out


def init_state(obs, configuration):
    ST.clear()
    me = obs["player"]
    ST["mem"] = [-1.0] * N
    ST["seen"] = [-1] * N
    ST["target"] = {}
    ST["prev"] = {}
    ST["cursor"] = {}
    ST["route"] = build_route(me)
    ST["route_pos"] = {q: i for i, q in enumerate(ST["route"])}
    ST["plant"] = cfg_get(configuration, "plantCost", 20)
    ST["infl"] = cfg_get(configuration, "plantCostInflationRatio", 5)
    ST["base"] = cfg_get(configuration, "plantCostInflationBase", 1.235)
    ST["seize"] = cfg_get(configuration, "seizeCost", 20)
    ST["regen"] = cfg_get(configuration, "regenRate", 0.03)
    ST["life"] = cfg_get(configuration, "treeLifespan", 50)
    ST["limit"] = cfg_get(configuration, "workerLimit", 10)
    ST["last"] = cfg_get(configuration, "episodeSteps", 300) - 2


def update_memory(obs, step, me):
    mem = ST["mem"]
    seen = ST["seen"]
    near_tree = set()
    tree_cells = set()
    for p in (0, 1):
        for tid, t in obs["players"][p][3].items():
            pos = t[0]
            tree_cells.add(pos)
            near_tree.add(pos)
            for o in OFF4:
                near_tree.add(nb(pos, o))
    rate = ST["regen"]
    for q in range(N):
        if mem[q] > 0 and q not in near_tree and seen[q] != step:
            gap = step - max(seen[q], 0)
            if gap > 0:
                mem[q] = min(100.0, mem[q] * ((1 + rate) ** gap))
    vis = set()
    for wid, w in obs["players"][me][2].items():
        pos = w[0]
        vis.add(pos)
        if w[2] == "PLANTER":
            for o in OFF4:
                vis.add(nb(pos, o))
    touched = set()
    for q in vis:
        v = float(obs["carbon"][q])
        if mem[q] < 0 and v > 0:
            # first sighting: mirror cells share the untouched value
            for m in mirror(q):
                if mem[m] < 0:
                    mem[m] = v
                    seen[m] = step
        mem[q] = v
        seen[q] = step
        touched.add(q)
    for q in tree_cells:
        if mem[q] < 0:
            mem[q] = 0.0
    return mem


def blockers(obs):
    occ = {}
    for p in (0, 1):
        for wid, w in obs["players"][p][2].items():
            occ[w[0]] = (p, w[1], "C" if w[2] == "COLLECTOR" else "P")
    return occ


def pick_base(pos, bases):
    best = bases[0]
    for b in bases[1:]:
        if tor_dist(pos, b) < tor_dist(pos, best):
            best = b
    return best


def hazard(occ, me, pos, cargo, is_col, opp_bases, reserved):
    if pos in reserved:
        return None
    o = occ.get(pos)
    if o is None:
        if pos in opp_bases and cargo > 0:
            return None
        return 0.0
    side, ocargo, otype = o
    if side == me:
        return None
    if otype == "P":
        return -0.8 if is_col else None
    if not is_col:
        return None
    if cargo < ocargo:
        return -0.9
    return None


def project_chase(start, pos, steps):
    """Cells an enemy collector crosses while chasing me: each step it takes the
    neighbour that shortens its distance to `pos`, which is how a hunting
    collector actually moves. Returns the cells after 1..steps moves.
    """
    out = []
    cur = start
    for _ in range(steps):
        best, bd = None, tor_dist(cur, pos)
        for o in OFF4:
            q = nb(cur, o)
            d = tor_dist(q, pos)
            if d < bd:
                best, bd = q, d
        if best is None:
            break
        cur = best
        out.append(cur)
    return out


def intercept_risk(occ, me, pos, cargo, home):
    """cell -> weight of being cut off there by a projected enemy collector.

    Only collectors carrying LESS carbon than me can kill me (the lighter
    collector survives a collision), so only those are projected. A cell is at
    risk when the enemy can be standing next to it no later than I can be on
    it: his step k puts him on e_k, so everything within 1 of e_k is exposed
    from step k, while my own arrival takes tor_dist(pos, cell) steps. Cells on
    my remaining shortest path home weigh more - a cut between me and my base
    ends the delivery, being shadowed from behind does not - and deeper
    projections weigh less.
    """
    risk = {}
    dh0 = tor_dist(pos, home)
    for q, o in occ.items():
        if o[0] == me or o[2] != "C" or o[1] >= cargo:
            continue
        for k, e in enumerate(project_chase(q, pos, INT_STEPS), 1):
            for c in [e] + [nb(e, off) for off in OFF4]:
                d_me = tor_dist(pos, c)
                if k > d_me + 1:
                    continue    # I am past it before it can be next door
                w = 1.5 if tor_dist(c, home) == dh0 - d_me else 0.4
                w *= 1.6 / (1.0 + 0.3 * (k - 1))
                if w > risk.get(c, 0.0):
                    risk[c] = w
    return risk


def risk_pen(risk, cell):
    """Exposure of standing on `cell`: full weight on the cell, half one step
    out. A lateral step costs 1.0 of progress home, so a projected cut worth
    more than that is worth stepping around and a stale one is not.
    """
    pen = risk.get(cell, 0.0)
    for o in OFF4:
        pen += 0.5 * risk.get(nb(cell, o), 0.0)
    return pen


def step_toward(me, pos, target, cargo, is_col, occ, opp_bases, risk,
                reserved, prev=None):
    """Greedy step toward target.

    v55: the greedy score has no memory, so when the direct move is blocked
    (our own collectors crowd the base) every remaining move ties and the unit
    flips UP/DOWN forever -- measured on seed 1: a planter oscillated 64<->49
    for 249 steps and the game ended with zero trees. A small penalty on
    stepping back into the cell we just left breaks the tie.
    """
    best, best_score = None, 1e9
    for name in ACT_NAMES:
        q = nb(pos, ACT[name])
        h = hazard(occ, me, q, cargo, is_col, opp_bases, reserved)
        if h is None:
            continue
        score = tor_dist(q, target) + h + risk_pen(risk, q)
        if prev is not None and q == prev:
            score += P["back_penalty"]
        if score < best_score:
            best_score, best = score, name
    return best


def escape_step(me, pos, home, cargo, occ, opp_bases, risk, reserved):
    """Walk a delivery home: progress first, exposure second.

    When a projected intercept lands on the direct route it costs more than one
    step of distance, so a lateral step wins and the unit slips past instead of
    driving into the cut. A move that gives up more than one step of progress is
    never taken, and the caller drops the risk map once arrival is tight, so the
    unit arrives instead of surviving in place.
    """
    cands = []
    for name in ACT_NAMES:
        q = nb(pos, ACT[name])
        h = hazard(occ, me, q, cargo, True, opp_bases, reserved)
        if h is None:
            continue
        dh = tor_dist(q, home)
        cands.append((dh, dh + h + risk_pen(risk, q), name))
    if not cands:
        return None
    near = min(c[0] for c in cands)
    best, best_score = None, 1e9
    for dh, score, name in cands:
        if dh > near + 1:
            continue
        if score < best_score:
            best_score, best = score, name
    return best


def route_target(wid, pos, mem, occ, claims, step, bases, avoid):
    """Next cell to visit: richest known first, else nearest unexplored."""
    route = ST["route"]
    rp = ST["route_pos"]
    start = ST["cursor"].get(wid, rp.get(pos, 0))
    n = len(route)
    leave = P["leave_at"] if step >= P["harvest_step"] else P["early_leave"]
    best, best_score = None, 0.0
    # best known cell in one lap of the route
    for k in range(n):
        q = route[(start + k) % n]
        if q in occ or q in claims:
            continue
        v = mem[q]
        if v < leave:
            continue
        b = pick_base(q, bases)
        # route-2 valuation: carbon DELIVERED per STEP, not a raw-value heuristic.
        # Include the return trip (v5 ignored it) and the regen that happens
        # while walking there (the cell is worth more on arrival than in memory).
        d_out = tor_dist(pos, q)
        d_home = tor_dist(b, q)
        arrived = min(100.0, v * ((1.0 + ST["regen"]) ** d_out))
        # mechanism, not constants: harvesting takes 30% of the CURRENT cell
        # carbon per step, and the hold is capped, so the real question is
        # "how many steps until the hold is full, and how much do I get".
        want = max(6.0, P["deliver_at"]) - 0.0
        rem, got, k = arrived, 0.0, 0
        while got < want and k < 6:
            take = 0.3 * rem
            got += take
            rem -= take
            k += 1
        score = min(got, want) / (d_out + k + d_home + 2.0)
        for a in avoid:
            if tor_dist(a, q) <= 1:
                score *= 0.55
        if score > best_score:
            best_score, best = score, q
    if best is not None:
        return best
    # else: nearest unexplored cell in route order
    for k in range(n):
        q = route[(start + k) % n]
        if mem[q] >= 0 or q in occ or q in claims:
            continue
        return q
    return None


def plant_cost(obs, me):
    own = obs["players"][me][3]
    if len(own) <= 0:
        return ST["plant"]
    return ST["plant"] + ST["infl"] * (ST["base"] ** len(obs["trees"]))


def plant_hurdle(obs, me, step):
    """Neighbour-carbon hurdle for planting, scaled by the tree race.

    Trees are immune income: whoever is behind on trees compounds slower, so
    the hurdle should depend on the gap, not on a constant. Behind on trees ->
    plant at a lower neighbour sum (catch up); ahead -> demand the full hurdle
    (an extra tree also blocks regen on cells our collectors still farm).
    """
    base = plant_need(step)
    mine = len(obs["players"][me][3])
    theirs = len(obs["players"][1 - me][3])
    if mine + 2 < theirs:
        return base * 0.65
    if mine > theirs + 2:
        return base * 1.10
    return base


def plant_need(step):
    # v37: wire the P["plant_sum"] floor -- it used to be a dead knob with no
    # read site, so the early-game hurdle could not be tuned at all.
    if step < 60:
        base = 85.0
    elif step < 150:
        base = 150.0
    else:
        base = 230.0
    return max(base, P["plant_sum"])


def tree_cells_of(obs):
    out = set()
    for p in (0, 1):
        for tid, t in obs["players"][p][3].items():
            out.add(t[0])
    return out


def local_plant_ok(obs, pos, mem, cash, step, bases):
    me = obs["player"]
    if pos in bases:
        return False
    if pos in tree_cells_of(obs):
        return False
    cost = plant_cost(obs, me)
    if cash < cost:
        return False
    steps_left = ST["last"] - step
    if steps_left < 8:
        return False
    known, tot = 0, 0.0
    for o in OFF4:
        v = mem[nb(pos, o)]
        if v > 0:
            known += 1
            tot += min(100.0, v)
    if known < 2 or tot < plant_hurdle(obs, me, step):
        return False
    return cost <= 0.55 * tot


def plant_target(obs, pos, mem, occ, claims, cash, step, bases, reserved):
    me = obs["player"]
    cost = plant_cost(obs, me)
    if cash < cost:
        return None
    steps_left = ST["last"] - step
    if steps_left < 0.75 * cost / 8.0:
        return None
    tree_cells = tree_cells_of(obs)
    best, best_score = None, 0.0
    for q in range(N):
        if q in occ or q in claims or q in reserved or q in tree_cells:
            continue
        if q in bases:
            continue
        known, tot = 0, 0.0
        for o in OFF4:
            v = mem[nb(q, o)]
            if v > 0:
                known += 1
                tot += min(100.0, v)
        hurdle = plant_hurdle(obs, me, step)
        if q == pos:
            if known < 2 or tot < hurdle:
                continue
        elif known < 3 or tot < hurdle * 1.25:
            continue
        # route-2 block 3: per-cell long-horizon value. Measured earlier: one
        # tree returns about 0.85x the carbon sum of its four neighbours over
        # its life; it also blocks their regen, so that sum is the whole yield
        # (no regen bonus). Score = expected yield - price, no fudge factors.
        value = tot * 0.85 * min(1.0, steps_left / 14.0)
        score = value - cost - 5.0 * tor_dist(pos, q)
        for o in OFF4:
            if nb(q, o) in tree_cells:
                score -= 60
        for o in OFF8:
            oq = nb(q, o)
            if oq in occ and occ[oq][0] != me and occ[oq][2] == "C":
                score -= 20
        if score > best_score:
            best_score, best = score, q
    return best


def siege_target(obs, pos, mem, occ, claims, cash, step, reserved):
    me = obs["player"]
    if cash < ST["seize"]:
        return None
    opp = 1 - me
    steps_left = ST["last"] - step
    best, best_score = None, 0.0
    for tid, t in obs["players"][opp][3].items():
        q, age = t[0], t[1]
        left = ST["life"] - age
        if left < 8 or q in occ or q in claims or q in reserved:
            continue
        tot = 0.0
        for o in OFF4:
            v = mem[nb(q, o)]
            if v > 0:
                tot += min(100.0, v)
        value = tot * 0.62 * min(1.0, left / 14.0) * min(1.0, steps_left / 14.0)
        score = value - 1.1 * ST["seize"] - 3.2 * tor_dist(pos, q)
        if score > best_score:
            best_score, best = score, q
    return best


def agent(observation, configuration):
    obs = observation
    step = int(obs["step"])
    if step == 0 or "mem" not in ST:
        init_state(obs, configuration)
    me = obs["player"]
    my = obs["players"][me]
    opp = obs["players"][1 - me]
    cash = my[0]
    bases = list(my[1].values())
    opp_bases = set(opp[1].values())
    # v59: the engine will not plant on ANY base cell, but the planting
    # helpers only knew our own. A planter could therefore pick the ENEMY
    # base as "plant right here", stand on it and plant nothing for the rest
    # of the game (idle seed 1: planter frozen on cell 70 from step 40 to 180,
    # zero trees, ~200 tree income over 250 steps with 1.4-2.9k cash idle).
    plant_bases = bases + sorted(opp_bases)
    occ = blockers(obs)
    mem = update_memory(obs, step, me)
    steps_left = ST["last"] - step
    actions = {}
    claims = {}
    reserved = set()

    my_w = my[2]
    n_col = sum(1 for w in my_w.values() if w[2] == "COLLECTOR")
    n_pla = sum(1 for w in my_w.values() if w[2] == "PLANTER")

    if len(my_w) < ST["limit"]:
        cost = plant_cost(obs, me)
        # v30 change 1 -- reserve a planter slot: while there is no planter the
        # collector cap is limit-1, so the worker cap can never be filled by
        # collectors alone (v1.3 ended with 0 planters and 2035 unspent cash).
        slot_reserve = 1 if n_pla == 0 else 0
        col_cap = max(1, ST["limit"] - slot_reserve)
        # v30 change 2 -- reserve cash for planting: once a planter exists,
        # collectors only recruit when there is cash left over the planting
        # cost, because a stop on a cell plants only if cash >= cost.
        col_cash = 30 + (cost if n_pla >= 1 else 0)
        want = None
        if n_col < min(2, col_cap) and cash >= col_cash:
            want = "RECCOLLECTOR"
        elif n_pla < 2 and cash >= 30 + cost:
            want = "RECPLANTER"
        elif n_col < min(P["col_target"], col_cap) and cash >= col_cash:
            want = "RECCOLLECTOR"
        elif n_pla < P["pla_target"] and cash >= 30 + cost:
            want = "RECPLANTER"
        elif cash >= 30 + cost:
            want = "RECPLANTER"
        elif cash >= 30 and n_pla >= 1:
            want = "RECCOLLECTOR"
        if want is not None:
            best, best_score = None, -1e9
            for bid, b in my[1].items():
                if b in occ:
                    continue
                sc = 0.0
                for q in range(N):
                    v = mem[q]
                    if v >= 45:
                        sc += v / (1.0 + tor_dist(b, q))
                if sc > best_score:
                    best_score, best = sc, bid
            if best is not None:
                actions[best] = want
                reserved.add(my[1][best])

    for wid in sorted(my_w, key=lambda k: my_w[k][0]):
        w = my_w[wid]
        pos, cargo, typ = w[0], w[1], w[2]
        prev_pos = ST["prev"].get(wid)
        ST["prev"][wid] = pos
        is_col = typ == "COLLECTOR"
        threats = [(q, o[1]) for q, o in occ.items()
                   if o[0] != me and o[2] == "C" and tor_dist(q, pos) <= 2]
        avoid = [q for q, c in threats if c < cargo]
        if INTERCEPT and is_col:
            risk = intercept_risk(occ, me, pos, cargo,
                                  pick_base(pos, bases))
        else:
            risk = {q: 1.0 for q in avoid}

        if not is_col:
            if steps_left < 12:
                continue
            if local_plant_ok(obs, pos, mem, cash, step, plant_bases):
                claims[pos] = wid
                continue                          # stay and plant here
            target = ST["target"].get(wid)
            if target is not None and target in occ:
                target = None
            if target is not None and pos == target:
                claims[pos] = wid
                if plant_target(obs, pos, mem, occ, claims, cash, step, bases,
                                reserved) == pos:
                    continue                       # stay and plant
                ST["target"].pop(wid, None)
                target = None
            if target is None and plant_target(obs, pos, mem, occ, claims,
                                               cash, step, bases,
                                               reserved) == pos:
                claims[pos] = wid
                continue                           # plant right here
            if target is None:
                target = plant_target(obs, pos, mem, occ, claims, cash, step,
                                      bases, reserved)
                if target is None:
                    target = siege_target(obs, pos, mem, occ, claims, cash,
                                          step, reserved)
                if target is None:
                    # no economic target: scout the route (planter sees 5 cells)
                    target = route_target(wid, pos, mem, occ, claims, step,
                                          bases, avoid)
                    if target is None:
                        claims[pos] = wid
                        continue
                    ST["target"][wid] = target
                    ST["cursor"][wid] = ST["route_pos"].get(target, 0)
                else:
                    ST["target"][wid] = target
            claims[target] = wid
            name = step_toward(me, pos, target, cargo, is_col, occ, opp_bases,
                              risk, reserved, prev_pos)
            if name:
                actions[wid] = name
                reserved.add(nb(pos, ACT[name]))
            continue

        if cargo <= 6 and steps_left > 30:
            hunt, hd = None, P["hunt_range"] + 1
            for q, o in occ.items():
                if o[0] == me:
                    continue
                d = tor_dist(q, pos)
                if o[2] == "C" and o[1] >= P["hunt_min_cargo"] and d <= P["hunt_range"]:
                    if d < hd:
                        hunt, hd = q, d
                elif o[2] == "P" and d <= P["hunt_range"] and d < hd:
                # an enemy planter is the upstream of their income: every tree
                # it drops is ~0.85x of four cells' carbon, compounding.
                # Hunt it at the same reach as a loaded collector.
                    hunt, hd = q, d
            if hunt is not None:
                claims[hunt] = wid
                name = step_toward(me, pos, hunt, cargo, is_col, occ,
                                   opp_bases, risk, reserved, prev_pos)
                if name:
                    actions[wid] = name
                    reserved.add(nb(pos, ACT[name]))
                continue
        home = pick_base(pos, bases)
        # v64: the two base cells are the only deposit points and every loaded
        # collector aims at the nearest one, so on seed 14 the base was
        # occupied/reserved on almost every step and 5 of 7 loaded collectors
        # orbited at distance 1-2 with cargo 24-42 and never banked a trip.
        # When the nearest base is taken this step, walk to the other one
        # instead of queueing behind it.
        if len(bases) > 1 and (home in reserved or home in occ):
            free = [b for b in bases if b != home and b not in reserved
                    and b not in occ]
            if free:
                home = min(free, key=lambda b: tor_dist(pos, b))
        d_home = tor_dist(pos, home)
        here = mem[pos]
        endgame = steps_left <= d_home + 2 or steps_left <= 22
        leave_at = P["leave_at"] if step >= P["harvest_step"] else P["early_leave"]
        drained = here < leave_at
        deliver = (cargo >= P["deliver_at"] or endgame or
                   (d_home <= 2 and cargo >= 20) or
                   (drained and cargo >= 30) or
                   (len(threats) and cargo >= 25))
        if deliver:
            ST["target"].pop(wid, None)
            claims[pos] = wid
            if d_home == 0:
                continue
            # arrival beats survival: once the trip home is tight, drop the
            # detour and walk it straight in
            use = risk if steps_left > d_home + 2 else {}
            name = escape_step(me, pos, home, cargo, occ, opp_bases, use,
                               reserved)
            if name:
                actions[wid] = name
                reserved.add(nb(pos, ACT[name]))
            continue
        if not drained and pos not in bases:
            # standing on a cell worth harvesting: stay put
            ST["target"].pop(wid, None)
            claims[pos] = wid
            continue
        target = ST["target"].get(wid)
        if target is not None and target in occ:
            target = None
        if target is None or mem[target] < leave_at:
            target = route_target(wid, pos, mem, occ, claims, step, bases,
                                  avoid)
            if target is not None:
                ST["target"][wid] = target
                ST["cursor"][wid] = ST["route_pos"].get(target, 0)
        if target is None:
            claims[pos] = wid
            continue
        claims[target] = wid
        if pos == target:
            continue
        name = step_toward(me, pos, target, cargo, is_col, occ, opp_bases,
                          risk, reserved)
        if name:
            actions[wid] = name
            reserved.add(nb(pos, ACT[name]))
    return actions
