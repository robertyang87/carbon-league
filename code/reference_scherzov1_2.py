"""Carbon League strategy "arbor" v1 (single file, stdlib only).

Written from scratch against the original zerosum_env carbon rules
(github.com/moliqingwa/carbon_challenge, helpers.py Board.next()).
Ideas such as "tree value = carbon of the 4 neighbours" and seizing enemy trees
were informed by reading the public de-modelled juzi reference strategy on the
Carbon League site; no code was copied from it.

Main parts
  * belief map of hidden carbon: replays the known transition rules every step
    (collection by stationary collectors, tree absorption, regeneration, withering)
    and uses the 4-fold mirror symmetry of the initial map to fill unseen cells;
  * collectors: harvest-rate planner (cargo per step including the trip home),
    hunting of adjacent loaded enemy collectors / enemy planters, threat avoidance;
  * planters: plant where the 4 neighbours hold the most carbon, seize enemy trees,
    never stand still by accident (standing still plants automatically);
  * recruiting: keep a collector/planter mix, keep the base cell free.
"""
import math
import os

_DEBUG = bool(os.environ.get("CARBON_DEBUG"))
# tunable parameters (kept together so experiments can override them)
P = dict(stuck_risk=4, end_crowd=4, end_margin=2, aura_w=1.0, planner="dfs", H=5, disc=0.97, min_stay=8.0, term=1.0, ret_hi=200.0, ret_near_d=2, ret_near_q=100.0, early_t=50, early_q=40.0, grow=0, r_mode="cyc", ema_a=0.02, stick=5.0, tree_eff=1.0, cost_mult=1.0, r_scale=1.0, r_min=1.5, r_max=12.0, guard_range=3, risk=0.01, own_tree_cell=0.55, deny_bonus=1.1, want_pl=1, want_col=9,
         seize_mult=1.6, plant_margin=5.0)
if os.environ.get("CARBON_P") and os.path.basename(globals().get("__file__", "")) == "submission.py":
    import json as _json
    P.update(_json.loads(os.environ["CARBON_P"]))
MOVES = ("UP", "RIGHT", "DOWN", "LEFT")
_ST = {}


def _get(o, k, d=None):
    try:
        return o[k]
    except Exception:  # noqa: BLE001
        return getattr(o, k, d)


class Geo:
    def __init__(self, n):
        self.n = n
        n2 = n * n
        self.nb = []  # list of (move, cell)
        for i in range(n2):
            r, c = divmod(i, n)
            self.nb.append((("UP", ((r - 1) % n) * n + c), ("RIGHT", r * n + (c + 1) % n),
                            ("DOWN", ((r + 1) % n) * n + c), ("LEFT", r * n + (c - 1) % n)))
        self.n4 = [[j for _, j in self.nb[i]] for i in range(n2)]
        self.n8 = []
        for i in range(n2):
            r, c = divmod(i, n)
            self.n8.append([((r + dr) % n) * n + (c + dc) % n for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                            if dr or dc])
        self.dist = []
        for i in range(n2):
            r1, c1 = divmod(i, n)
            row = []
            for j in range(n2):
                r2, c2 = divmod(j, n)
                dr = abs(r1 - r2)
                dc = abs(c1 - c2)
                row.append(min(dr, n - dr) + min(dc, n - dc))
            self.dist.append(row)
        self.orbit = []
        for i in range(n2):
            r, c = divmod(i, n)
            self.orbit.append(sorted({r * n + c, r * n + (n - 1 - c), (n - 1 - r) * n + c,
                                      (n - 1 - r) * n + (n - 1 - c)}))


def _prior(cfg, t):
    """Expected carbon of a never-observed, untouched cell at step t."""
    n = cfg["size"]
    half = math.ceil(n / 2)
    min_cells = int(math.ceil(cfg["startingCarbon"] / max(cfg["startingCellCarbon"], 1e-9) / 4))
    ncell = max(int(math.ceil(half * half * cfg["carbonCoverage"] / 100)), min_cells)
    p = min(1.0, ncell / (half * half))
    hi = 2 * min(cfg["startingCarbon"] / 4 / ncell, cfg["startingCellCarbon"])
    hi = min(hi, cfg["startingCellCarbon"] * 2)
    g = (1 + cfg["regenRate"]) ** t
    cap = cfg["maxCellCarbon"]
    if hi * g <= cap:
        e = hi * g / 2
    else:
        a = cap / g
        e = (g * a * a / 2 + cap * (hi - a)) / hi
    return p * e


def _cfg(conf):
    keys = ["size", "episodeSteps", "startingCarbon", "carbonCoverage", "startingCellCarbon",
            "maxCellCarbon", "recCollectorCost", "recPlanterCost", "recCollectorIncreaseCost",
            "recPlanterIncreaseCost", "workerLimit", "plantCost", "plantCostInflationRatio",
            "plantCostInflationBase", "playerTreeProtectiveNumber", "co2FrmWithered", "seizeCost",
            "initialCollectRate", "collectDecreaseRate", "regenRate", "treeLifespan",
            "cellAbsorptionRate", "collectorAbsorptionRate", "smeltCost", "moveCost"]
    defaults = dict(size=15, episodeSteps=300, startingCarbon=2500, carbonCoverage=45,
                    startingCellCarbon=50, maxCellCarbon=100, recCollectorCost=30, recPlanterCost=30,
                    recCollectorIncreaseCost=0, recPlanterIncreaseCost=0, workerLimit=10, plantCost=20,
                    plantCostInflationRatio=5, plantCostInflationBase=1.235,
                    playerTreeProtectiveNumber=0, co2FrmWithered=30, seizeCost=20,
                    initialCollectRate=0.3, collectDecreaseRate=0.0, regenRate=0.03, treeLifespan=50,
                    cellAbsorptionRate=0.0375, collectorAbsorptionRate=0.1, smeltCost=0, moveCost=0)
    out = {}
    for k in keys:
        v = _get(conf, k, None)
        out[k] = defaults[k] if v is None else v
    return out


def _init_state(cfg, obs):
    n = cfg["size"]
    geo = _ST.get("geo")
    if geo is None or geo.n != n:
        geo = Geo(n)
    st = {"geo": geo, "cfg": cfg, "est": [_prior(cfg, 0)] * (n * n), "known": [False] * (n * n),
          "pristine": [True] * (n * n), "prev": None, "last_step": -1}
    for p in _get(obs, "players"):
        for pos in p[1].values():
            st["est"][pos] = 0.0
            st["known"][pos] = True
            st["pristine"][pos] = False
    return st


def _snapshot(obs):
    workers = {}
    trees = {}
    for pid, p in enumerate(_get(obs, "players")):
        for wid, (pos, carbon, typ) in p[2].items():
            workers[wid] = (pos, float(carbon), typ == "COLLECTOR", pid)
        for tid, (pos, age) in p[3].items():
            trees[tid] = (pos, age, pid)
    return workers, trees


def _update_belief(st, obs, me):
    cfg = st["cfg"]
    geo = st["geo"]
    n2 = geo.n * geo.n
    est = st["est"]
    known = st["known"]
    pristine = st["pristine"]
    step = _get(obs, "step")
    workers, trees = _snapshot(obs)
    bases = set()
    for p in _get(obs, "players"):
        bases.update(p[1].values())
    prev = st["prev"]
    if prev is not None:
        pworkers, ptrees = prev
        cur_tree_cells = {v[0] for v in trees.values()}
        new_tree_cells = {v[0] for k, v in trees.items() if k not in ptrees}
        withered = {v[0] for k, v in ptrees.items() if k not in trees and v[1] >= cfg["treeLifespan"]}
        occ = {}
        for wid, (pos, carbon, is_col, pid) in workers.items():
            moved = not (wid in pworkers and pworkers[wid][0] == pos)
            occ[pos] = (is_col, moved, pid, wid)
        rate = cfg["initialCollectRate"]
        new = list(est)
        touched = [False] * n2
        # collection by stationary collectors
        for pos, (is_col, moved, pid, wid) in occ.items():
            if is_col and not moved and pos not in cur_tree_cells and pos not in bases:
                pw = pworkers.get(wid)
                inferred = None
                if pid != me and pw is not None:
                    delta = workers[wid][1] - pw[1]
                    aura = any(trees_pos_owner != pid for (tp, _, trees_pos_owner) in trees.values()
                               if tp == pos or tp in geo.n8[pos])
                    if delta > 1e-6 and not aura:
                        inferred = delta / rate
                if inferred is not None:
                    if pristine[pos]:
                        for j in geo.orbit[pos]:
                            if j != pos and pristine[j] and not known[j]:
                                new[j] = min(cfg["maxCellCarbon"], inferred * (1 + cfg["regenRate"]))
                                known[j] = True
                    new[pos] = inferred * (1 - rate)
                    known[pos] = True
                else:
                    new[pos] = est[pos] * (1 - rate)
                touched[pos] = True
        for pos in new_tree_cells:
            new[pos] = 0.0
            touched[pos] = True
            known[pos] = True
        # tree absorption
        quad = set()
        red = [0.0] * n2
        for tid, (tpos, age, pid) in trees.items():
            for j in geo.n4[tpos]:
                quad.add(j)
                if tid in ptrees and j not in cur_tree_cells:
                    o = occ.get(j)
                    if o is None or (not o[0]) or o[1]:
                        red[j] += est[j] * cfg["cellAbsorptionRate"]
        for j in quad:
            if red[j] > 0:
                new[j] = max(0.0, new[j] - red[j])
            touched[j] = True
        for pos in withered:
            new[pos] = float(cfg["co2FrmWithered"])
            touched[pos] = True
            known[pos] = True
        # regeneration
        g = 1 + cfg["regenRate"]
        cap = cfg["maxCellCarbon"]
        for i in range(n2):
            if touched[i] or i in cur_tree_cells or i in bases:
                continue
            o = occ.get(i)
            if o is not None and not o[1]:
                touched[i] = True
                continue
            new[i] = min(cap, new[i] * g)
        for i in range(n2):
            if touched[i]:
                pristine[i] = False
        est[:] = new
        # unknown pristine cells follow the prior exactly in expectation
        pr = _prior(cfg, step)
        for i in range(n2):
            if not known[i] and pristine[i]:
                est[i] = pr
    # overwrite with what we can actually see
    carbon = _get(obs, "carbon")
    vis = set()
    for wid, (pos, c, is_col, pid) in workers.items():
        if pid != me:
            continue
        if is_col:
            vis.add(pos)
        else:
            vis.update(geo.n4[pos])
    for i in vis:
        v = float(carbon[i])
        est[i] = v
        known[i] = True
        if pristine[i]:
            for j in geo.orbit[i]:
                if j != i and pristine[j] and j not in vis:
                    est[j] = v
                    known[j] = True
    st["prev"] = (workers, trees)
    return workers, trees, bases


def agent(observation, configuration):
    try:
        return _agent(observation, configuration)
    except Exception:  # noqa: BLE001 - never crash, an empty dict is a legal action
        if _DEBUG:
            raise
        return {}


def _agent(obs, conf):
    step = _get(obs, "step")
    me = _get(obs, "player")
    if step == 0 or "geo" not in _ST or _ST.get("last_step", -1) >= step:
        cfg = _cfg(conf)
        _ST.clear()
        _ST.update(_init_state(cfg, obs))
    st = _ST
    st["last_step"] = step
    cfg = st["cfg"]
    geo = st["geo"]
    n2 = geo.n * geo.n
    D = geo.dist
    workers, trees, bases = _update_belief(st, obs, me)
    est = st["est"]
    players = _get(obs, "players")
    my = players[me]
    cash = float(my[0])
    my_bases = list(my[1].items())  # (id, pos)
    my_base_cells = [p for _, p in my_bases]
    opp = 1 - me
    opp_base_cells = list(players[opp][1].values()) if len(players) > 1 else []
    T_end = cfg["episodeSteps"] - 1
    left = T_end - step  # actions remaining including this one

    my_col = [(wid, w[0], w[1]) for wid, w in workers.items() if w[3] == me and w[2]]
    my_pl = [(wid, w[0]) for wid, w in workers.items() if w[3] == me and not w[2]]
    en_col = [(wid, w[0], w[1]) for wid, w in workers.items() if w[3] != me and w[2]]
    en_pl = [(wid, w[0]) for wid, w in workers.items() if w[3] != me and not w[2]]
    tree_at = {}
    for tid, (tpos, age, pid) in trees.items():
        tree_at[tpos] = (pid, age, tid)
    my_trees = [p for p, v in tree_at.items() if v[0] == me]
    n_alive = sum(1 for v in tree_at.values() if v[1] < cfg["treeLifespan"])
    if len(my_trees) <= cfg["playerTreeProtectiveNumber"]:
        plant_cost = cfg["plantCost"]
    else:
        plant_cost = cfg["plantCost"] + cfg["plantCostInflationRatio"] * cfg["plantCostInflationBase"] ** n_alive

    dbase = [min(D[i][b] for b in my_base_cells) if my_base_cells else 99 for i in range(n2)]

    # threat maps
    INF = 1e18
    en_min_near = [INF] * n2  # min cargo of an enemy collector able to step on the cell
    en_any_near = [False] * n2
    for wid, pos, c in en_col:
        for j in (pos, *geo.n4[pos]):
            if c < en_min_near[j]:
                en_min_near[j] = c
            en_any_near[j] = True
    en_pl_near = [False] * n2
    for wid, pos in en_pl:
        for j in (pos, *geo.n4[pos]):
            en_pl_near[j] = True
    en_tree_aura = [0] * n2
    my_tree_adj = [0] * n2
    tree_adj = [0] * n2
    for tpos, (pid, age, tid) in tree_at.items():
        for j in geo.n4[tpos]:
            tree_adj[j] += 1
            if pid == me:
                my_tree_adj[j] += 1
        if pid != me:
            en_tree_aura[tpos] += 1
            for j in geo.n8[tpos]:
                en_tree_aura[j] += 1

    claimed = set()  # cells my units will occupy next step
    actions = {}
    targets_taken = set()

    def collector_danger(x, q):
        if x in my_base_cells:
            return 0.0
        m = en_min_near[x]
        if m == INF:
            return 0.0
        if q > 0 and m < q:
            return 1000.0 + q
        if m == q:
            return 300.0 if q > 0 else 40.0
        return 0.0

    def planter_danger(x):
        d = 0.0
        if en_any_near[x]:
            d += 1000.0
        if en_pl_near[x]:
            d += 30.0
        return d

    def choose_move(pos, target, danger_fn, stay_ok, stay_bonus=0.0):
        best = None
        for mv, x in ((None, pos),) + geo.nb[pos]:
            if x in claimed:
                continue
            if mv is None and not stay_ok:
                continue
            s = -10.0 * D[x][target] - danger_fn(x)
            if mv is None:
                s += stay_bonus
            if x in opp_base_cells:
                s -= 500.0
            if best is None or s > best[0]:
                best = (s, mv, x)
        if best is None:
            # everything claimed: fall back to any non-claimed or stay
            for mv, x in geo.nb[pos]:
                if x not in claimed:
                    return mv, x
            return None, pos
        return best[1], best[2]

    # ---------------------------------------------------------------- collectors
    g = 1 + cfg["regenRate"]
    cap = cfg["maxCellCarbon"]
    rate = cfg["initialCollectRate"]
    keep = [1 - (1 - rate) ** k for k in range(0, 8)]
    # adaptive opportunity cost: what a fresh round trip from base would earn per step
    cyc = []
    for j in range(n2):
        e0 = est[j]
        if e0 < 5 or j in tree_at or j in bases:
            continue
        fj = P["own_tree_cell"] if my_tree_adj[j] else 1.0
        db = max(1, dbase[j])
        cyc.append(max(e0 * fj * keep[k] / (2 * db + k) for k in (1, 2, 3, 4)))
    cyc.sort(reverse=True)
    top = cyc[:max(1, len(my_col))]
    r_cyc = (sum(top) / len(top)) if top else P["r_min"]
    # realised deposit rate per collector-step (exponential average)
    ema = st.get("rate_ema")
    prevw = st.get("my_prev_col", {})
    dep = 0.0
    for wid, pos, q in my_col:
        pw = prevw.get(wid)
        if pw is not None and pos in my_base_cells and pw[1] > 0 and q == 0:
            dep += pw[1]
    if prevw:
        inst = dep / max(1, len(prevw))
        ema = inst if ema is None else ema + P["ema_a"] * (inst - ema)
    st["rate_ema"] = ema
    st["my_prev_col"] = {wid: (pos, q) for wid, pos, q in my_col}
    if P["r_mode"] == "ema" and ema is not None and step > 30:
        r_est = P["r_scale"] * ema
    else:
        r_est = P["r_scale"] * r_cyc
    R_STAR = min(P["r_max"], max(P["r_min"], r_est))
    RISK = P["risk"]
    col_by_id = {wid: (pos, q) for wid, pos, q in my_col}
    assigned = {}  # wid -> (mode, target)

    def step_onto(pos, x):
        for m2, y in geo.nb[pos]:
            if y == x:
                return m2
        return None

    # 1) guards / hunters: kill enemy planters, rob adjacent loaded enemy collectors
    my_tree_set = set(my_trees)
    pl_threats = []
    for ewid, epos in en_pl:
        near_tree = min((D[epos][t] for t in my_tree_set), default=99)
        stayed = st.get("en_prev_pos", {}).get(ewid) == epos
        val = 40.0 + (80.0 if near_tree <= 1 else 40.0 if near_tree <= 2 else 0.0) + (30.0 if stayed else 0.0)
        pl_threats.append((val, epos, near_tree, stayed or epos in my_tree_set))
    pl_threats.sort(reverse=True)
    for val, epos, near_tree, still in pl_threats:
        # strike: an adjacent collector steps onto a planter that is (likely) standing still
        if epos in opp_base_cells or not still:
            continue
        best = None
        for wid, pos, q in my_col:
            if wid in assigned:
                continue
            if D[pos][epos] != 1 or collector_danger(epos, q) >= 100:
                continue
            if q > 0 and left <= dbase[pos] + 2:
                continue
            if best is None or q < best[0]:
                best = (q, wid)
        if best is not None:
            assigned[best[1]] = ("hunt", epos)
    # guard: stand on / next to own trees that an enemy planter is approaching
    guarded = set()
    for val, epos, near_tree, still in pl_threats:
        if near_tree > 2:
            continue
        for t in sorted(my_tree_set, key=lambda t: D[epos][t]):
            if D[epos][t] > 2 or t in guarded:
                continue
            if any(m[0] in ("guard", "hunt") and D[col_by_id[w][0]][t] <= 1 for w, m in assigned.items()):
                guarded.add(t)
                continue
            best = None
            for wid, pos, q in my_col:
                if wid in assigned:
                    continue
                d = D[pos][t]
                if d > P["guard_range"] or (q > 0 and left <= dbase[pos] + d + 1):
                    continue
                if best is None or (d, q) < best[0]:
                    best = ((d, q), wid)
            if best is not None:
                assigned[best[1]] = ("guard", t)
                guarded.add(t)
    for wid, pos, q in my_col:
        if wid in assigned:
            continue
        bestv, bt = 0.0, None
        if left <= dbase[pos] + 4:
            continue
        for ewid, epos, ec in en_col:
            if (D[pos][epos] == 1 and ec > q + 15 and epos not in opp_base_cells
                    and st.get("en_prev_pos", {}).get(ewid) == epos and est[epos] > 10):
                # nobody else lighter than us next to the target
                ok = all(not (c2 <= q and D[p2][epos] <= 1) for w2, p2, c2 in en_col if w2 != ewid)
                if ok and ec - q > bestv:
                    bestv, bt = ec - q, epos
        if bt is not None and bestv > 25:
            assigned[wid] = ("hunt", bt)

    if P["planner"] == "dfs":
        # depth-limited route search: maximise carbon gathered over the next H steps
        H = P["H"]
        disc = [P["disc"] ** t for t in range(H + 1)]
        cellfac = [1.0] * n2
        for j in range(n2):
            f = 1.0
            if my_tree_adj[j]:
                f *= P["own_tree_cell"]
            if en_tree_aura[j]:
                f *= 0.75
            if tree_adj[j] and not my_tree_adj[j]:
                f *= P["deny_bonus"]
            cellfac[j] = f
        blocked = set(bases) | set(tree_at)
        taken = {}  # cell -> carbon left after already planned collection
        back = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        min_stay = P["min_stay"]
        for wid, pos, q in sorted(my_col, key=lambda t: -t[2]):
            if wid in assigned:
                continue
            db = dbase[pos]
            home = min(my_base_cells, key=lambda b: D[pos][b]) if my_base_cells else pos
            danger_here = collector_danger(pos, q) >= 100
            if q > 0 and (left <= db + P["end_margin"] or q >= P["ret_hi"] or (db <= P["ret_near_d"] and q >= P["ret_near_q"])
                          or (danger_here and q >= 60) or (step < P["early_t"] and q >= P["early_q"])):
                assigned[wid] = ("home", home)
                continue
            if left <= 1:
                assigned[wid] = ("idle", pos)
                continue
            best = [-1.0, None, None]
            vals = {}
            path = []

            absr = cfg["collectorAbsorptionRate"] * P["aura_w"]

            def dfs(p, t, gained, last, q=q):
                if en_tree_aura[p] and t > 0:
                    gained -= absr * en_tree_aura[p] * (q + gained)
                if t == H or t >= left:
                    v = vals.get(p, taken.get(p, est[p]))
                    tot = gained + (P["term"] * v * rate * cellfac[p] * disc[t] if t + 1 + dbase[p] <= left else 0.0)
                    if tot > best[0]:
                        best[0] = tot
                        best[1] = list(path)
                    return
                if p not in blocked and t + 1 + dbase[p] <= left - P["end_margin"] + 1:
                    v = vals.get(p, taken.get(p, est[p]))
                    if v > min_stay:
                        gch = v * rate
                        vals[p] = v - gch
                        path.append((p, True))
                        fac = cellfac[p]
                        if en_min_near[p] < q + gch:
                            fac *= 0.6
                        dfs(p, t + 1, gained + gch * fac * disc[t], None)
                        path.pop()
                        vals[p] = v
                for mv, x in geo.nb[p]:
                    if mv == last or x in opp_base_cells:
                        continue
                    if t == 0 and collector_danger(x, q) >= 100:
                        continue
                    path.append((x, False))
                    dfs(x, t + 1, gained, back[mv])
                    path.pop()

            dfs(pos, 0, 0.0, None)
            if best[1] is None or best[0] <= 0.5:
                if q > 0:
                    assigned[wid] = ("home", home)
                else:
                    # nothing worth it nearby: head for the richest known cell
                    tgt = max((j for j in range(n2) if j not in blocked),
                              key=lambda j: est[j] / (3 + D[pos][j]))
                    assigned[wid] = ("harvest", tgt)
                continue
            first = best[1][0][0]
            for c, stayed in best[1]:
                if stayed:
                    v = taken.get(c, est[c])
                    taken[c] = v * (1 - rate)
            assigned[wid] = ("path", first)
    # 2) harvest / return planning (value minus opportunity cost of time)
    prev_tgt = st.get("col_tgt", {})
    options = []
    for wid, pos, q in my_col:
        if wid in assigned:
            continue
        db_here = dbase[pos]
        home = min(my_base_cells, key=lambda b: D[pos][b]) if my_base_cells else pos
        if q > 0:
            options.append((-(R_STAR + RISK * q) * db_here, wid, "home", home))
            if left <= db_here + 1:
                continue
        else:
            options.append((-1e6, wid, "idle", home))
        cand = []
        for j in range(n2):
            e0 = est[j]
            if e0 < 2 or j in tree_at or j in bases:
                continue
            d1 = D[pos][j]
            if d1 > 10:
                continue
            e = e0
            if d1 > 0 and not tree_adj[j] and P["grow"]:
                e = min(cap, e0 * g ** min(d1, P["grow"]))
            fac = 1.0
            if my_tree_adj[j]:
                fac *= P["own_tree_cell"]
            if en_tree_aura[j]:
                fac *= 0.75
            if tree_adj[j] and not my_tree_adj[j]:
                fac *= P["deny_bonus"]
            if en_min_near[j] < INF and j != pos and en_min_near[j] <= q:
                fac *= 0.6
            db = dbase[j]
            bestk = None
            for k in (1, 2, 3, 4, 5, 6):
                tt = d1 + k + db
                if tt > left:
                    break
                gain = e * keep[k] * fac
                sc = gain - (R_STAR + RISK * (q + gain * 0.5)) * tt
                if bestk is None or sc > bestk:
                    bestk = sc
            if bestk is not None:
                if prev_tgt.get(wid) == j and pos != j:
                    bestk += P["stick"]
                cand.append((bestk, j))
        cand.sort(reverse=True)
        for sc, j in cand[:8]:
            options.append((sc, wid, "harvest", j))
    options.sort(key=lambda t: -t[0])
    for sc, wid, mode, tgt in options:
        if wid in assigned:
            continue
        if mode == "harvest" and tgt in targets_taken:
            continue
        assigned[wid] = (mode, tgt)
        if mode == "harvest":
            targets_taken.add(tgt)

    st["col_tgt"] = {w: m[1] for w, m in assigned.items() if m[0] == "harvest"}
    # homing along a safe shortest path (greedy steps get trapped by a single parked enemy)
    stuck = st.get("stuck", {})
    new_stuck = {}

    def home_move(wid, pos, q):
        blocked = set(opp_base_cells)
        for x in range(n2):
            if x not in my_base_cells and collector_danger(x, q) >= 100:
                blocked.add(x)
        sd = {b: 0 for b in my_base_cells}
        frontier = list(my_base_cells)
        while frontier:
            nxt = []
            for u in frontier:
                for _, v in geo.nb[u]:
                    if v not in sd and v not in blocked:
                        sd[v] = sd[u] + 1
                        nxt.append(v)
            frontier = nxt
        best = None
        for mv, x in geo.nb[pos]:
            if x in claimed and not (x in my_base_cells and left <= P["end_crowd"]):
                continue
            if x in sd:
                key = (sd[x], collector_danger(x, q))
                if best is None or key < best[0]:
                    best = (key, mv, x)
        if best is not None and (pos not in sd or best[0][0] < sd[pos] or pos in blocked):
            return best[1], best[2]
        if pos in sd and pos not in claimed and pos not in blocked:
            return None, pos  # safe route exists but next cell is taken by a teammate: wait
        if best is not None:
            return best[1], best[2]
        # no safe route: wait a little, then accept the risk on the shortest route
        k = stuck.get(wid, 0) + 1
        new_stuck[wid] = k
        if k < P["stuck_risk"] and pos not in claimed and collector_danger(pos, q) < 100:
            return None, pos
        cand = []
        for mv, x in geo.nb[pos]:
            if x in claimed or x in opp_base_cells:
                continue
            lighter = sum(1 for _, ep, ec in en_col if ec < q and D[ep][x] <= 1)
            cand.append((dbase[x] * 10 + lighter * 6, mv, x))
        if not cand:
            return None, pos
        cand.sort()
        return cand[0][1], cand[0][2]

    # gate breaker: ram an enemy collector parked next to our base while our loaded collectors wait
    if stuck:
        waiting = [col_by_id[w] for w in stuck if w in col_by_id]
        for ewid, epos, ec in en_col:
            if dbase[epos] > 2 or st.get("en_prev_pos", {}).get(ewid) != epos:
                continue
            if not any(D[p][epos] <= 3 and qq > 50 for p, qq in waiting):
                continue
            best = None
            for wid, pos, q in my_col:
                if q > 0 or D[pos][epos] > 2 or assigned.get(wid, ("",))[0] in ("hunt", "guard"):
                    continue
                if best is None or D[pos][epos] < best[0]:
                    best = (D[pos][epos], wid)
            if best is not None:
                assigned[best[1]] = ("hunt", epos)

    # 3) movement, most valuable cargo first
    for wid, pos, q in sorted(my_col, key=lambda t: -t[2]):
        mode, tgt = assigned.get(wid, ("idle", pos))
        if mode == "hunt" and tgt not in claimed and D[pos][tgt] == 1:
            actions[wid] = step_onto(pos, tgt)
            claimed.add(tgt)
            continue
        if q > 0 and left <= P["end_crowd"] and dbase[pos] == 1:
            # last steps: collisions on our own base still credit all cargo to us
            b = next(b for b in my_base_cells if D[pos][b] == 1)
            actions[wid] = step_onto(pos, b)
            claimed.add(b)
            continue
        if mode == "home" and q > 0 and my_base_cells:
            mv, x = home_move(wid, pos, q)
        else:
            stay_here = mode in ("harvest", "guard", "path") and pos == tgt
            mv, x = choose_move(pos, tgt, lambda x, q=q: collector_danger(x, q), True,
                                stay_bonus=5.0 if stay_here else -3.0)
        if mv is not None:
            actions[wid] = mv
        claimed.add(x)
    st["stuck"] = new_stuck
    st["en_prev_pos"] = {wid: w[0] for wid, w in workers.items() if w[3] != me}
    if _DEBUG:
        st["dbg_assigned"] = {wid: (col_by_id[wid], assigned.get(wid), actions.get(wid)) for wid in col_by_id}

    # ---------------------------------------------------------------- planters
    decay = 1 - cfg["cellAbsorptionRate"]
    life = cfg["treeLifespan"] - 1
    planned_adj = [0] * n2
    incoming_cash = cash
    for wid, pos, q in my_col:
        if q > 0 and pos in my_base_cells:
            pass
    for wid, pos, q in my_col:
        a = actions.get(wid)
        nxt = pos if a is None else dict(geo.nb[pos])[a]
        if nxt in my_base_cells:
            incoming_cash += q

    def site_value(x, d, for_seize=False, age=0):
        L = min(life, left - d - 2)
        if for_seize:
            L = min(cfg["treeLifespan"] - age - d - 1, left - d - 2)
        if L <= 0:
            return -1e9
        f = 1 - decay ** L
        s = 0.0
        for j in geo.n4[x]:
            if j in tree_at or j in bases:
                continue
            share = 1 + tree_adj[j] + planned_adj[j] - (1 if for_seize else 0)
            s += est[j] / max(1, share)
        risk = 1.0
        if not for_seize:
            dmin_en_pl = min((D[x][p] for _, p in en_pl), default=99)
            if dmin_en_pl <= 3:
                risk *= 0.8
        return s * f * risk

    pl_order = sorted(my_pl, key=lambda t: t[0])
    for wid, pos in pl_order:
        best = None
        for x in range(n2):
            if x in bases:
                continue
            d = D[pos][x]
            if x in tree_at:
                pid, age, tid = tree_at[x]
                if pid == me:
                    continue
                v = P["seize_mult"] * site_value(x, d, True, age) - cfg["seizeCost"]
                need = cfg["seizeCost"]
            else:
                if x in targets_taken and x != pos:
                    pass
                v = P["tree_eff"] * site_value(x, d) - P["cost_mult"] * plant_cost - 0.5 * est[x]
                need = plant_cost
            if v <= P["plant_margin"]:
                continue
            sc = v / (d + 3)
            if best is None or sc > best[0]:
                best = (sc, x, v, need)
        here_tree = tree_at.get(pos)
        on_safe_park = pos in my_base_cells or (here_tree is not None and here_tree[0] == me)
        if best is not None:
            tgt = best[1]
            if tgt == pos and planter_danger(pos) < 1000 and incoming_cash >= best[3] and pos not in claimed:
                # stand still: plant or seize here
                claimed.add(pos)
                incoming_cash -= best[3]
                for j in geo.n4[pos]:
                    planned_adj[j] += 1
                continue
            for j in geo.n4[tgt]:
                planned_adj[j] += 1
            # if we cannot afford, standing still is harmless unless cash may arrive
            stay_ok = on_safe_park or incoming_cash < plant_cost
            mv, x = choose_move(pos, tgt, planter_danger, stay_ok, stay_bonus=-2.0)
        else:
            # no good site: drift toward nearest own tree / base
            home = min(my_base_cells, key=lambda b: D[pos][b]) if my_base_cells else pos
            stay_ok = on_safe_park or incoming_cash < plant_cost
            mv, x = choose_move(pos, home, planter_danger, stay_ok, stay_bonus=1.0)
        if mv is not None:
            actions[wid] = mv
        claimed.add(x)

    # ---------------------------------------------------------------- recruiting
    n_col = len(my_col)
    n_pl = len(my_pl)
    total = n_col + n_pl
    c_cost = cfg["recCollectorCost"] + cfg["recCollectorIncreaseCost"] * sum(1 for w in workers.values() if w[2])
    p_cost = cfg["recPlanterCost"] + cfg["recPlanterIncreaseCost"] * sum(1 for w in workers.values() if not w[2])
    want_pl = P["want_pl"] if step < T_end - 60 else (1 if step < T_end - 25 else 0)
    want_col = P["want_col"]
    budget = cash
    # sort bases by local richness
    for bid, bpos in my_bases:
        if total >= cfg["workerLimit"] or bpos in claimed:
            continue
        kind = None
        if n_col < 2 and step < T_end - 20:
            kind = "c"
        elif n_pl < want_pl and n_pl <= n_col and step < T_end - 40:
            kind = "p"
        elif n_col < want_col and step < T_end - 30:
            kind = "c"
        elif n_pl < want_pl and step < T_end - 40:
            kind = "p"
        if kind is None:
            continue
        cost = c_cost if kind == "c" else p_cost
        reserve = 0.0 if kind == "c" and n_col < 2 else (plant_cost if n_pl > 0 else 0.0)
        if kind == "p" and n_pl == 0:
            reserve = 0.0
        if step == 0:
            reserve = 0.0
        if budget - cost < reserve and not (step == 0):
            continue
        if budget < cost:
            continue
        actions[bid] = "RECCOLLECTOR" if kind == "c" else "RECPLANTER"
        budget -= cost
        claimed.add(bpos)
        total += 1
        if kind == "c":
            n_col += 1
        else:
            n_pl += 1

    return {k: v for k, v in actions.items() if v is not None}
