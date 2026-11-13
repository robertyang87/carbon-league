"""Carbon League submission — strategy name: XiaoD-Bowl-v4.

名字来历：碗，是把散落的碳一勺一勺盛回来的容器；也是我能收回、又收不回的那只。
（对外只用「小D」，正名不出现在任何公开名字里。）

Single-file strategy for the partially-observable "Carbon Neutrality" game
(15x15 torus, 300 steps, two bases per player).  Standard library only.

v4 changes after comparing deposit curves with the rank-1 replays:
* dynamic early cargo cap: 45 cash at the start rising to 160 by step ~290.
  In self-play the first deposits came at step 40 with 45 banked; with the new
  ramp it is ~245, and season deposits rise from ~4.1k to ~6.8k per side.
  The rank-1 player banks 353 by step 40 and 752 by step 60, so the old flat
  cap was starving the whole crew during the compounding phase.
* ablation (4 seeds, both seats) of the other v4 ideas: lowering the harvest
  threshold to 8, a 6-cell leash around the bases and 8C+2P each LOST to v3
  head to head, so they were dropped.

v3 changes after studying 16 online games of v2 and two rank-1 replays:
* 8 collectors + 2 planters (middle ground: 9+1 loses the mirror badly, 7+3
  feeds too many trees to seize-heavy opponents).
* cargo cap 160: against a seize-spam opponent 200 scored 1/8 and 120 3/8,
  while 160 scored 5/8 - shorter trips keep the crew intact.
* adaptive anti-theft: when a tree of ours reappears in the opponent's tree
  set it was seized; after two such thefts planting retreats to our own half
  and the orchard cap drops, so planting stops being a donation.

v2 changes after studying the rank-1 replay (TJB-o18-rush-v41):
* composition 7 collectors + 3 planters (rank 1 runs 7+3; planters scout and
  keep planting all game instead of only at the end).
* trees are planted from the opening on: price is cheap while few trees are
  alive, and one tree banks ~85% of its four neighbours over its 50 turns
  without costing a collector turn.  A price cap stops buying into inflation.
* hostile takeovers: enemy trees are seized for 20 whenever the remaining
  income stream is worth clearly more - the rank-1 player did this 16 times.
* anti-predation: a collector with cargo is a liability, because the unique
  *lowest*-cargo collector at a cell wins and takes everyone else's cargo.
  Loaded collectors now flee adjacent enemy collectors and never end a move
  next to one.
* trips last longer (cargo ~200) and leave cells earlier (~22 carbon), which
  matches the rank-1 harvesting sweep instead of shuttling home constantly.
"""
TUNE = {
    "harvest_min": 22.0,      # leave a cell / skip a cell below this estimate
    "target_fresh": 0,        # only chase cells seen this recently (0 = any age)
    "cargo_cap": 160.0,       # walk cargo home above this amount
    "cargo_cap_start": 45.0,  # early-game cap (bank small loads, recruit fast)
    "cargo_cap_gain": 0.40,   # extra cap per step until cargo_cap is reached
    "target_max_dist": 11,    # ignore harvest cells farther than this
    "dist_pow": 1.0,          # distance penalty exponent when scoring harvest targets
    "home_bias_cargo": 30.0,  # once carrying this much, only harvest on the way home
    "leash": 99,              # never harvest farther than this from one of our bases
    "collector_target": 7,    # desired collectors
    "planter_target": 3,      # desired planters
    "endgame_planters": 4,    # planters wanted late
    "endgame_step": 200,      # recruitment priority switch
    "plant_start": 10,        # first step where planting is allowed
    "plant_stop": 292,        # no planting after this step
    "plant_price_cap": 200.0, # never plant when the market price is above this
    "plant_margin": 1.2,      # plant if value > price * margin
    "plant_max": 20,          # keep at most this many own trees alive
    "plant_fresh": 20,        # neighbour estimate must be at most this old
    "home_plant": 0,          # only plant on our side of the board (harder to seize)
    "seize_value": 40.0,      # seize enemy trees worth at least this much
    "seize_dist": 6,          # only chase seizure targets within this range
    "seize_guard": 1,         # skip trees with an enemy unit this close (races kill both planters)
    "recruit_stop": 286,      # no recruitment after this step
    "home_slack": 2,          # spare turns reserved for the trip home
    "flee_cargo": 25.0,       # do not bother fleeing with less cargo
    "reserve": 0,             # cash kept while the crew is still tiny
}

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _get(cfg, key, default):
    try:
        v = cfg[key]
        if v is not None:
            return v
    except Exception:
        pass
    try:
        v = getattr(cfg, key)
        if v is not None:
            return v
    except Exception:
        pass
    return default


def _idx(r, c, size):
    return (r % size) * size + (c % size)


def _dist(a, b, size):
    ar, ac = divmod(a, size)
    br, bc = divmod(b, size)
    dr = abs(ar - br)
    dc = abs(ac - bc)
    return min(dr, size - dr) + min(dc, size - dc)


def _move_dir(pos, dst, size):
    r, c = divmod(pos, size)
    tr, tc = divmod(dst, size)
    dr = ((tr - r + size // 2) % size) - size // 2
    dc = ((tc - c + size // 2) % size) - size // 2
    if dr == 0 and dc == 0:
        return None
    if abs(dr) >= abs(dc):
        return "DOWN" if dr > 0 else "UP"
    return "RIGHT" if dc > 0 else "LEFT"


def _nbrs(pos, size, dirs):
    r, c = divmod(pos, size)
    return [_idx(r + dr, c + dc, size) for dr, dc in dirs]


class _State(object):
    def __init__(self, cfg):
        self.size = int(_get(cfg, "size", 15))
        self.n = self.size * self.size
        self.bel = [None] * self.n          # [carbon, step_seen] or None
        self.mirror = [0] * self.n
        S = self.size
        for r in range(S):
            for c in range(S):
                self.mirror[r * S + c] = (S - 1 - r) * S + (S - 1 - c)
        self.step = -1
        self.tgt = {}
        self.prev_mine = set()
        self.stolen = 0
        self.cfg = {
            "regen": float(_get(cfg, "regenRate", 0.03)),
            "max_carbon": float(_get(cfg, "maxCellCarbon", 100)),
            "collect_rate": float(_get(cfg, "initialCollectRate", 0.3)),
            "cell_abs": float(_get(cfg, "cellAbsorptionRate", 0.0375)),
            "col_abs": float(_get(cfg, "collectorAbsorptionRate", 0.1)),
            "tree_life": int(_get(cfg, "treeLifespan", 50)),
            "plant_cost": float(_get(cfg, "plantCost", 20)),
            "inflation": float(_get(cfg, "plantCostInflationRatio", 5)),
            "inflate_base": float(_get(cfg, "plantCostInflationBase", 1.235)),
            "protective": int(_get(cfg, "playerTreeProtectiveNumber", 0)),
            "seize_cost": float(_get(cfg, "seizeCost", 20)),
            "rec_collector": float(_get(cfg, "recCollectorCost", 30)),
            "rec_planter": float(_get(cfg, "recPlanterCost", 30)),
            "worker_limit": int(_get(cfg, "workerLimit", 10)),
            "episode_steps": int(_get(cfg, "episodeSteps", 300)),
        }
        self.cfg["end_step"] = self.cfg["episode_steps"] - 1
        self.T = dict(TUNE)
        shift = self.cfg["episode_steps"] - 300
        if shift:
            for k in ("plant_start", "plant_stop", "recruit_stop", "endgame_step"):
                self.T[k] = max(0, self.T[k] + shift)

    # ---------------- perception ----------------
    def observe(self, obs, step):
        self.step = step
        carbon = obs["carbon"]
        me = obs["player"]
        workers = obs["players"][me][2]
        S = self.size
        seen = {}
        for w in workers.values():
            pos = w[0]
            if w[2] == "COLLECTOR":
                seen[pos] = carbon[pos]
            else:
                for p in _nbrs(pos, S, DIRS4):
                    seen[p] = carbon[p]
        bel = self.bel
        for p, v in seen.items():
            v = float(v)
            bel[p] = [v, step]
            m = self.mirror[p]
            if m != p and bel[m] is None:
                bel[m] = [v, step]

    def estimate(self, pos, step, tree_cells):
        b = self.bel[pos]
        if b is None:
            return None
        v, t = b
        if v <= 0:
            return 0.0
        if pos in tree_cells:
            return 0.0
        age = step - t
        if age <= 0:
            return v
        adj = False
        for p in _nbrs(pos, self.size, DIRS4):
            if p in tree_cells:
                adj = True
                break
        if adj:
            return v * ((1.0 - self.cfg["cell_abs"]) ** age)
        return min(v * ((1.0 + self.cfg["regen"]) ** min(age, 60)), self.cfg["max_carbon"])

    def fresh_estimate(self, pos, step, tree_cells, max_age):
        b = self.bel[pos]
        if b is None or step - b[1] > max_age:
            return None
        return self.estimate(pos, step, tree_cells)

    # ---------------- prices ----------------
    def plant_price(self, my_tree_count, all_tree_count):
        if my_tree_count <= self.cfg["protective"]:
            return self.cfg["plant_cost"]
        return self.cfg["plant_cost"] + self.cfg["inflation"] * (
            self.cfg["inflate_base"] ** all_tree_count)

    # ---------------- targets ----------------
    def best_harvest_target(self, pos, step, tree_cells, my_tree_cells, opp_workers,
                            opp_tree_cells, banned, cargo, T, bases):
        size = self.size
        end = self.cfg["end_step"]
        best, best_score = None, 0.0
        bel = self.bel
        for i in range(self.n):
            if i in banned or i in tree_cells or i in bases:
                continue
            b = bel[i]
            if b is None:
                continue
            if T["target_fresh"] and step - b[1] > T["target_fresh"]:
                continue
            e = self.estimate(i, step, tree_cells)
            if e is None or e < T["harvest_min"]:
                continue
            if my_tree_cells and any(p in my_tree_cells for p in _nbrs(i, size, DIRS4)):
                e *= 0.3
            d = _dist(pos, i, size)
            if d > T["target_max_dist"]:
                continue
            home = min((_dist(i, bb, size) for bb in bases), default=0)
            if step + d + 2 + home + T["home_slack"] > end:
                continue
            if home > T["leash"]:
                continue
            if cargo >= T["home_bias_cargo"] and bases:
                here = min(_dist(pos, bb, size) for bb in bases)
                if home > here + 1:
                    continue
            score = e / ((d + 1.5) ** T["dist_pow"])
            r, c = divmod(i, size)
            for dr, dc in DIRS8:
                if _idx(r + dr, c + dc, size) in opp_tree_cells:
                    score *= 0.35
                    break
            threat = False
            for ow in opp_workers.values():
                if ow[2] == "COLLECTOR" and _dist(i, ow[0], size) <= 1 and cargo >= ow[1]:
                    threat = True
                    break
            if threat:
                score *= 0.25
            if score > best_score:
                best, best_score = i, score
        return best

    def explore_target(self, pos, step, tree_cells, banned, T):
        size = self.size
        end = self.cfg["end_step"]
        best, bd = None, 999
        for i in range(self.n):
            if self.bel[i] is None and i not in banned and i not in tree_cells:
                d = _dist(pos, i, size)
                if d < bd and step + d + 2 < end:
                    best, bd = i, d
        return best

    def best_plant_site(self, pos, step, cash, tree_cells, my_tree_count,
                        all_tree_count, T, equip, own_bases, opp_bases):
        size = self.size
        end = self.cfg["end_step"]
        if step < T["plant_start"] or step > T["plant_stop"]:
            return None
        if my_tree_count >= T["plant_max"]:
            return None
        price = self.plant_price(my_tree_count, all_tree_count)
        if price > T["plant_price_cap"] or cash < price:
            return None
        remaining = min(end - step + 1, self.cfg["tree_life"])
        if remaining < 8:
            return None
        capture = 1.0 - (1.0 - self.cfg["cell_abs"]) ** remaining
        best, best_score = None, 0.0
        for i in range(self.n):
            if i in tree_cells or i in equip:
                continue
            if T["home_plant"] and own_bases and opp_bases:
                d_own = min(_dist(i, b, size) for b in own_bases)
                d_opp = min(_dist(i, b, size) for b in opp_bases)
                if d_own > d_opp + 1:
                    continue
            nbrs = _nbrs(i, size, DIRS4)
            if any(p in tree_cells for p in nbrs):
                continue
            s = 0.0
            ok = True
            for p in nbrs:
                e = self.fresh_estimate(p, step, tree_cells, T["plant_fresh"])
                if e is None:
                    ok = False
                    break
                s += e
            if not ok:
                continue
            e_self = self.estimate(i, step, tree_cells) or 0.0
            value = s * capture - e_self
            if value < price * T["plant_margin"]:
                continue
            d = _dist(pos, i, size)
            if step + d + 1 > end:
                continue
            score = value / (d + 3.0)
            if score > best_score:
                best, best_score = i, score
        return best

    def best_seize_target(self, pos, step, opp_trees, tree_cells, cash, T, opp_workers):
        size = self.size
        end = self.cfg["end_step"]
        if cash < self.cfg["seize_cost"]:
            return None
        best, best_score = None, 0.0
        for tid, tv in opp_trees.items():
            tpos, age = tv[0], tv[1]
            remaining = min(self.cfg["tree_life"] - age, end - step)
            if remaining < 10:
                continue
            capture = 1.0 - (1.0 - self.cfg["cell_abs"]) ** remaining
            s = 0.0
            ok = True
            for p in _nbrs(tpos, size, DIRS4):
                if p in tree_cells:
                    ok = False
                    break
                e = self.estimate(p, step, tree_cells)
                if e is None:
                    ok = False
                    break
                s += e
            if not ok:
                continue
            value = s * capture - self.cfg["seize_cost"]
            if value < T["seize_value"]:
                continue
            if T["seize_guard"]:
                near_enemy = False
                for ow in opp_workers.values():
                    if _dist(ow[0], tpos, size) <= T["seize_guard"]:
                        near_enemy = True
                        break
                if near_enemy:
                    continue
            d = _dist(pos, tpos, size)
            if d > T["seize_dist"] or step + d + 1 > end:
                continue
            score = value / (d + 3.0)
            if score > best_score:
                best, best_score = tpos, score
        return best

    def roam_step(self, pos, step, size, banned, tree_cells):
        r, c = divmod(pos, size)
        best, best_score = None, -1.0
        for name, (dr, dc) in sorted(DIRS.items()):
            p2 = _idx(r + dr, c + dc, size)
            if p2 in banned:
                continue
            unk = 0
            for q in _nbrs(p2, size, DIRS4):
                if self.bel[q] is None:
                    unk += 1
            e = self.estimate(p2, step, tree_cells) or 0.0
            score = unk + 0.02 * e
            if p2 in tree_cells:
                score -= 0.5
            if score > best_score:
                best, best_score = (name, p2), score
        return best


_STATE = None


def _agent(observation, configuration):
    global _STATE
    step = int(observation["step"])
    if _STATE is None or step == 0 or step <= _STATE.step:
        _STATE = _State(configuration)
    st = _STATE
    st.observe(observation, step)

    players = observation["players"]
    me = int(observation["player"])
    size = st.size
    T = st.T
    end = st.cfg["end_step"]
    my, opp = players[me], players[1 - me]
    cash = float(my[0])
    centers = my[1]
    workers = my[2]
    my_trees = my[3]
    opp_workers = opp[2]
    opp_trees = opp[3]
    own_bases = list(centers.values())
    own_base_set = set(own_bases)

    # hostile takeovers against us: a tree that leaves our set and shows up in
    # the opponent's set was seized.  Repeated theft turns planting into a
    # liability, so fall back to defensible home turf and a smaller orchard.
    cur_mine = set(my_trees)
    cur_opp = set(opp_trees)
    if st.prev_mine:
        for tid in st.prev_mine - cur_mine:
            if tid in cur_opp:
                st.stolen += 1
    st.prev_mine = cur_mine
    if st.stolen >= 2:
        T["home_plant"] = 1
        T["plant_max"] = min(T["plant_max"], 12)

    tree_cells = set()
    for t in my_trees.values():
        tree_cells.add(t[0])
    for t in opp_trees.values():
        tree_cells.add(t[0])
    my_tree_cells = set(t[0] for t in my_trees.values())
    all_tree_count = len(my_trees) + len(opp_trees)
    opp_tree_cells = set(t[0] for t in opp_trees.values())
    equip = set(own_bases) | set(opp[1].values())

    actions = {}

    # ------------------------------------------------------------------
    # recruitment: 1 collector + 1 planter first (planter scouts), then
    # 7 collectors + 3 planters like the rank-1 build.
    # ------------------------------------------------------------------
    n_col = sum(1 for w in workers.values() if w[2] == "COLLECTOR")
    n_pl = len(workers) - n_col
    n_workers = len(workers)
    if step <= T["recruit_stop"] and centers:
        want_pl = T["endgame_planters"] if step >= T["endgame_step"] else T["planter_target"]
        order = []
        if n_workers == 0:
            order.append("collector")
        elif n_pl == 0:
            order.append("planter")
        order += ["collector"] * max(0, T["collector_target"] - n_col)
        order += ["planter"] * max(0, want_pl - n_pl)
        pending_cash = cash
        for cid in sorted(centers.keys()):
            if n_workers >= st.cfg["worker_limit"] or not order:
                break
            kind = order[0]
            cost = st.cfg["rec_planter"] if kind == "planter" else st.cfg["rec_collector"]
            hold = T["reserve"] if (n_workers + 1) <= 2 else 0
            if pending_cash - cost < hold:
                break
            pending_cash -= cost
            n_workers += 1
            if kind == "planter":
                n_pl += 1
            else:
                n_col += 1
            actions[cid] = "RECPLANTER" if kind == "planter" else "RECCOLLECTOR"
            order.pop(0)

    # ------------------------------------------------------------------
    # unit planning
    # ------------------------------------------------------------------
    planned = {}
    claimed = set()

    def pick_move(pos, cargo, target, banned, flee=False, any_enemy=False):
        """One step toward target, but avoid cells occupied by enemy
        collectors that would win a collision with us, and cells right next
        to such predators."""
        enemy_cells = set()
        for ow in opp_workers.values():
            if ow[2] == "COLLECTOR" or any_enemy:
                enemy_cells.add(ow[0])
        pred_cells = [ow[0] for ow in opp_workers.values()
                      if ow[2] == "COLLECTOR" and (any_enemy or cargo >= ow[1])]
        r, c = divmod(pos, size)
        cand = []
        for name, (dr, dc) in DIRS.items():
            p2 = _idx(r + dr, c + dc, size)
            if p2 in banned or p2 in enemy_cells:
                continue
            dth = min((_dist(p2, t, size) for t in pred_cells), default=9)
            cand.append((name, p2, dth, _dist(p2, target, size)))
        if not cand:
            return None, pos
        if flee:
            cand.sort(key=lambda x: (-x[2], x[3]))
        else:
            cand.sort(key=lambda x: (0 if x[2] >= 2 else 1, x[3], -x[2]))
        name, p2, _, _ = cand[0]
        return name, p2

    for wid in sorted(workers.keys()):
        w = workers[wid]
        pos, cargo, proto = w[0], float(w[1]), w[2]
        act = None
        dpos = pos

        if proto == "COLLECTOR":
            cap_now = min(T["cargo_cap"], T["cargo_cap_start"] + T["cargo_cap_gain"] * step)
            home = min((_dist(pos, b, size) for b in own_bases), default=0)
            near_threat = False
            if cargo >= T["flee_cargo"] and pos not in own_base_set:
                for ow in opp_workers.values():
                    if ow[2] == "COLLECTOR" and cargo >= ow[1] and _dist(pos, ow[0], size) <= 1:
                        near_threat = True
                        break
            if near_threat:
                tgt_home = min(own_bases, key=lambda b: _dist(pos, b, size)) if own_bases else pos
                act, dpos = pick_move(pos, cargo, tgt_home, set(planned), flee=True)
                st.tgt.pop(wid, None)
            elif cargo >= cap_now or step + home + T["home_slack"] >= end:
                tgt_home = min(own_bases, key=lambda b: _dist(pos, b, size)) if own_bases else pos
                st.tgt.pop(wid, None)
                if tgt_home != pos:
                    act, dpos = pick_move(pos, cargo, tgt_home, set(planned))
            else:
                tgt = st.tgt.get(wid)
                if tgt is not None:
                    e = st.estimate(tgt, step, tree_cells)
                    if e is None or e < T["harvest_min"]:
                        st.tgt.pop(wid, None)
                        tgt = None
                if tgt is None:
                    busy = set(st.tgt.values())
                    tgt = st.best_harvest_target(pos, step, tree_cells, my_tree_cells,
                                                 opp_workers, opp_tree_cells,
                                                 claimed | busy, cargo, T, own_bases)
                    if tgt is None:
                        if cargo > 0.0 and own_bases:
                            tgt_home = min(own_bases, key=lambda b: _dist(pos, b, size))
                            act, dpos = pick_move(pos, cargo, tgt_home, set(planned))
                        else:
                            tgt = st.explore_target(pos, step, tree_cells, claimed | busy, T)
                    if tgt is not None:
                        st.tgt[wid] = tgt
                if tgt is not None:
                    claimed.add(tgt)
                    if tgt == pos:
                        act, dpos = None, pos
                    else:
                        act, dpos = pick_move(pos, cargo, tgt, set(planned))
        else:
            tgt = st.best_seize_target(pos, step, opp_trees, tree_cells, cash, T, opp_workers)
            if tgt is None:
                tgt = st.best_plant_site(pos, step, cash, tree_cells, len(my_trees),
                                         all_tree_count, T, equip, own_bases,
                                         list(opp[1].values()))
            if tgt is not None:
                claimed.add(tgt)
                if tgt == pos:
                    act, dpos = None, pos
                else:
                    act, dpos = pick_move(pos, 0.0, tgt, set(planned), any_enemy=True)
            else:
                mv = st.roam_step(pos, step, size, set(planned) | claimed, tree_cells)
                if mv is not None:
                    claimed.add(mv[1])
                    act, dpos = mv
                else:
                    act, dpos = None, pos

        # avoid stacking our own units (self collisions kill workers)
        if dpos in planned:
            alt_act, alt_pos = None, pos
            r, c = divmod(pos, size)
            for name, (dr, dc) in sorted(DIRS.items()):
                p2 = _idx(r + dr, c + dc, size)
                if p2 not in planned:
                    alt_act, alt_pos = name, p2
                    break
            act, dpos = alt_act, alt_pos
        planned[dpos] = wid
        if act is not None:
            actions[wid] = act

    return actions


def agent(observation, configuration):
    """Entry point required by the league.  Never raise: an exception in any
    step counts as a lost game, so a defensive fallback keeps the crew alive."""
    try:
        return _agent(observation, configuration)
    except Exception:
        return {}