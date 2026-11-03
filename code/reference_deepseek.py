"""Carbon League strategy (v3) - tree engine + grazing collectors.

Facts driving the design:
  * cell carbon compounds +3%/step while untouched and never recovers from 0,
    so cells want grazing, not stripping;
  * a standing collector takes 30% of its cell, a moving one nothing;
  * a tree converts 3.75%/step of each of its 4 neighbours (plus 10% of enemy
    cargo in the 8-ring) into cash and drops 30 carbon when it withers, so a
    well-placed tree returns ~0.85 * (sum of neighbour carbon) + 30 for a
    price of 20 + 5 * 1.235**live_trees (inflated globally -> plant early);
  * on a shared cell only the lightest collector survives and absorbs the
    rest, so keep our units apart from each other and from stronger enemies.
"""

import os

P = {
    "target_collectors": 8,
    "target_planters": 2,
    "cargo_base": 40.0,
    "cargo_per_dist": 9.0,
    "cargo_near": 35.0,
    "collect_floor": 18.0,
    "collect_floor_drained": 6.0,
    "unknown_prior": 30.0,
    "search_radius": 6,
    "enemy_penalty": 0.45,
    "plant_cap": 10,
    "plant_min_gain": 40.0,
    "plant_reserve": 0.0,
    "planter_radius": 7,
}

MOVES = ("UP", "RIGHT", "DOWN", "LEFT")


def _rc(i, n):
    return i // n, i % n


def _idx(r, c, n):
    return (r % n) * n + (c % n)


def _mv(i, d, n):
    r, c = _rc(i, n)
    if d == 0:
        r -= 1
    elif d == 1:
        c += 1
    elif d == 2:
        r += 1
    else:
        c -= 1
    return _idx(r, c, n)


def _dist(a, b, n):
    ar, ac = _rc(a, n)
    br, bc = _rc(b, n)
    dr = abs(ar - br)
    dc = abs(ac - bc)
    return min(dr, n - dr) + min(dc, n - dc)


class Bot:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step = -1
        self.mem = {}
        self.home = {}
        self.target = {}

    # ------------------------------------------------------------- belief
    def _est(self, pos, n, step):
        e = self.mem.get(pos)
        if e is None:
            return P["unknown_prior"]
        val, seen = e
        dt = step - seen
        if dt <= 0:
            return val
        if val <= 0.5:
            return 0.0
        return min(val * (1.03 ** dt), 100.0)

    # -------------------------------------------------------------- entry
    def act(self, obs, conf):
        step = obs["step"]
        n = conf["size"]
        if step == 0 or step < self.step:
            self.reset()
        self.step = step
        episode = conf.get("episodeSteps", 300)
        remaining = episode - step

        me = obs["player"]
        players = obs["players"]
        my, opp = players[me], players[1 - me]
        cash = float(my[0])
        base_ids = {k: int(v) for k, v in my[1].items()}
        bases = list(base_ids.values())
        workers = {k: (int(v[0]), float(v[1]), v[2]) for k, v in my[2].items()}
        my_trees = {int(v[0]): int(v[1]) for v in my[3].values()}
        e_units = [(int(v[0]), float(v[1]), v[2]) for v in opp[2].values()]
        e_trees = {int(v[0]): int(v[1]) for v in opp[3].values()}
        carbon = obs["carbon"]

        for wid, (pos, cargo, typ) in workers.items():
            self.mem[pos] = (float(carbon[pos]), step)
            if typ == "PLANTER":
                for d in range(4):
                    p = _mv(pos, d, n)
                    self.mem[p] = (float(carbon[p]), step)

        for wid in list(self.home):
            if wid not in workers:
                self.home.pop(wid, None)
                self.target.pop(wid, None)

        e_coll = {}
        for pos, cargo, typ in e_units:
            if typ == "COLLECTOR":
                e_coll.setdefault(pos, []).append(cargo)
        e_occ = set(u[0] for u in e_units)
        danger = set()
        for p0 in e_occ:
            danger.add(p0)
            for d in range(4):
                danger.add(_mv(p0, d, n))

        actions = {}
        my_occ = {}
        for wid, (pos, cargo, typ) in workers.items():
            my_occ.setdefault(pos, []).append(wid)

        # ---- recruitment ----
        n_col = sum(1 for w in workers.values() if w[2] == "COLLECTOR")
        n_pla = len(workers) - n_col
        price = self._plant_price(my_trees, e_trees, conf)
        if len(workers) < 10:
            spot_exists = self._plant_spot(bases, workers, my_trees, e_trees, n, step, conf,
                                           price) is not None
            # keep one recruit's worth of cash in hand so a wipe cannot strand us
            buffer = 30.0 if len(workers) < 9 else 0.0
            want_pl = (n_pla < P["target_planters"] and spot_exists and
                       cash >= 30.0 + price + buffer + 30.0)
            want_col = n_col < P["target_collectors"]
            for bid, bpos in sorted(base_ids.items()):
                if bpos in my_occ:
                    continue
                if want_col and cash >= 30.0 + buffer:
                    cash -= 30.0
                    actions[bid] = "RECCOLLECTOR"
                    want_col = False
                elif want_pl and cash >= 30.0 + price + buffer:
                    cash -= 30.0
                    actions[bid] = "RECPLANTER"
                    want_pl = False

        # ---- orders ----
        order = sorted(workers.items(), key=lambda kv: -kv[1][1])
        claimed = {}
        plan = {}
        for wid, (pos, cargo, typ) in order:
            if typ == "COLLECTOR":
                kind, tgt = self._collector(pos, cargo, wid, bases, e_coll, n, step, remaining)
            else:
                kind, tgt = self._planter(pos, wid, bases, workers, my_trees, e_trees,
                                          n, step, conf, cash, price)
            plan[wid] = (kind, tgt)
            if kind == "stay":
                claimed[pos] = wid

        for wid, (pos, cargo, typ) in order:
            kind, tgt = plan[wid]
            if kind == "stay":
                continue
            mv = self._pick_move(pos, tgt, claimed, e_coll, e_occ, n, kind == "home", danger)
            if mv is None:
                continue
            d, dest = mv
            if dest in claimed:
                continue
            claimed[dest] = wid
            actions[wid] = MOVES[d]
        return actions

    # ---------------------------------------------------------- utilities
    def _plant_price(self, my_trees, e_trees, conf):
        alive = len(my_trees) + len(e_trees)
        try:
            return conf.get("plantCost", 20) + conf.get("plantCostInflationRatio", 5) * \
                   (conf.get("plantCostInflationBase", 1.235) ** alive)
        except Exception:
            return 1e9

    def _nearest_base(self, pos, bases, n):
        return min(bases, key=lambda b: _dist(pos, b, n))

    def _rammed_if_stay(self, pos, cargo, e_coll, n):
        if cargo <= 1:
            return False
        for d in range(4):
            for ec in e_coll.get(_mv(pos, d, n), ()):
                if ec < cargo - 1e-9:
                    return True
        return False

    # --------------------------------------------------------- collectors
    def _collector(self, pos, cargo, wid, bases, e_coll, n, step, remaining):
        home = self._nearest_base(pos, bases, n)
        dh = _dist(pos, home, n)
        here = self._est(pos, n, step)

        if cargo > 0 and (remaining <= dh + 1 or remaining <= 6):
            self.home[wid] = True

        if self.home.get(wid):
            if pos in bases:
                self.home[wid] = False
                self.target[wid] = None
                return ("stay", pos)
            return ("home", home)

        if cargo >= P["cargo_base"] + P["cargo_per_dist"] * dh:
            self.home[wid] = True
            return ("home", home)
        if cargo >= P["cargo_near"] and dh <= 1:
            self.home[wid] = True
            return ("home", home)

        safe = not self._rammed_if_stay(pos, cargo, e_coll, n)
        floor = P["collect_floor"] if cargo == 0 or self.mem.get(pos) else P["collect_floor_drained"]
        if here >= floor and safe:
            self.target[wid] = pos
            return ("stay", pos)

        tgt = self.target.get(wid)
        if tgt is not None and (tgt == pos or self._est(tgt, n, step) < 5.0):
            tgt = None
        if tgt is None:
            tgt = self._pick_target(pos, cargo, home, e_coll, n, step, wid)
            self.target[wid] = tgt
        if tgt is None:
            if cargo > 0:
                self.home[wid] = True
                return ("home", home)
            return ("stay", pos)
        return ("move", tgt)

    def _pick_target(self, pos, cargo, home, e_coll, n, step, wid):
        taken = {v for k, v in self.target.items() if k != wid}
        dh_self = _dist(pos, home, n)
        enemies = list(e_coll.keys())
        best, best_score = None, -1e9
        for c in range(n * n):
            if c == pos or c in taken:
                continue
            est = self._est(c, n, step)
            if est < 8.0:
                continue
            d = _dist(pos, c, n)
            if d > P["search_radius"]:
                continue
            score = est / ((d + 1.0) ** 1.4)
            for ep in enemies:
                de = _dist(c, ep, n)
                if de == 0:
                    score *= 0.05
                elif de == 1:
                    score *= P["enemy_penalty"]
                elif de == 2:
                    score *= 0.75
            if cargo > 50 and _dist(c, home, n) < dh_self:
                score *= 1.2
            if score > best_score:
                best_score, best = score, c
        return best

    # ----------------------------------------------------------- planters
    def _planter(self, pos, wid, bases, workers, my_trees, e_trees, n, step, conf, cash, price):
        # seize an enemy tree under our feet when it is young enough to be worth it
        if pos in e_trees and cash >= conf.get("seizeCost", 20) and e_trees[pos] <= 40:
            return ("stay", pos)
        if cash >= price + P["plant_reserve"] and len(my_trees) < P["plant_cap"]:
            spot = self._plant_spot(bases, workers, my_trees, e_trees, n, step, conf, price,
                                    origin=pos)
            if spot == pos:
                return ("stay", pos)
            if spot is not None:
                return ("move", spot)
        return ("move", self._scout(pos, n, step))

    def _plant_spot(self, bases, workers, my_trees, e_trees, n, step, conf, price=None,
                    origin=None):
        if len(my_trees) >= P["plant_cap"]:
            return None
        if price is None:
            price = self._plant_price(my_trees, e_trees, conf)
        blocked = set(my_trees) | set(e_trees) | set(bases)
        occ = {v[0] for v in workers.values() if v[2] == "COLLECTOR"}
        tree_cells = blocked
        best, best_val = None, -1e9
        for c in range(n * n):
            if c in blocked or c in occ:
                continue
            tot = 0.0
            shares = 0
            for d in range(4):
                p = _mv(c, d, n)
                if p in tree_cells:
                    continue
                v = self._est(p, n, step)
                tot += v
                shares += 1
            if shares == 0:
                continue
            # planting burns the carbon standing on that cell
            gain = tot * 0.85 + 30.0 - self._est(c, n, step)
            val = gain - price
            if val < P["plant_min_gain"]:
                continue
            if origin is not None:
                dd = _dist(origin, c, n)
                if dd > P["planter_radius"]:
                    continue
                val -= 2.0 * dd
            if val > best_val:
                best_val, best = val, c
        return best

    def _scout(self, pos, n, step):
        best, best_score = None, -1e9
        for c in range(n * n):
            e = self.mem.get(c)
            if e is not None and step - e[1] < 10:
                continue
            v = P["unknown_prior"] if e is None else max(e[0], P["unknown_prior"])
            score = v / (_dist(pos, c, n) + 2.0)
            if score > best_score:
                best_score, best = score, c
        return best

    # ----------------------------------------------------------- movement
    def _pick_move(self, pos, tgt, claimed, e_coll, e_occ, n, urgent=False, danger=()):
        if tgt is None or tgt == pos:
            return None
        danger = danger if danger is not None else set()
        best = None
        for d in range(4):
            dest = _mv(pos, d, n)
            if dest in claimed or dest in e_occ or dest in danger:
                continue
            risk = 0
            if not urgent:
                for dd in range(4):
                    if _mv(dest, dd, n) in e_coll:
                        risk += 1
            key = (_dist(dest, tgt, n), risk, d)
            if best is None or key < best[0]:
                best = (key, d, dest)
        if best is None:
            # no safe move: accept adjacency, still refuse enemy-occupied cells
            for d in range(4):
                dest = _mv(pos, d, n)
                if dest not in claimed and dest not in e_occ:
                    return (d, dest)
            return None
        return (best[1], best[2])


_BOT = Bot()


def agent(observation, configuration):
    try:
        return _BOT.act(observation, configuration)
    except Exception:
        return {}