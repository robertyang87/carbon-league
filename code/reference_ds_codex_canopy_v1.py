"""Carbon League: Canopy v1, a deterministic, single-file strategy.

Rules: moliqingwa/carbon_challenge, commit
3a24c67f0b3bb75e4c528fcbe8594f0e4bc29948 (Apache-2.0).
Design references, read during development:
  https://132.232.176.153/source/a938df4254388b8df8ee04f6e6e0aae5
  https://132.232.176.153/source/ae293cad1e06e4b0661da04e345491e2
Their carbon-memory, harvesting and tree-seizure ideas informed this agent.
This implementation does not embed their code or a trained model.

Only the supplied partial observation is used. No files, network, randomness,
third-party dependencies, or online training are required.
"""

import heapq
import math


_MOVES = (None, "UP", "RIGHT", "DOWN", "LEFT")
_DEFAULTS = dict(
    size=15, episodeSteps=300, startingCarbon=2500, carbonCoverage=45,
    startingCellCarbon=50, maxCellCarbon=100, recCollectorCost=30,
    recPlanterCost=30, recCollectorIncreaseCost=0, recPlanterIncreaseCost=0,
    workerLimit=10, plantCost=20, plantCostInflationRatio=5,
    plantCostInflationBase=1.235, playerTreeProtectiveNumber=0,
    co2FrmWithered=30, seizeCost=20, initialCollectRate=0.3,
    collectDecreaseRate=0, regenRate=0.03, treeLifespan=50,
    cellAbsorptionRate=0.0375, collectorAbsorptionRate=0.1,
    smeltCost=0, moveCost=0,
)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class _Geometry:
    def __init__(self, size):
        self.size = size
        self.count = size * size
        self.steps = []
        self.eight = []
        self.mirrors = []
        for p in range(self.count):
            y, x = divmod(p, size)
            # Observations use row-major indices; UP decreases the row.
            self.steps.append((p, ((y - 1) % size) * size + x,
                               y * size + (x + 1) % size,
                               ((y + 1) % size) * size + x,
                               y * size + (x - 1) % size))
            self.eight.append(tuple({((y + dy) % size) * size + (x + dx) % size
                                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)}))
            self.mirrors.append(tuple(sorted({p, y * size + size - 1 - x,
                                              (size - 1 - y) * size + x,
                                              self.count - 1 - p})))
        self.neighbors = [tuple(dict.fromkeys(s[1:])) for s in self.steps]
        self.distance = []
        for p in range(self.count):
            py, px = divmod(p, size)
            row = []
            for q in range(self.count):
                qy, qx = divmod(q, size)
                dy, dx = abs(py - qy), abs(px - qx)
                row.append(min(dy, size - dy) + min(dx, size - dx))
            self.distance.append(row)


def _assets(observation):
    workers, trees = {}, {}
    for owner, player in enumerate(_get(observation, "players")):
        for uid, (pos, cargo, occupation) in player[2].items():
            workers[uid] = (int(pos), float(cargo), occupation == "COLLECTOR", owner)
        for uid, (pos, age) in player[3].items():
            trees[uid] = (int(pos), int(age), owner)
    return workers, trees


class _Memory:
    def __init__(self, observation, configuration):
        self.cfg = {k: _get(configuration, k, v) for k, v in _DEFAULTS.items()}
        self.geo = _Geometry(int(self.cfg["size"]))
        self.player = int(_get(observation, "player"))
        self.last_step = -1
        self.previous = None
        self.goals = {}
        self.carbon = [self.prior(0)] * self.geo.count
        self.known = [False] * self.geo.count
        self.pristine = [True] * self.geo.count
        for player in _get(observation, "players"):
            for p in player[1].values():
                self.carbon[p] = 0.0
                self.known[p] = True
                self.pristine[p] = False

    def prior(self, step):
        c = self.cfg
        half = math.ceil(c["size"] / 2)
        if c["startingCellCarbon"] <= 0 or c["startingCarbon"] <= 0:
            return 0.0
        cells = max(math.ceil(half * half * c["carbonCoverage"] / 100),
                    math.ceil(c["startingCarbon"] / c["startingCellCarbon"] / 4))
        cells = min(half * half, max(1, cells))
        hi = max(1.0, 2 * min(c["startingCarbon"] / (4 * cells), c["startingCellCarbon"]))
        scale = (1 + c["regenRate"]) ** step
        cap = c["maxCellCarbon"]
        split = min(hi, max(0.0, cap / max(scale, 1e-12)))
        # Expected capped carbon, rather than capping its expectation.
        return cells / (half * half) * (scale * split * split / 2 + cap * (hi - split)) / hi

    def update(self, observation):
        step = int(_get(observation, "step", 0))
        c, g = self.cfg, self.geo
        workers, trees = _assets(observation)
        bases = {p for player in _get(observation, "players") for p in player[1].values()}
        old_workers, old_trees = self.previous or ({}, {})
        if self.previous is not None and step == self.last_step + 1:
            est = self.carbon[:]
            tree_positions = {t[0] for t in trees.values()}
            adjacent = [0] * g.count
            active_adjacent = [0] * g.count
            for tid, (pos, age, owner) in trees.items():
                for j in g.neighbors[pos]:
                    adjacent[j] += 1
                    if tid in old_trees:
                        active_adjacent[j] += 1
                est[pos] = 0.0
                self.pristine[pos] = False
            stationary = {}
            for uid, w in workers.items():
                if uid not in old_workers or old_workers[uid][0] == w[0]:
                    stationary[w[0]] = (uid, w)
            withered = {pos for tid, (pos, age, _) in old_trees.items()
                        if tid not in trees and age >= c["treeLifespan"]}
            for p in range(g.count):
                if p in tree_positions or p in bases:
                    est[p] = 0.0
                    continue
                if p in withered:
                    est[p] = float(c["co2FrmWithered"])
                    self.known[p] = True
                stationary_worker = stationary.get(p)
                if stationary_worker is not None:
                    uid, w = stationary_worker
                    if w[2]:
                        count = sum(v[2] and v[3] == w[3] for v in workers.values())
                        rate = max(0.0, c["initialCollectRate"] - count * c["collectDecreaseRate"])
                        # Enemy cargo reveals collection only without tree tax or collisions.
                        delta = w[1] - old_workers.get(uid, w)[1]
                        aura = any(owner != w[3] and tp in g.eight[p]
                                   for tp, _, owner in trees.values())
                        if w[3] != self.player and delta > 0 and rate > 0 and not aura:
                            inferred = min(c["maxCellCarbon"], delta / rate)
                            if self.pristine[p]:
                                for j in g.mirrors[p]:
                                    if j != p and self.pristine[j] and not self.known[j]:
                                        est[j] = min(c["maxCellCarbon"], inferred * (1 + c["regenRate"]))
                                        self.known[j] = True
                            est[p] = inferred * (1 - rate)
                            self.known[p] = True
                        else:
                            est[p] *= 1 - rate
                    else:
                        est[p] *= max(0.0, 1 - active_adjacent[p] * c["cellAbsorptionRate"])
                    self.pristine[p] = False
                elif adjacent[p]:
                    est[p] *= max(0.0, 1 - active_adjacent[p] * c["cellAbsorptionRate"])
                    self.pristine[p] = False
                elif p not in withered:
                    est[p] = min(c["maxCellCarbon"], est[p] * (1 + c["regenRate"]))
                if p in withered:
                    self.pristine[p] = False
            prior = self.prior(step)
            for p in range(g.count):
                if self.pristine[p] and not self.known[p]:
                    est[p] = prior
            self.carbon = est
        visible = set()
        for pos, _, collector, owner in workers.values():
            if owner == self.player:
                if collector:
                    visible.add(pos)
                else:
                    visible.update(g.neighbors[pos])
        observed = _get(observation, "carbon")
        for p in sorted(visible):
            self.carbon[p] = max(0.0, float(observed[p]))
            self.known[p] = True
            if self.pristine[p]:
                for j in g.mirrors[p]:
                    if j not in visible and self.pristine[j] and not self.known[j]:
                        self.carbon[j] = self.carbon[p]
                        self.known[j] = True
        self.last_step = step
        self.previous = (workers, trees)
        return workers, trees, old_workers


def _assignment(rows):
    """Maximum-weight rectangular matching, with no dependency on SciPy."""
    if not rows:
        return []
    cells = sorted({cell for row in rows for cell in row})
    n, m = len(rows), len(cells)
    u, v, match, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        match[0] = i
        j0, minimum, used = 0, [float("inf")] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = match[j0], float("inf"), 0
            row = rows[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = -row.get(cells[j - 1], -1e12) - u[i0] - v[j]
                if cur < minimum[j]:
                    minimum[j], way[j] = cur, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
    answer = [None] * n
    for j in range(1, m + 1):
        if match[j]:
            answer[match[j] - 1] = cells[j - 1]
    return answer


class _Turn:
    def __init__(self, memory, observation):
        self.memory, self.c, self.g = memory, memory.cfg, memory.geo
        self.me = memory.player
        self.step = int(_get(observation, "step", 0))
        self.left = max(0, int(self.c["episodeSteps"]) - 1 - self.step)
        self.players = _get(observation, "players")
        self.workers, self.trees, self.old_workers = memory.update(observation)
        self.cash = float(self.players[self.me][0])
        self.bases = list(self.players[self.me][1].items())
        self.home = {p for _, p in self.bases}
        self.enemy_bases = {p for i, pl in enumerate(self.players) if i != self.me for p in pl[1].values()}
        self.all_bases = self.home | self.enemy_bases
        self.mine = [(uid, *w[:3]) for uid, w in self.workers.items() if w[3] == self.me]
        self.collectors = [w for w in self.mine if w[3]]
        self.planters = [w for w in self.mine if not w[3]]
        self.en_col = [(uid, *w[:2]) for uid, w in self.workers.items() if w[3] != self.me and w[2]]
        self.en_pl = [(uid, w[0]) for uid, w in self.workers.items() if w[3] != self.me and not w[2]]
        self.at_tree = {p: (owner, age) for p, age, owner in self.trees.values()}
        self.tree_adj = [0] * self.g.count
        self.own_adj = [0] * self.g.count
        self.enemy_aura = [0] * self.g.count
        self.enemy_min = [float("inf")] * self.g.count
        self.planter_reach = [0] * self.g.count
        self.col_prob = [[] for _ in range(self.g.count)]
        self.attack = [0.0] * self.g.count
        self.estimate = memory.carbon[:]
        self.distance_home = [min((self.g.distance[p][b] for b in self.home), default=999)
                              for p in range(self.g.count)]
        self.rate = max(0.0, self.c["initialCollectRate"] - len(self.collectors) * self.c["collectDecreaseRate"])
        for p, age, owner in self.trees.values():
            if age >= self.c["treeLifespan"]:
                continue
            for j in self.g.neighbors[p]:
                self.tree_adj[j] += 1
                if owner == self.me:
                    self.own_adj[j] += 1
            if owner != self.me:
                for j in self.g.eight[p]:
                    self.enemy_aura[j] += 1
        alive = sum(age < self.c["treeLifespan"] for _, age, _ in self.trees.values())
        owned = sum(owner == self.me for _, _, owner in self.trees.values())
        self.plant_cost = self.c["plantCost"]
        if owned > self.c["playerTreeProtectiveNumber"]:
            self.plant_cost += self.c["plantCostInflationRatio"] * self.c["plantCostInflationBase"] ** alive
        self.collector_cost = self.c["recCollectorCost"] + self.c["recCollectorIncreaseCost"] * sum(w[2] for w in self.workers.values())
        self.planter_cost = self.c["recPlanterCost"] + self.c["recPlanterIncreaseCost"] * sum(not w[2] for w in self.workers.values())
        self._threats()
        cycles = []
        for p, value in enumerate(self.estimate):
            if p in self.at_tree or p in self.all_bases or value < 3:
                continue
            d = self.distance_home[p]
            cycles.append(max(value * (1 - (1 - self.rate) ** k) / (2 * d + k)
                              for k in (1, 2, 3, 4)))
        cycles.sort(reverse=True)
        top = cycles[:max(3, len(self.collectors))]
        self.opportunity = max(1.0, min(9.0, (sum(top) / len(top) if top else 1.5) * 0.85))
        self.goals = {}
        self.spending = {}
        self.planned_adj = [0] * self.g.count

    def _threats(self):
        D = self.g.distance
        for uid, p, cargo in self.en_col:
            choices = self.g.steps[p]
            enemy_owner = self.workers[uid][3]
            enemy_home = tuple(self.players[enemy_owner][1].values())
            dist = min((D[p][b] for b in enemy_home), default=99)
            previous = self.old_workers.get(uid)
            staying = previous is not None and previous[0] == p
            raw = []
            for a, j in enumerate(choices):
                score = 1.0
                if a == 0:
                    score += (4.5 if staying else 2.0) if self.estimate[p] > 12 and p not in self.at_tree else 0
                if cargo > 80 or self.left <= dist + 3:
                    after = min((D[j][b] for b in enemy_home), default=99)
                    if after < dist:
                        score += 5.0
                for _, our_pos, our_cargo, _ in self.collectors:
                    if j == our_pos and cargo < our_cargo:
                        score += 4.0
                raw.append(score)
                self.enemy_min[j] = min(self.enemy_min[j], cargo * (1 - self.c["moveCost"] if a else 1))
            total = sum(raw)
            for j, score in zip(choices, raw):
                self.col_prob[j].append((cargo, score / total))
        for uid, p in self.en_pl:
            old = self.old_workers.get(uid)
            stays = old is not None and old[0] == p
            tree = self.at_tree.get(p)
            attractive = tree is None or tree[0] == self.me
            probability = 0.72 if attractive and stays else 0.5 if attractive else 0.15
            for j in self.g.steps[p]:
                self.planter_reach[j] += 1
            self.attack[p] += probability * (48 + (45 if tree and tree[0] == self.me else 0))
            if old is not None and old[0] != p:
                for a, j in enumerate(self.g.steps[old[0]]):
                    if a and j == p:
                        self.attack[self.g.steps[p][a]] += 9.0

    def _forecast(self, p, time):
        if time <= 0:
            return self.estimate[p]
        factor = 1 - self.c["cellAbsorptionRate"] * self.tree_adj[p] if self.tree_adj[p] else 1 + self.c["regenRate"]
        return min(self.c["maxCellCarbon"], self.estimate[p] * max(0.0, factor) ** time)

    def _collector_goals(self):
        D, c = self.g.distance, self.c
        military = 0
        for uid, p, cargo, _ in sorted(self.collectors, key=lambda w: (-w[2], w[0])):
            db = self.distance_home[p]
            nearest = min(self.home, key=lambda b: (D[p][b], b)) if self.home else p
            if cargo > 0 and (self.left <= db + 2 or cargo >= 480 or
                              (self.step < 55 and self.cash < 65 and cargo >= 30 and db <= 5) or
                              (cargo >= 160 and db <= 2)):
                self.goals[uid] = ("home", nearest)
                continue
            if cargo < 12 and self.left > db + 8 and military < max(1, len(self.collectors) // 3):
                best = (14.0, None)
                for _, ep in self.en_pl:
                    d = D[p][ep]
                    if d > 4 or ep in self.enemy_bases:
                        continue
                    protected = max((self._tree_income(tp, age, 0) for tp, (owner, age) in self.at_tree.items()
                                     if owner == self.me and D[tp][ep] <= 2), default=0.0)
                    value = (35 + min(100, protected) * 0.9) / (d + 1)
                    if value > best[0]:
                        best = (value, ep)
                if best[1] is not None:
                    self.goals[uid] = ("hunt", best[1])
                    military += 1
                    continue
            route = self._harvest_route(p, cargo)
            if route is not None:
                first, harvested = route
                self.goals[uid] = ("route", first)
                for cell, value in harvested:
                    self.estimate[cell] = min(self.estimate[cell], value)
                continue
            rate, R = self.rate, self.opportunity
            best_score, best_target = (-0.2 if cargo > 0 else -1e9), None
            previous_target = self.memory.goals.get(uid, (None, None))[1]
            for target in range(self.g.count):
                if target in self.all_bases or target in self.at_tree:
                    continue
                d, back = D[p][target], self.distance_home[target]
                if d > 10 or d + back + 1 > self.left:
                    continue
                value = self._forecast(target, d)
                if value < 2:
                    continue
                if d > 0 and self.enemy_min[target] <= cargo:
                    value *= 0.70
                own_penalty = 1 - 0.13 * min(3, self.own_adj[target])
                best_here = -1e9
                for duration in (1, 2, 3, 4, 6):
                    total_time = d + duration + back
                    if total_time > self.left:
                        break
                    gain = value * (1 - (1 - rate) ** duration)
                    extra = total_time - db
                    score = gain * own_penalty - R * extra - 0.006 * cargo * extra
                    score -= (cargo + gain * 0.5) * min(0.6, 0.045 * self.enemy_aura[target])
                    if score > best_here:
                        best_here = score
                if target == previous_target:
                    best_here += 1.8
                if d and not self.memory.known[target]:
                    best_here += 1.2
                # A rich next patch makes multi-stop harvesting preferable to commuting.
                continuation = max((self.estimate[j] * rate - R * (2 + self.distance_home[j] - back)
                                    for j in self.g.neighbors[target] if j not in self.at_tree and j not in self.all_bases),
                                   default=0.0)
                best_here += max(0.0, continuation) * 0.16
                if best_here > best_score:
                    best_score, best_target = best_here, target
            if best_target is not None:
                self.goals[uid] = ("mine", best_target)
                self.estimate[best_target] *= 0.20
            elif cargo > 0:
                self.goals[uid] = ("home", nearest)
            else:
                targets = [j for j in range(self.g.count) if j not in self.at_tree and j not in self.all_bases]
                target = max(targets, key=lambda j: (self.estimate[j] / (2 + D[p][j]), -j), default=p)
                self.goals[uid] = ("mine", target)

    def _harvest_route(self, start, initial_cargo):
        """Compare complete, equal-length harvesting routes using beam search.

        Moving never collects. Every candidate explicitly simulates stationary
        harvests, cargo lost to trees, deposits, and the remaining time to bank it.
        """
        horizon = min(8, self.left)
        if horizon <= 0 or self.rate <= 0:
            return None
        g, c, R = self.g, self.c, self.opportunity
        db = self.distance_home
        forecasts = [[self._forecast(p, t) for p in range(g.count)] for t in range(horizon + 1)]
        # (position, cargo, deposited, depletion, first destination, previous action, risk)
        beam = [(start, initial_cargo, 0.0, (), -1, -1, 0.0)]
        best_value, best = -1e20, None
        opposite = (0, 3, 4, 1, 2)
        for t in range(1, horizon + 1):
            candidates = []
            for pos, cargo, deposited, depletion, first, previous_action, risk in beam:
                amounts = dict(depletion)
                for action, nxt in enumerate(g.steps[pos]):
                    if action and previous_action > 0 and action == opposite[previous_action]:
                        continue
                    if nxt in self.enemy_bases:
                        continue
                    if t + db[nxt] > self.left and cargo > 0:
                        continue
                    q, paid, changed = cargo, deposited, depletion
                    if action:
                        q *= 1 - c["moveCost"]
                    elif pos not in self.at_tree and pos not in self.all_bases:
                        amount = amounts.get(pos, forecasts[t - 1][pos])
                        if amount < 3:
                            continue
                        collected = amount * self.rate
                        q += collected
                        changed_dict = amounts.copy()
                        changed_dict[pos] = amount - collected
                        changed = tuple(changed_dict.items())
                    elif q <= 0:
                        continue
                    if nxt in self.home:
                        paid += max(0.0, q - c["smeltCost"])
                        q = 0.0
                    else:
                        q *= max(0.0, 1 - c["collectorAbsorptionRate"] * self.enemy_aura[nxt])
                    if t + db[nxt] > self.left and q > 0:
                        continue
                    danger = risk + (self._risk(nxt, cargo, True) if t == 1 else 0.0)
                    extra_time = t
                    value = paid + q - initial_cargo - R * extra_time - 0.004 * initial_cargo * extra_time - danger
                    first_dest = nxt if t == 1 else first
                    if t == horizon and value > best_value:
                        best_value, best = value, (first_dest, changed)
                    amount = dict(changed).get(nxt, forecasts[t][nxt])
                    potential = max(0.0, amount * self.rate - R) * 0.5 if nxt not in self.at_tree and nxt not in self.all_bases else 0
                    candidates.append((value + potential, nxt, q, paid, changed, first_dest, action, danger))
            candidates.sort(key=lambda node: node[0], reverse=True)
            beam, seen = [], set()
            for _, pos, q, paid, changed, first, action, danger in candidates:
                signature = (pos, first, int(q / 12), action == 0)
                if signature in seen:
                    continue
                seen.add(signature)
                beam.append((pos, q, paid, changed, first, action, danger))
                if len(beam) >= 36:
                    break
            if not beam:
                break
        return best

    def _tree_income(self, pos, age, travel):
        life = min(self.left - travel, self.c["treeLifespan"] - age - travel)
        if life <= 0:
            return 0.0
        value = 0.0
        r = self.c["cellAbsorptionRate"]
        for j in self.g.neighbors[pos]:
            if j in self.at_tree or j in self.all_bases:
                continue
            count = max(1, self.tree_adj[j])
            value += self._forecast(j, travel) * (1 - max(0, 1 - r * count) ** life) / count
        return value

    def _site(self, pos, travel):
        if pos in self.all_bases:
            return -1e9, 0.0
        c, D = self.c, self.g.distance
        tree = self.at_tree.get(pos)
        if tree is not None and tree[1] + travel >= c["treeLifespan"]:
            tree = None
        if tree is not None:
            if tree[0] == self.me:
                return -1e9, 0.0
            value = 1.85 * self._tree_income(pos, tree[1], travel)
            cost = c["seizeCost"]
        else:
            life = min(c["treeLifespan"] - 1, self.left - travel - 1)
            if life <= 0:
                return -1e9, self.plant_cost
            value = 0.0
            r = c["cellAbsorptionRate"]
            for j in self.g.neighbors[pos]:
                if j in self.at_tree or j in self.all_bases:
                    continue
                count = self.tree_adj[j] + self.planned_adj[j]
                own = self.own_adj[j] + self.planned_adj[j]
                before = 1 - max(0.0, 1 - r * count) ** life if count else 0
                after = 1 - max(0.0, 1 - r * (count + 1)) ** life
                own_gain = after * (own + 1) / (count + 1) - (before * own / count if count else 0)
                denied = (before / count - after / (count + 1)) * (count - own) if count else 0
                value += self._forecast(j, travel) * (own_gain + 0.85 * denied)
            value -= 0.35 * self._forecast(pos, travel)
            cost = self.plant_cost
        # Trees also tax enemy cargo in their 3x3 area, including on the tree itself.
        for _, p, cargo in self.en_col:
            dist = min(D[p][j] for j in self.g.eight[pos])
            if dist <= 2 and travel <= 3:
                value += cargo * c["collectorAbsorptionRate"] * max(0.0, 1.6 - 0.4 * (dist + travel))
        closest = min((D[pos][p] for _, p, _ in self.en_col), default=99)
        if travel and closest <= 2:
            value *= 0.65 if closest <= 1 else 0.85
        # Much of a new tree's nominal income would otherwise be harvested by
        # our collectors. Discount it; seizing also denies the opponent income.
        return (value * 0.5 if tree is None else value) - cost, cost

    def _planter_goals(self):
        cash = self.cash
        if (len(self.mine) < min(5, self.c["workerLimit"]) and len(self.collectors) < 4
                and self.left > 50 and cash >= self.collector_cost):
            cash -= self.collector_cost
        for uid, p, _, _ in sorted(self.planters):
            best = (-1e9, None, 0.0)
            old_goal = self.memory.goals.get(uid, (None, None))[1]
            for target in range(self.g.count):
                d = self.g.distance[p][target]
                if d + 2 > self.left:
                    continue
                profit, cost = self._site(target, d)
                if profit < 6:
                    continue
                score = profit / (d + 2.5)
                if target == old_goal:
                    score += 1.2
                if self.enemy_min[target] < float("inf"):
                    score -= 9.0 / (d + 1)
                if score > best[0]:
                    best = (score, target, cost)
            target = best[1]
            if target is not None:
                self.goals[uid] = ("plant", target)
                for j in self.g.neighbors[target]:
                    self.planned_adj[j] += 1
                if target == p and cash >= best[2]:
                    self.spending[uid] = best[2]
                    cash -= best[2]
                elif target == p:
                    self.goals[uid] = ("wait", target)
            else:
                # Park off the bases and away from our collectors' mining patches.
                options = [j for j in range(self.g.count) if j not in self.all_bases]
                target = min(options, key=lambda j: (self.g.distance[p][j] +
                             (0 if j in self.at_tree and self.at_tree[j][0] == self.me else 2) +
                             (30 if self.enemy_min[j] < float("inf") else 0), j), default=p)
                self.goals[uid] = ("park", target)

    def _risk(self, p, cargo, collector):
        if collector:
            probabilities = [prob for enemy_cargo, prob in self.col_prob[p] if enemy_cargo <= cargo + 1e-9]
            probability = min(1.0, sum(probabilities))
            worst = self.enemy_min[p] <= cargo + 1e-9
            cargo_value = 0.0 if p in self.home else cargo * 1.7
            penalty = probability * (45 + cargo_value) + (12 + cargo_value * 0.32 if worst else 0)
            return penalty
        probability = min(1.0, sum(prob for _, prob in self.col_prob[p]))
        return 180 * probability + (65 if self.enemy_min[p] < float("inf") else 0) + self.planter_reach[p] * 7

    def _routes(self, targets, cargo, collector):
        distance = [float("inf")] * self.g.count
        queue = []
        for p in targets:
            distance[p] = 0.0
            heapq.heappush(queue, (0.0, p))
        while queue:
            d, p = heapq.heappop(queue)
            if d != distance[p]:
                continue
            risk = 0.0
            if p not in self.home:
                if collector:
                    if self.enemy_min[p] <= cargo:
                        risk += 8 if cargo > 10 else 1.5
                    risk += min(10, cargo * self.c["collectorAbsorptionRate"] * self.enemy_aura[p] / 12)
                elif self.enemy_min[p] < float("inf"):
                    risk += 15
            if p in self.enemy_bases:
                risk += 50
            nd = d + 1 + risk
            for j in self.g.neighbors[p]:
                if nd < distance[j]:
                    distance[j] = nd
                    heapq.heappush(queue, (nd, j))
        return distance

    def _weights(self, worker):
        uid, p, cargo, collector = worker
        mode, target = self.goals[uid]
        route = self._routes(self.home if mode == "home" else (target,), cargo, collector)
        priority = 1 + min(cargo, 480) / 160 if collector else 1.25
        row = {}
        for action, dest in enumerate(self.g.steps[p]):
            score = -8 * priority * route[dest] - self._risk(dest, cargo, collector)
            if dest in self.enemy_bases:
                score -= 800
            if collector:
                if dest in self.home and cargo > 0:
                    score += min(cargo, 200) * 0.25 + (50 if self.left <= 3 else 0)
                if action == 0:
                    if mode in ("mine", "route") and p == target and p not in self.at_tree and p not in self.all_bases:
                        score += 10
                    elif mode == "hunt" and p == target:
                        score += 5
                    elif p not in self.at_tree and p not in self.all_bases:
                        score += 0.25 * self.memory.carbon[p] * self.rate - 3
                    if mode == "home" and cargo > 0:
                        score -= 9 * priority
                for enemy_id, ep, ec in self.en_col:
                    if dest == ep and cargo < ec * (1 - self.c["moveCost"]):
                        old = self.old_workers.get(enemy_id)
                        held = old is not None and old[0] == ep
                        score += min(220, ec * 0.9 + 30) * (0.68 if held else 0.28)
                if cargo < 60:
                    score += self.attack[dest] * (1 if mode == "hunt" else 0.7)
                if self.left <= self.distance_home[p] and cargo > 0 and self.distance_home[dest] >= self.distance_home[p]:
                    score -= cargo + 100
                score -= cargo * self.c["collectorAbsorptionRate"] * self.enemy_aura[dest] * 1.1
                if self.left <= 2 and dest in self.home and cargo > 0:
                    # Duplicate base slots let all cargo be credited even in a friendly collision.
                    for copy in range(len(self.collectors)):
                        row[dest + copy * self.g.count] = score
            elif action == 0:
                tree = self.at_tree.get(p)
                safe_park = p in self.all_bases or (tree and tree[0] == self.me)
                if uid in self.spending:
                    score += 18
                elif not safe_park:
                    # A no-op can silently buy a bad tree or seize an unprofitable tree.
                    score -= 10000
                else:
                    score += 1
            if dest in self.home and cargo == 0:
                score -= 10 if self.left > 30 and len(self.mine) < self.c["workerLimit"] else 2
            # Multiple action indices can refer to the same cell on very small boards.
            row[dest] = max(row.get(dest, -1e12), score)
        return row

    def play(self):
        if self.left <= 0 or not self.home:
            return {}
        self._collector_goals()
        self._planter_goals()
        workers = sorted(self.mine, key=lambda w: w[0])
        chosen = _assignment([self._weights(w) for w in workers])
        actions, occupied, plant_spend, deposits = {}, set(), 0.0, 0.0
        for worker, slot in zip(workers, chosen):
            uid, p, cargo, collector = worker
            dest = slot % self.g.count
            occupied.add(dest)
            if dest != p:
                action = next(a for a, j in enumerate(self.g.steps[p]) if a and j == dest)
                actions[uid] = _MOVES[action]
            elif not collector:
                tree = self.at_tree.get(p)
                if p not in self.all_bases and (tree is None or tree[0] != self.me):
                    plant_spend += self.c["seizeCost"] if tree else self.plant_cost
            if collector and dest in self.home:
                deposits += max(0, cargo * (1 - self.c["moveCost"] if dest != p else 1) - self.c["smeltCost"])
        self._recruit(actions, occupied, max(0.0, plant_spend - deposits))
        self.memory.goals = self.goals
        return actions

    def _recruit(self, actions, occupied, plant_reserve):
        nc, np = len(self.collectors), len(self.planters)
        total, budget = nc + np, max(0.0, self.cash - plant_reserve)
        wanted_planters = 1 if self.step < 15 else 2
        enemy_tree_count = sum(owner != self.me for _, _, owner in self.trees.values())
        if enemy_tree_count >= 7 and self.step > 45:
            wanted_planters = 3
        wanted_planters = min(wanted_planters, max(0, int(self.c["workerLimit"]) - 2))
        def value(base):
            return sum(self.memory.carbon[p] / (1 + self.g.distance[base][p]) ** 2
                       for p in range(self.g.count) if self.g.distance[base][p] <= 4)
        for uid, pos in sorted(self.bases, key=lambda b: (-value(b[1]), b[0])):
            if total >= self.c["workerLimit"] or pos in occupied:
                continue
            if self.left <= 25:
                continue
            planter = np < wanted_planters and self.left > 45 and (nc >= max(1, np * 2 + 1))
            cost = self.planter_cost if planter else self.collector_cost
            if budget + 1e-9 < cost:
                continue
            # Avoid feeding fresh workers to an adjacent empty enemy collector.
            threatened = self.enemy_min[pos] <= 0
            need_break = any(cargo > 60 and self.g.distance[p][pos] <= 3 for _, p, cargo, _ in self.collectors)
            if threatened and not need_break:
                continue
            if planter and self.enemy_min[pos] < float("inf"):
                continue
            actions[uid] = "RECPLANTER" if planter else "RECCOLLECTOR"
            occupied.add(pos)
            budget -= cost
            total += 1
            if planter:
                np += 1
            else:
                nc += 1


_MEMORIES = {}


def agent(observation, configuration):
    """Return plain {owned_unit_id: legal_action} from a partial observation."""
    player = int(_get(observation, "player", 0))
    step = int(_get(observation, "step", 0))
    memory = _MEMORIES.get(player)
    if (memory is None or step == 0 or step <= memory.last_step or
            int(_get(configuration, "size", 15)) != memory.geo.size):
        memory = _Memory(observation, configuration)
        _MEMORIES[player] = memory
    return _Turn(memory, observation).play()
