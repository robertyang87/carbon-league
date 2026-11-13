"""policy_raid_v0.1 -- Carbon League offensive line "raid" v0.1, on the v0.2 skeleton.

v0.2 changes over v0.1 (see policy_v0_2_report.md; baseline = the v0.1
ablation-R configuration, which the v0.1 round proved to be the best
known setting of this skeleton):
  a. The v0.1 beam-search route planner is REMOVED (ablation showed it
     was a strong negative in adversarial games: static belief map,
     42-step routes, no enemy siphon/hunt accounting). Collector target
     selection reverts to the v1 candidate-pool greedy assignment.
     Kept from v0.1: the return-threshold group (450 / 300 low-cash /
     380 late / 200 early / 180 quick-deposit, -15 per base-distance
     step beyond 5, steps-left hard rule) and the opponent-model return
     calibration (enemy collectors head home at cargo >= 400).
  b. Freeze fix: _choose_dest no longer prices movement with raw torus
     Manhattan distance, which has local minima wherever the only
     descent cell is an (impassable) enemy base -- v0.1 traced loaded
     collectors frozen for whole endgames by it. Distances are now BFS
     distances to the goal over passable cells (enemy bases blocked),
     so every non-goal cell has a strictly descending neighbour.
     Returning collectors additionally may not stay and price distance
     at 1000/step, so siphon/soft danger can delay but never stall a
     homeward descent. Post-hoc instrumentation found two more stall
     mechanisms in the reservation protocol, fixed in the same pass:
     (i) own base cells are never forbidden to a returner -- entering an
     occupied own base resolves by collision into the least-cargo
     friendly collector, which then deposits the combined cargo in
     Phase 3, so the "jam" was pure loss (this is the endgame base-mouth
     consolidation, generalised to all steps); an empty collector is
     also never allowed to park on a base cell, and the fallback target
     never names a base cell or the unit's own cell. (ii) A Phase-A
     stay-collector trickle-collecting (a few carbon/step) on the ONLY
     descent cell of a loaded returner (>= 150) yields: it is demoted
     to a mover when its cell belief is < 40.
  c. Tree warfare module (the 54% loss bucket in the v0.1 diagnosis:
     juzi seized 12.9 of our trees per game, we seized 0.5):
     * DEFENCE (P tree_def_on): enemy planters closing on a valuable
       own tree (proximity <= 2, or the opponent model's rob goal at
       distance <= 5) trigger a guard dispatch -- the nearest free
       planter stands ON the threatened tree. A guard physically
       blocks seizure: an enemy planter stepping onto the guarded cell
       dies in the Phase-2 collision (before the Phase-4 seize) along
       with the guard, and the tree keeps earning for us.
     * OFFENCE (P tree_seize_on): enemy trees are valued with juzi's
       rob_feature formula (neighbour carbon with travel-time sharing
       decay, integrated over remaining life at the 3.75% rate, halved
       when defended, base-proximity bonus) instead of the planting
       metric plant_roi, which systematically undervalued them. A rob
       target must clear value > seize cost 20 + 3 per travel step
       (after the race discount); trees with an enemy planter parked
       on top are not entered (stepping in would be suicide by
       collision with no seize). Recruitment keeps a 20-cash re-seize
       reserve while enemy planters live and we own trees.

Everything else (belief map, opponent model, endgame protocol) is
the v0.1/v1 skeleton unchanged.

raid v0.1 (this file) -- the offensive line.  User directive: stop
only patching holes; attack and take points.  Built on
policy_v0_2.py, changing the offensive mix only (the v0.3 main line
is untouched):
  1. HUNT rewritten: v0.2's 3-step mode-path chase becomes a
     return-route interception.  Loaded enemy collectors (cargo
     >= 200) are predicted home along a danger-aware shortest path
     (juzi's collectors skip cells within one step of a lower-cargo
     enemy collector, so the simulation bends around our current
     units -- and a hunter parked on the route deflects/delays the
     real carrier the same way).  A low-cargo collector of ours is
     sent to a future path cell outside the enemy base ring, timed
     to be the unique least-cargo collector there when the carrier
     arrives: the Phase-2 collision winner captures the whole load.
     Captured hunters are forced home.  Sorties / successes /
     captured carbon are accounted in raid_stats.
  2. SEIZE offence: inside the step 50-150 expansion window, prime
     enemy trees (age <= 25, most of their income integral still
     ahead) get a cheaper travel gate (2.0/step) and 1.5x assignment
     priority; tail trees must clear a 60-value floor.  Outside the
     window the v0.2 rob gates apply unchanged.
  3. HARVEST SWITCH: at step >= 100, if the enemy leads on living
     trees by >= 3 (>= 2 after step 130), latch into AlphaCarbon's
     nemesis shape -- no planting, no tree defence, the roster is
     filled to 10 with collectors, return thresholds x0.8: pure
     deposit throughput for the rest of the game.
  Defence stays exactly the v0.2 baseline (danger soft costs,
  freeze fixes); no new defensive module -- this line measures the
  marginal value of offence.  Ablation switches: P raid_hunt_on /
  raid_seize_on / raid_switch_on.
"""

import numpy as np

# --------------------------------------------------------------------------
# Board geometry (index space: idx = (size-1-y)*size + x, UP=(0,+1) etc.,
# identical to fastsim.py / helpers.py Point mapping)
# --------------------------------------------------------------------------
SIZE = 15
N = SIZE * SIZE
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
_OFFSETS = ((0, 1), (1, 0), (0, -1), (-1, 0))

_xs = np.arange(N) % SIZE
_ys = SIZE - 1 - np.arange(N) // SIZE


def _idx(x, y):
    return (SIZE - 1 - (y % SIZE)) * SIZE + (x % SIZE)


MOVE_NBR = np.zeros((N, 4), dtype=np.int64)
NBR4 = MOVE_NBR
NBR8 = np.zeros((N, 8), dtype=np.int64)
MIRROR = np.zeros((N, 3), dtype=np.int64)
for _i in range(N):
    _x, _y = int(_xs[_i]), int(_ys[_i])
    for _k, (_dx, _dy) in enumerate(_OFFSETS):
        MOVE_NBR[_i, _k] = _idx(_x + _dx, _y + _dy)
    _n8 = []
    for _dx in (-1, 0, 1):
        for _dy in (-1, 0, 1):
            if _dx or _dy:
                _n8.append(_idx(_x + _dx, _y + _dy))
    NBR8[_i] = _n8
    MIRROR[_i] = [_idx(SIZE - 1 - _x, _y), _idx(_x, SIZE - 1 - _y),
                  _idx(SIZE - 1 - _x, SIZE - 1 - _y)]

_dxg = np.abs(_xs[:, None] - _xs[None, :])
_dyg = np.abs(_ys[:, None] - _ys[None, :])
DIST = (np.minimum(_dxg, SIZE - _dxg) + np.minimum(_dyg, SIZE - _dyg)).astype(np.int64)

BASE_CELLS = {_idx(4, 4), _idx(4, 10), _idx(10, 4), _idx(10, 10)}

# --------------------------------------------------------------------------
# Tunable parameters
# --------------------------------------------------------------------------
P = dict(
    unknown_default=20.0,       # belief placeholder for unseen cells (juzi)
    harvest_min_belief=14.0,    # target cells need at least this belief
    stay_collect_belief=15.0,   # stay & collect if cell belief >= this
    return_cargo=450.0,         # normal return-home cargo threshold (v0.1)
    return_cargo_lowcash=300.0, # when cash < 150, cycle back earlier
    return_cargo_late=380.0,    # step > 250
    early_return_cargo=200.0,   # step < 50: fast cycling funds expansion
    quick_deposit_cargo=180.0,  # return anyway if this loaded and base <= 3 away
    return_dist_discount=15.0,  # threshold -15 per base-distance step beyond 5
    target_collectors_mid=7,    # Alpha-style mid composition
    target_collectors_late=9,   # post-150 harvest tilt (Alpha's loss pattern)
    target_planters=2,
    max_my_trees=10,            # planting discipline: inflation punishes more
    expand_stock_ratio=0.33,    # reCurse: expand only if resource stock > 33%
    recruit_stop_step=262,
    plant_stop_step=272,        # no new planting after this (payback horizon)
    endgame_start=270,          # collectors start scheduled returns
    endgame_free_base=280,      # last ~20 steps: base cell anti-collision exempt
    hunt_min_enemy_cargo=170.0,
    hunt_max_own_cargo=80.0,
    # --- raid v0.1 offensive line (ablation switches) ---
    raid_hunt_on=0,             # return-route interception hunting
    raid_seize_on=1,            # prime-tree offensive seizing (steps 50-150)
    raid_switch_on=1,           # step>=100 pure-harvest switch when behind on trees
    raid_hunt_min_cargo=200.0,  # only hunt enemy collectors carrying >= this
    raid_hunt_max_hunters=2,    # concurrent hunters cap
    raid_seize_travel_cost=2.0, # offensive rob travel gate inside the window
    raid_seize_tail_min=60.0,   # non-prime rob value floor inside the window
    # --- v0.2 tree warfare module (ablation switches) ---
    tree_def_on=1,              # guard dispatch for threatened own trees
    tree_seize_on=1,            # juzi-style rob valuation + targeting
    def_value_min=80.0,         # own tree worth guarding above this value
    def_radius=2,               # enemy planter this close = threat
    def_goal_radius=5,          # ... or this close when it is his rob goal
    seize_travel_cost=3.0,      # rob gate: value > 20 + this * travel steps
    seize_stay_value=25.0,      # rob gate when already standing on the tree
)

def _plant_cost(my_tree_count, total_alive_trees):
    """Market plant price, mirroring helpers.py Board.next()."""
    if my_tree_count <= 0:
        return 20.0
    return 20.0 + 5.0 * (1.235 ** total_alive_trees)


class BeliefState:
    """Four-fold-symmetric belief carbon map (juzi's ObservationParser idea,
    re-implemented with numpy; see module docstring)."""

    def __init__(self):
        self.belief = np.full(N, -1.0)
        for b in BASE_CELLS:
            self.belief[b] = 0.0
        self.last_enemy_collectors = {}   # wid -> (pos, cargo)
        self.last_step = -1

    def _mirror_add(self, pos, value, updated):
        for m in MIRROR[pos]:
            if self.belief[m] < 0:
                self.belief[m] = value
                updated[m] = True

    def update(self, obs):
        carbon = obs["carbon"]
        me = obs["player"]
        players = obs["players"]
        my_workers = players[me][2]
        opp_workers = players[1 - me][2]
        belief = self.belief
        updated = np.zeros(N, dtype=bool)

        # My collectors see their own cell; my planters see 4-neighbours.
        for wid, (pos, cargo, typ) in my_workers.items():
            if typ == "COLLECTOR":
                belief[pos] = carbon[pos]
                updated[pos] = True
                self._mirror_add(pos, carbon[pos], updated)
            else:
                for nb in NBR4[pos]:
                    belief[nb] = carbon[nb]
                    updated[nb] = True
                    self._mirror_add(nb, carbon[nb], updated)

        # Enemy collectors that stayed put: cell decays x0.7, and a cargo
        # increment reveals the cell value (inc / 0.3 * 0.7) -- juzi's trick.
        new_enemy = {}
        for wid, (pos, cargo, typ) in opp_workers.items():
            if typ != "COLLECTOR":
                continue
            new_enemy[wid] = (pos, cargo)
            prev = self.last_enemy_collectors.get(wid)
            if prev is not None and prev[0] == pos:
                if belief[pos] > 0:
                    belief[pos] *= 0.7
                inc = cargo - prev[1]
                if inc > 0:
                    belief[pos] = inc / 0.3 * 0.7
                updated[pos] = True
        self.last_enemy_collectors = new_enemy
        # Trees: living tree cell = 0; a tree at age>=50 leaves 30 behind.
        for pdata in players:
            for tid, (pos, age) in pdata[3].items():
                if age < 50:
                    belief[pos] = 0.0
                else:
                    belief[pos] = 30.0 / 1.03
                updated[pos] = True
        # Tree absorption decay on 4-neighbours with no collector standing.
        collector_cells = {v[0] for v in my_workers.values() if v[2] == "COLLECTOR"}
        collector_cells |= {v[0] for v in opp_workers.values() if v[2] == "COLLECTOR"}
        for pdata in players:
            for tid, (pos, age) in pdata[3].items():
                if age <= 1:
                    continue
                for nb in NBR4[pos]:
                    if nb not in collector_cells and belief[nb] > 0:
                        belief[nb] *= (1.0 - 0.0375)
                    updated[nb] = True
        # Regrowth on everything not refreshed this step.
        mask = (belief > 0) & (~updated)
        belief[mask] = np.minimum(belief[mask] * 1.03, 100.0)
        self.last_step = obs["step"]


class BoardView:
    """Parsed per-step view of the (fully visible) entity state."""

    def __init__(self, obs):
        self.step = obs["step"]
        self.me = me = obs["player"]
        players = obs["players"]
        self.cash = float(players[me][0])
        self.opp_cash = float(players[1 - me][0])
        self.my_bases = dict(players[me][1])          # bid -> pos
        self.opp_bases = dict(players[1 - me][1])
        self.my_base_cells = set(self.my_bases.values())
        self.opp_base_cells = set(self.opp_bases.values())
        self.my_workers = players[me][2]              # wid -> [pos, cargo, type]
        self.opp_workers = players[1 - me][2]
        self.trees = {}                               # pos -> (owner, age, tid)
        for pdata_owner, pdata in enumerate(players):
            for tid, (pos, age) in pdata[3].items():
                self.trees[pos] = (pdata_owner, age, tid)
        self.my_tree_count = len(players[me][3])
        self.total_trees = len(self.trees)
        self.my_collectors = {w: v for w, v in self.my_workers.items() if v[2] == "COLLECTOR"}
        self.my_planters = {w: v for w, v in self.my_workers.items() if v[2] == "PLANTER"}
        self.opp_collectors = {w: v for w, v in self.opp_workers.items() if v[2] == "COLLECTOR"}
        self.opp_planters = {w: v for w, v in self.opp_workers.items() if v[2] == "PLANTER"}
        self.worker_at = {}
        for w, v in self.my_workers.items():
            self.worker_at[v[0]] = (w, True)
        for w, v in self.opp_workers.items():
            self.worker_at[v[0]] = (w, False)
        # enemy trees within 8-neighbourhood count per cell (siphon exposure)
        self.opp_tree_near = np.zeros(N, dtype=np.int64)
        self.my_tree_near = np.zeros(N, dtype=np.int64)
        for pos, (owner, age, tid) in self.trees.items():
            tgt = self.opp_tree_near if owner != me else self.my_tree_near
            tgt[pos] += 1
            for nb in NBR8[pos]:
                tgt[nb] += 1
        self.tree_at = np.full(N, -1, dtype=np.int64)  # owner or -1
        for pos, (owner, age, tid) in self.trees.items():
            self.tree_at[pos] = owner
        # trees in the 4-neighbourhood of each cell (juzi four_tree_nums)
        self.tree_around = (self.tree_at[NBR4] >= 0).sum(axis=1)
        # distance from every cell to the nearest own / enemy base
        mb = sorted(self.my_base_cells)
        ob = sorted(self.opp_base_cells)
        self.base_dist = DIST[:, mb].min(axis=1) if mb else np.zeros(N, dtype=np.int64)
        self.opp_base_dist = DIST[:, ob].min(axis=1) if ob else np.zeros(N, dtype=np.int64)
        self.nearest_base = {c: mb[int(np.argmin(DIST[c, mb]))] for c in range(N)} if mb else {}


def compute_plant_value(belief, view):
    """juzi's plant_value_map: sum over 4-neighbours of
    belief/(1 + trees around that neighbour), x1.5 where enemy trees
    dominate the neighbourhood, /1.5 where own trees dominate (only when
    the cell itself is rich)."""
    bv = np.where(belief < 0, P["unknown_default"], belief)
    tree_any = (view.tree_at >= 0).astype(np.float64)
    my_tree = (view.tree_at == view.me).astype(np.float64)
    opp_tree = (view.tree_at == (1 - view.me)).astype(np.float64)
    cnt_around = tree_any[NBR4].sum(axis=1)
    term = bv / (1.0 + cnt_around)
    pv = term[NBR4].sum(axis=1)
    k = my_tree[NBR4].sum(axis=1) - opp_tree[NBR4].sum(axis=1)
    rich = bv > 40
    pv = np.where(rich & (k < 0), pv * 1.5, pv)
    pv = np.where(rich & (k > 0), pv / 1.5, pv)
    for b in BASE_CELLS:
        pv[b] = -1e9
    return pv


def tree_income_estimate(belief, cell, horizon):
    """Total carbon a tree on `cell` absorbs from its 4 neighbours over
    `horizon` steps (neighbours deplete at 3.75%/step, no regen there).
    Raw (undiscounted) neighbour sum -- used for defence valuation."""
    bv = np.where(belief < 0, P["unknown_default"], belief)
    s = float(bv[NBR4[cell]].sum())
    h = max(0, min(horizon, 60))
    return s * (1.0 - (1.0 - 0.0375) ** h)


def rob_value(belief, view, cell, d, steps_left):
    """juzi's rob_feature valuation of an enemy tree at `cell` for a
    planter `d` steps away: each 4-neighbour's carbon is discounted by
    the tree-sharing decay over the travel time, then integrated over
    the tree's remaining life after arrival at the 3.75% absorption
    rate. Halved if an enemy planter is parked on the tree (contested),
    plus juzi's proximity bonus for trees close to our bases. Returns
    -1e18 for trees juzi would not consider (remaining life < 15 or
    too short after the trip)."""
    owner, age, tid = view.trees[cell]
    remain = 50 - age
    if remain < 15 or remain - d < 5:
        return -1e18
    bv = np.where(belief < 0, P["unknown_default"], belief)
    a = 0.0375
    horizon = min(remain - d, steps_left)
    val = 0.0
    for nb in NBR4[cell]:
        nb = int(nb)
        share = max(0.0, 1.0 - a * float(view.tree_around[nb]))
        val += float(bv[nb]) * (share ** d) * (1.0 - (1.0 - a) ** horizon)
    w_at = view.worker_at.get(cell)
    if w_at is not None and not w_at[1] \
            and view.opp_workers[w_at[0]][2] == "PLANTER":
        val *= 0.5
    d_base = int(view.base_dist[cell])
    if 1 <= d_base <= 2:
        val += 35.0 / d_base
    return val


def plant_roi(plant_value, cell, plant_cost, steps_left):
    """Planting ROI using the tree-shared discounted neighbour sum
    (plant_value already divides by 1 + trees around each neighbour, so
    adjacent trees do not double-count the same carbon)."""
    h = max(0, min(50, steps_left))
    return plant_value[cell] * (1.0 - (1.0 - 0.0375) ** h) - plant_cost


class OpponentModel:
    """Rule predictor v1: per enemy unit, a <=5-cell distribution over its
    next position. Enemy visible cells follow deterministically from its
    unit positions (collector: own cell; planter: 4 neighbours), so the
    enemy's information state is approximated by our own belief map."""

    def __init__(self, belief_state):
        self.belief_state = belief_state

    def predict(self, view, plant_value):
        belief = self.belief_state.belief
        bv = np.where(belief < 0, P["unknown_default"], belief)
        attr = bv + 0.5 * bv[NBR4].mean(axis=1)
        out = {}
        opp = 1 - view.me
        opp_base_cells = view.opp_base_cells
        plant_cost = _plant_cost(
            sum(1 for o, a, t in view.trees.values() if o == opp), view.total_trees)
        for wid, (pos, cargo, typ) in view.opp_workers.items():
            cand = {}
            if typ == "COLLECTOR":
                if pos in opp_base_cells and cargo > 0:
                    cand[pos] = 0.80
                    for nb in NBR4[pos]:
                        cand[int(nb)] = 0.05
                elif cargo >= 400 or (cargo >= 200 and view.step >= 250):
                    # returning: shortest-path first steps toward nearest base.
                    # Threshold calibrated on replay validation (see report):
                    # top players keep collecting until ~400 cargo, so a
                    # 150-400 cargo collector is usually still harvesting.
                    cand[pos] = 0.08
                    best = None
                    steps = []
                    for b in sorted(opp_base_cells):
                        d = DIST[pos, b]
                        if best is None or d < best:
                            best, steps = d, [int(nb) for nb in NBR4[pos] if DIST[nb, b] == d - 1]
                    share = 0.84 / max(1, len(steps))
                    for nb in steps:
                        cand[nb] = cand.get(nb, 0.0) + share
                    for nb in NBR4[pos]:
                        cand.setdefault(int(nb), 0.02)
                elif bv[pos] >= 24:
                    cand[pos] = 0.62
                    wts = {int(nb): max(attr[nb], 0.01) for nb in NBR4[pos]}
                    tot = sum(wts.values())
                    for nb, w in wts.items():
                        cand[nb] = 0.30 * w / tot
                    cand[pos] += 0.08
                else:
                    cand[pos] = 0.20
                    wts = {int(nb): max(attr[nb], 0.01) for nb in NBR4[pos]}
                    tot = sum(wts.values())
                    for nb, w in wts.items():
                        cand[nb] = 0.80 * w / tot
            else:  # PLANTER
                tree_here = view.trees.get(pos)
                if tree_here is not None and tree_here[0] == view.me and view.opp_cash >= 20:
                    cand[pos] = 0.80  # seizing our tree
                    for nb in NBR4[pos]:
                        cand[int(nb)] = 0.05
                elif tree_here is None and pos not in view.my_base_cells \
                        and pos not in opp_base_cells \
                        and plant_value[pos] > 60 and view.opp_cash >= plant_cost:
                    cand[pos] = 0.68  # planting here
                    for nb in NBR4[pos]:
                        cand[int(nb)] = 0.08
                else:
                    cand[pos] = 0.24
                    goal = self._planter_goal(view, plant_value, pos)
                    if goal is not None and goal != pos:
                        steps = [int(nb) for nb in NBR4[pos]
                                 if DIST[nb, goal] == DIST[pos, goal] - 1]
                        share = 0.72 / max(1, len(steps))
                        for nb in steps:
                            cand[nb] = cand.get(nb, 0.0) + share
                        for nb in NBR4[pos]:
                            cand.setdefault(int(nb), 0.01)
                    else:
                        for nb in NBR4[pos]:
                            cand[int(nb)] = cand.get(int(nb), 0.0) + 0.19
            tot = sum(cand.values()) or 1.0
            items = sorted(((c, p / tot) for c, p in cand.items()),
                           key=lambda kv: -kv[1])[:5]
            out[wid] = items
        return out

    def _planter_goal(self, view, plant_value, pos):
        # best enemy rob target = our trees with high remaining value;
        # else best plant cell by plant value / distance.
        best, best_score = None, -1e18
        for tpos in sorted(view.trees):
            owner, age, tid = view.trees[tpos]
            if owner != view.me:
                continue
            remain = 50 - age
            if remain < 8:
                continue
            score = tree_income_estimate(self.belief_state.belief, tpos, remain) \
                / (1.0 + DIST[pos, tpos])
            if score > best_score:
                best, best_score = tpos, score
        score = plant_value / (1.0 + DIST[pos])
        cand = int(np.argmax(score))
        if score[cand] > best_score:
            best = cand
        return best

    def mode_path(self, view, plant_value, wid, steps=3):
        """Deterministic argmax trajectory for one enemy unit (used for
        interception planning)."""
        pos, cargo, typ = view.opp_workers[wid]
        path = []
        cur = pos
        fake = _FakeView(view, wid, cur, cargo, typ)
        for _ in range(steps):
            preds = self.predict(fake, plant_value)
            items = preds.get(wid) or [(cur, 1.0)]
            cur = items[0][0]
            path.append(cur)
            fake = _FakeView(view, wid, cur, cargo, typ)
        return path


class _FakeView:
    """Minimal view clone with one enemy worker relocated (for mode_path)."""

    def __init__(self, view, wid, pos, cargo, typ):
        self.__dict__.update(view.__dict__)
        ow = dict(view.opp_workers)
        ow[wid] = [pos, cargo, typ]
        self.opp_workers = ow


# --------------------------------------------------------------------------
# Main policy
# --------------------------------------------------------------------------
class Policy:
    def __init__(self):
        self.belief_state = BeliefState()
        self.opponent = OpponentModel(self.belief_state)
        self.last_step = -1
        self._bfs_cache = {}
        # raid v0.1 state
        self.raid_stats = dict(hunt_attempts=0, hunt_success=0,
                               hunt_captured=0.0, hunt_hunter_deaths=0,
                               hunt_escaped=0, hunt_both_died=0,
                               switch_step=-1)
        self.active_hunts = {}    # hunter wid -> hunt record
        self.must_return = set()  # hunters that captured: forced home
        self.switched = False     # harvest-switch latch

    def _goal_dist(self, view, goal):
        """BFS distance-to-`goal` over passable cells (enemy base cells
        blocked), cached per act() call. Unlike raw torus Manhattan
        distance this has no local minima: every reachable non-goal
        cell has a neighbour one step closer."""
        if goal in view.opp_base_cells:
            return DIST[:, goal]
        cached = self._bfs_cache.get(goal)
        if cached is not None:
            return cached
        from collections import deque
        dist = np.full(N, 10 ** 6, dtype=np.int64)
        dist[goal] = 0
        dq = deque([goal])
        blocked = view.opp_base_cells
        while dq:
            c = dq.popleft()
            dc = dist[c] + 1
            for nb in NBR4[c]:
                nb = int(nb)
                if dist[nb] == 10 ** 6 and nb not in blocked:
                    dist[nb] = dc
                    dq.append(nb)
        self._bfs_cache[goal] = dist
        return dist

    @staticmethod
    def _action_for(pos, dest):
        if dest is None or dest == pos:
            return None
        for k in range(4):
            if int(NBR4[pos, k]) == dest:
                return ACTIONS[k]
        return None

    def _choose_dest(self, view, pos, goal, cargo, forbidden, can_stay,
                     soft, siphon_scale=1.0, dist_weight=10.0,
                     descent_first=False):
        """Pick a destination cell minimising dist_weight*BFS-dist-to-goal
        + siphon exposure + soft danger, among cells not in `forbidden`
        (stay only if can_stay). Falls back to the cheapest forbidden
        cell only when no legal option exists. BFS distances (enemy
        bases blocked) replace Manhattan distance, whose plateaus next
        to enemy bases froze loaded collectors in v0.1."""
        goal_dist = self._goal_dist(view, goal)
        options = []
        cells = [int(c) for c in NBR4[pos]]
        if can_stay:
            cells.append(pos)
        legal, illegal = [], []
        for cell in cells:
            if cell in view.opp_base_cells and cell != pos:
                continue  # never step on an enemy base (cargo donates / death)
            cost = dist_weight * float(goal_dist[cell])
            cost += siphon_scale * float(view.opp_tree_near[cell]) * cargo * 0.08
            if cell in soft:
                cost += 30.0
            (illegal if cell in forbidden else legal).append((cost, cell))
        if descent_first and legal:
            # staying is only allowed when no legal step descends
            gd_pos = float(goal_dist[pos])
            if any(cell != pos and float(goal_dist[cell]) < gd_pos
                   for _, cell in legal):
                legal = [e for e in legal if e[1] != pos]
        pool = legal if legal else illegal
        if not pool:
            return pos
        pool.sort()
        return pool[0][1]

    def _predict_return_path(self, view, ewid, max_steps=12):
        """Simulate a loaded enemy collector heading home, juzi-style:
        danger-aware descent.  At each step it moves to a neighbour
        one step closer to its chosen base, skipping cells within one
        step of a lower-cargo enemy collector (us) -- exactly juzi's
        is_dangerous(); it waits when no safe descent exists.  Our
        units are frozen at their current cells for the simulation;
        the plan is rebuilt every step, so hunter motion folds in on
        the next call.  The home base is the nearest one, tie-broken
        by fewer of our collectors camped within 2 steps (juzi's
        home_weight counts nearby enemy workers the same way)."""
        pos, cargo, _ = view.opp_workers[ewid]
        bases = sorted(view.opp_base_cells)
        if not bases:
            return []

        def base_key(b):
            near = sum(1 for (wp, wc, wt) in view.my_collectors.values()
                       if int(DIST[wp, b]) <= 2)
            return (int(DIST[pos, b]), near, b)

        home = min(bases, key=base_key)
        danger = set()
        for (wp, wc, wt) in view.my_collectors.values():
            if wc < cargo:
                danger.add(wp)
                for nb in NBR4[wp]:
                    danger.add(int(nb))
        path = []
        cur = pos
        for _ in range(max_steps):
            if cur == home:
                break
            cands = [int(nb) for nb in NBR4[cur]
                     if nb not in danger or nb == home]
            if not cands:
                nxt = cur
            else:
                dmin = min(int(DIST[c, home]) for c in cands)
                if dmin >= int(DIST[cur, home]):
                    nxt = cur   # no safe descent: it waits / wanders
                else:
                    nxt = min(c for c in cands if int(DIST[c, home]) == dmin)
            path.append(nxt)
            cur = nxt
            if cur == home:
                break
        return path

    def _reconcile_hunts(self, view):
        """Settle active hunt sorties against the freshly observed
        state (called once per act, before planning).  A sortie is a
        success only when the target died during the last transition
        AND our hunter's cargo jumped by >= 60% of the target's last
        seen load -- conservative attribution, no credit for kills we
        did not make."""
        self.must_return = {w for w in self.must_return
                            if w in view.my_collectors
                            and view.my_collectors[w][1] > 0}
        if not self.active_hunts:
            return
        stats = self.raid_stats
        for hwid in list(self.active_hunts):
            rec = self.active_hunts[hwid]
            ewid = rec["enemy"]
            hunter = view.my_collectors.get(hwid)
            enemy = view.opp_collectors.get(ewid)
            if enemy is None:
                if hunter is not None and hunter[1] >= \
                        rec["hunter_cargo"] + 0.6 * rec["enemy_cargo"]:
                    stats["hunt_success"] += 1
                    gained = hunter[1] - rec["hunter_cargo"]
                    stats["hunt_captured"] += min(
                        gained, rec["enemy_cargo"] + 40.0)
                    self.must_return.add(hwid)
                elif hunter is None:
                    stats["hunt_both_died"] += 1
                del self.active_hunts[hwid]
                continue
            if hunter is None:
                stats["hunt_hunter_deaths"] += 1
                del self.active_hunts[hwid]
                continue
            if enemy[1] <= 0.0 or enemy[0] in view.opp_base_cells:
                stats["hunt_escaped"] += 1   # deposited: cargo is gone
                del self.active_hunts[hwid]
                continue
            rec["enemy_cargo"] = enemy[1]
            if view.step - rec["step"] > 30:
                del self.active_hunts[hwid]  # stale sortie, no verdict

    # -- main ---------------------------------------------------------------
    def act(self, obs, configuration):
        step = obs["step"]
        if step == 0 or step < self.last_step:
            self.belief_state = BeliefState()
            self.opponent = OpponentModel(self.belief_state)
            self.raid_stats = dict(hunt_attempts=0, hunt_success=0,
                                   hunt_captured=0.0, hunt_hunter_deaths=0,
                                   hunt_escaped=0, hunt_both_died=0,
                                   switch_step=-1)
            self.active_hunts = {}
            self.must_return = set()
            self.switched = False
        self.last_step = step
        self._bfs_cache = {}
        self.belief_state.update(obs)
        view = BoardView(obs)
        belief = self.belief_state.belief
        bv = np.where(belief < 0, P["unknown_default"], belief)
        plant_value = compute_plant_value(belief, view)
        preds = self.opponent.predict(view, plant_value)
        actions = {}
        me = view.me
        steps_left = 300 - step
        plant_cost = _plant_cost(view.my_tree_count, view.total_trees)
        endgame = step >= P["endgame_start"]
        free_base = step >= P["endgame_free_base"]

        collectors = sorted(view.my_collectors.items(), key=lambda kv: (-kv[1][1], kv[0]))
        planters = sorted(view.my_planters.items(), key=lambda kv: (-plant_value[kv[1][0]], kv[0]))
        self._reconcile_hunts(view)
        # raid v0.1 harvest switch: behind on living trees at step
        # >= 100 -> latch into the pure-harvest shape (see docstring).
        if P["raid_switch_on"] and not self.switched and step >= 100:
            opp_trees = view.total_trees - view.my_tree_count
            if opp_trees - view.my_tree_count >= 3 or \
                    (step >= 130 and opp_trees - view.my_tree_count >= 2):
                self.switched = True
                self.raid_stats["switch_step"] = step

        # Danger sets per cargo level (cargo protection, alpha_analysis #5):
        # an enemy collector carrying less than me wins a collision against
        # me, so every cell it can reach next step is hard danger; its
        # predicted destinations are soft danger.
        def danger_for(cargo):
            hard, soft = set(), set()
            for wid, (epos, ecargo, etyp) in view.opp_collectors.items():
                if ecargo < cargo:
                    hard.add(epos)
                    for nb in NBR4[epos]:
                        hard.add(int(nb))
                    for cell, prob in preds.get(wid, []):
                        if prob >= 0.25:
                            soft.add(cell)
            return hard, soft

        # ================= Phase A: stationary units commit first ========
        dest_of = {}
        reserved = set()      # committed friendly destinations (incl. stays)

        # Per-collector return decision (needs own base distance).
        returning = {}
        for wid, (pos, cargo, _) in collectors:
            if pos in view.my_base_cells and cargo > 0:
                returning[wid] = True   # will stay & deposit in phase A
                continue
            ret = P["return_cargo_late"] if step > 250 else (
                P["return_cargo_lowcash"] if view.cash < 150 else (
                    P["early_return_cargo"] if step < 50 else P["return_cargo"]))
            # v0.1 distance adjustment: the trip home earns nothing, so a
            # far collector heads back with a lighter load.
            ret -= P["return_dist_discount"] * max(0, int(view.base_dist[pos]) - 5)
            ret = max(ret, 50.0)
            if self.switched:
                ret = max(ret * 0.8, 50.0)   # faster cycling, pure harvest
            if cargo >= P["quick_deposit_cargo"] and view.base_dist[pos] <= 3:
                ret = min(ret, P["quick_deposit_cargo"])
            r = cargo >= ret
            # v0.1 steps-left hard rule: if the trip home must start now
            # for the cargo to land at all, start it (any step, any load).
            if cargo > 0 and steps_left <= int(view.base_dist[pos]) + 1:
                r = True
            if endgame and cargo > 0:
                r = True
            if step >= 292 and cargo > 0:
                r = True
            if wid in self.must_return and cargo > 0:
                r = True   # raid: a hunter that captured goes straight home
            returning[wid] = r

        stayers = set()
        for wid, (pos, cargo, _) in collectors:
            if pos in view.my_base_cells and cargo > 0:
                stayers.add(wid)                       # deposit
            elif not returning[wid] and bv[pos] >= P["stay_collect_belief"] \
                    and view.tree_at[pos] < 0:
                hard, _ = danger_for(cargo) if cargo >= 60 else (set(), set())
                if pos not in hard:
                    stayers.add(wid)                   # stay & collect
        # ================= v0.2 tree warfare: threat map ==============
        # defend_assign: my planter wid -> own tree cell it must guard.
        # A tree is threatened when an enemy planter is within
        # def_radius steps, or within def_goal_radius when the opponent
        # model's rob goal for that planter is exactly this tree.
        defend_assign = {}
        if P["tree_def_on"] and not self.switched and planters and view.opp_planters:
            opp_goals = {ewid: self.opponent._planter_goal(view, plant_value, epos)
                         for ewid, (epos, _, _) in view.opp_planters.items()}
            threats = []   # (enemy arrival steps, -tree value, tree pos)
            for tpos, (owner, age, tid) in view.trees.items():
                if owner != me:
                    continue
                val = tree_income_estimate(belief, tpos, 50 - age)
                if val <= P["def_value_min"]:
                    continue
                arrival = None
                for ewid, (epos, _, _) in view.opp_planters.items():
                    d = int(DIST[epos, tpos])
                    if d <= P["def_radius"] or \
                            (opp_goals.get(ewid) == tpos and d <= P["def_goal_radius"]):
                        arrival = d if arrival is None else min(arrival, d)
                if arrival is not None:
                    threats.append((arrival, -val, tpos))
            threats.sort()
            guardable = [w for w, _ in planters
                         if not (view.trees.get(view.my_planters[w][0]) is not None
                                 and view.trees[view.my_planters[w][0]][0] != me)]
            for _arr, _nv, tpos in threats:
                best_w, best_d = None, None
                for w in guardable:
                    if w in defend_assign:
                        continue
                    d = int(DIST[view.my_planters[w][0], tpos])
                    if best_d is None or d < best_d:
                        best_w, best_d = w, d
                if best_w is not None:
                    defend_assign[best_w] = tpos

        planter_stay = {}
        for wid, (pos, cargo, _) in planters:
            tree_here = view.trees.get(pos)
            stay = False
            if tree_here is not None and tree_here[0] != me:
                # standing on an enemy tree: seize when juzi's rob
                # valuation clears the cost (v0.2 module; off = never)
                if P["tree_seize_on"] and view.cash >= 20:
                    if rob_value(belief, view, pos, 0, steps_left) \
                            > P["seize_stay_value"]:
                        stay = True                        # seize
            elif wid in defend_assign and defend_assign[wid] == pos:
                stay = True                                # guard post
            elif wid in defend_assign:
                stay = False   # must move to its guard post this step
            elif tree_here is None and pos not in view.my_base_cells \
                    and pos not in view.opp_base_cells \
                    and step < P["plant_stop_step"] and not self.switched \
                    and view.my_tree_count < P["max_my_trees"]:
                roi = plant_roi(plant_value, pos, plant_cost, steps_left)
                if plant_value[pos] >= 70 and roi > max(12.0, 0.5 * plant_cost) \
                        and view.cash >= plant_cost:
                    stay = True                        # plant
            elif tree_here is not None and tree_here[0] == me:
                if P["tree_def_on"]:
                    val = tree_income_estimate(belief, pos, 50 - tree_here[1])
                    threat = any(DIST[view.opp_planters[w][0], pos] <= 1
                                 for w in view.opp_planters)
                    if val > 120 and threat:
                        stay = True                        # defend
            planter_stay[wid] = stay
            if stay:
                stayers.add(wid)
        # v0.2 yield pass: a stay-collector trickle-collecting on the
        # ONLY descent cell of a loaded returner yields the corridor
        # (traced in seed 2000: 442-cargo returner stalled 15 steps
        # behind a collector gaining 1-4 carbon/step). Demoted stayers
        # become movers with can_stay=False (see Phase B).
        yielders = set()
        for wid, (pos, cargo, _) in collectors:
            if not returning.get(wid) or cargo < 150:
                continue
            gd = self._goal_dist(view, view.nearest_base[pos])
            gd_pos = int(gd[pos])
            if gd_pos == 0:
                continue
            descent = [int(c) for c in NBR4[pos] if int(gd[c]) == gd_pos - 1]
            if not descent or any(c in view.my_base_cells for c in descent):
                continue
            blockers = [c for c in descent if c in reserved]
            if len(blockers) != len(descent):
                continue   # a free descent cell exists
            for c in blockers:
                occ = view.worker_at.get(c)
                if occ is None or not occ[1]:
                    continue
                owid = occ[0]
                if owid in stayers and owid not in yielders \
                        and view.my_workers[owid][2] == "COLLECTOR" \
                        and bv[c] < 40.0:
                    yielders.add(owid)
                    stayers.discard(owid)
                    reserved.discard(c)
                    break
        for wid in sorted(stayers):
            pos = view.my_workers[wid][0]
            dest_of[wid] = pos
            reserved.add(pos)

        # ================= assignments for movers ========================
        # ============ raid v0.1: return-route interception ==========
        # (replaces v0.2's 3-step mode-path chase)  For every loaded
        # enemy collector, predict its danger-aware route home and
        # send a low-cargo collector of ours to a future path cell,
        # timed to stand there as the unique least-cargo collector
        # when the carrier arrives (Phase-2 winner captures all).
        # Ambush cells stay out of the enemy base cells themselves
        # (kills there pay the base owner, not us); a 0.0-cargo
        # hunter additionally avoids the base-exit ring (distance 1),
        # where a freshly emptied enemy collector could tie it into
        # mutual annihilation.
        hunt_target = {}
        if P["raid_hunt_on"] and step >= 8 \
                and step < P["endgame_start"] - 10 and collectors:
            hunters_avail = [w for w, _ in collectors
                             if w not in stayers and not returning[w]
                             and view.my_collectors[w][1] <= P["hunt_max_own_cargo"]]
            # fractional cargo first (tie-safety), then lightest
            hunters_avail.sort(key=lambda w: (view.my_collectors[w][1] == 0.0,
                                              view.my_collectors[w][1], w))
            foes = sorted(view.opp_collectors.items(),
                          key=lambda kv: (-kv[1][1], kv[0]))
            n_hunts = 0
            for ewid, (epos, ecargo, _) in foes:
                if n_hunts >= P["raid_hunt_max_hunters"] or not hunters_avail:
                    break
                if ecargo < P["raid_hunt_min_cargo"]:
                    continue
                path = self._predict_return_path(view, ewid)
                best = None
                for wid in hunters_avail:
                    hpos, hcargo, _ = view.my_collectors[wid]
                    if hcargo > ecargo - 50:
                        continue
                    for t, cell in enumerate(path, start=1):
                        if cell in view.opp_base_cells or cell in reserved:
                            continue
                        occ = view.worker_at.get(cell)
                        if occ is not None and occ[1]:
                            continue   # a friendly unit already sits there
                        d = int(DIST[hpos, cell])
                        if d > t:
                            continue   # cannot arrive by the carrier's step
                        slack = t - d
                        score = ecargo * (0.93 ** t) - 30.0 * t + 12.0 * slack
                        if int(view.opp_base_dist[cell]) >= 2:
                            score += 25.0        # tie/spill-safe ring
                        elif hcargo == 0.0:
                            continue             # base-exit ring: tie risk
                        if best is None or score > best[0]:
                            best = (score, wid, cell)
                if best is not None:
                    _, wid, cell = best
                    hunt_target[wid] = cell
                    hunters_avail.remove(wid)
                    n_hunts += 1
                    rec = self.active_hunts.get(wid)
                    if rec is None or rec["enemy"] != ewid:
                        self.raid_stats["hunt_attempts"] += 1
                        self.active_hunts[wid] = {
                            "enemy": ewid, "enemy_cargo": ecargo,
                            "hunter_cargo": view.my_collectors[wid][1],
                            "step": step, "cell": cell}
                    else:
                        rec["cell"] = cell

        # Harvest targets: candidate pool + greedy global assignment with
        # per-target load discount (reCurse / juzi share-discounting) --
        # v0.2 restores the v1 logic here (v0.1's beam planner removed;
        # see policy_v0_1_report.md ablation R).
        attr = bv + 0.5 * bv[NBR4].mean(axis=1)
        cand_cells = [int(c) for c in np.argsort(-attr, kind="stable")[:40]
                      if bv[c] >= P["harvest_min_belief"] and c not in view.opp_base_cells]
        assigned = {}
        target_load = {}
        remaining = [w for w, _ in collectors
                     if w not in stayers and w not in hunt_target and not returning[w]]
        rem_set = list(remaining)
        while rem_set:
            best = None
            for wid in rem_set:
                pos, cargo, _ = view.my_collectors[wid]
                for c in cand_cells:
                    load = target_load.get(c, 0)
                    est = bv[c] * 0.76 * (0.35 ** load)
                    if view.opp_tree_near[c] > 0:
                        est *= 0.8
                    cycle = DIST[pos, c] + 4.0 + 0.6 * float(view.base_dist[c])
                    score = est * 100.0 / max(cycle, 1.0) - 6.0 * float(view.opp_tree_near[c])
                    if best is None or score > best[0]:
                        best = (score, wid, c)
            if best is None or best[0] <= 0:
                break
            _, wid, c = best
            assigned[wid] = c
            target_load[c] = target_load.get(c, 0) + 1
            rem_set.remove(wid)

        # Planter targets: guard posts first (v0.2 defence), then plant
        # cells / rob targets, globally assigned.
        plant_cands = [int(c) for c in np.argsort(-plant_value, kind="stable")[:24]]
        rob_cands = []
        if P["tree_seize_on"]:
            for tpos, (owner, age, tid) in view.trees.items():
                if owner == me or 50 - age < 15:
                    continue
                w_at = view.worker_at.get(tpos)
                if w_at is not None and not w_at[1]:
                    continue   # enemy worker parked on it: entry = suicide
                rob_cands.append(tpos)
        p_assigned = {}
        p_rem = [w for w, _ in planters if w not in stayers]
        for w in list(p_rem):
            if w in defend_assign:
                p_assigned[w] = defend_assign[w]
                p_rem.remove(w)
        while p_rem:
            best = None
            for wid in p_rem:
                pos = view.my_planters[wid][0]
                if step < P["plant_stop_step"] and not self.switched \
                        and view.my_tree_count < P["max_my_trees"]:
                    for c in plant_cands:
                        if view.tree_at[c] >= 0 or c in view.my_base_cells \
                                or c in view.opp_base_cells or c in p_assigned.values():
                            continue
                        roi = plant_roi(plant_value, c, plant_cost, steps_left)
                        if roi <= max(12.0, 0.5 * plant_cost):
                            continue
                        score = roi * 100.0 / (4.0 + DIST[pos, c])
                        if best is None or score > best[0]:
                            best = (score, wid, c)
                for c in rob_cands:
                    if c in p_assigned.values():
                        continue
                    d = int(DIST[pos, c])
                    val = rob_value(belief, view, c, d, steps_left)
                    if val <= 0:
                        continue   # includes juzi's remain/trip filters
                    for ewid, (epos, _, _) in view.opp_planters.items():
                        if DIST[epos, c] <= d:
                            val *= 0.5     # race discount (juzi: defender halves)
                            break
                    if P["raid_seize_on"] and 50 <= step <= 150:
                        # raid offence window: prime trees (age <= 25,
                        # most of the income integral still ahead) get
                        # the cheap travel gate; tail trees must clear
                        # a real value floor -- no more tail-grabbing.
                        if view.trees[c][1] <= 25:
                            gate = 20.0 + P["raid_seize_travel_cost"] * d
                        else:
                            gate = max(P["raid_seize_tail_min"],
                                       20.0 + P["raid_seize_travel_cost"] * d)
                        if val <= gate:
                            continue
                    elif val <= 20.0 + P["seize_travel_cost"] * d:
                        continue   # must clear seize cost + travel
                    score = val * 100.0 / (4.0 + d)
                    if P["raid_seize_on"] and 50 <= step <= 150 \
                            and view.trees[c][1] <= 25:
                        score *= 1.5   # prime-tree priority in the window
                    if best is None or score > best[0]:
                        best = (score, wid, c)
            if best is None:
                break
            _, wid, c = best
            p_assigned[wid] = c
            p_rem.remove(wid)

        # ================= Phase B: movers ===============================
        movers = [(wid, True) for wid, _ in collectors if wid not in stayers]
        movers += [(wid, False) for wid, _ in planters if wid not in stayers]
        unprocessed = {view.my_workers[wid][0] for wid, _ in movers}
        unknown_count = int((belief < 0).sum())
        for wid, is_collector in movers:
            pos, cargo, _ = view.my_workers[wid]
            unprocessed.discard(pos)
            forbidden = reserved | unprocessed
            can_stay = pos not in reserved
            dest = None
            if is_collector:
                hard, soft = danger_for(cargo) if cargo >= 60 else (set(), set())
                forbidden = forbidden | hard
                if wid in yielders or (cargo <= 0 and pos in view.my_base_cells):
                    can_stay = False   # yielded corridor / never park on a base
                if returning[wid]:
                    goal = view.nearest_base[pos]
                    # v0.2 freeze fix: BFS descent is priced at 1000/step
                    # and staying is dropped whenever a legal descent
                    # exists, so siphon/soft costs can bend the path but
                    # never stall it (v0.1's plateau freeze).
                    if free_base:
                        # base mouth exempt: consolidation collisions allowed
                        dest = self._choose_dest(view, pos, goal, cargo,
                                                 forbidden - view.my_base_cells,
                                                 can_stay, soft, siphon_scale=0.5,
                                                 dist_weight=1000.0,
                                                 descent_first=True)
                    else:
                        # own base cells are never forbidden to a
                        # returner: entering an occupied own base
                        # consolidates by collision and the winner
                        # deposits the combined cargo (Phase 2 -> 3)
                        dest = self._choose_dest(view, pos, goal, cargo,
                                                 forbidden - view.my_base_cells,
                                                 can_stay, soft,
                                                 dist_weight=1000.0,
                                                 descent_first=True)
                    # en-route collect pause handled in phase A (rich cell
                    # stay) -- returning collectors keep moving.
                elif wid in hunt_target:
                    dest = self._choose_dest(view, pos, hunt_target[wid], cargo,
                                             forbidden, can_stay, soft)
                elif wid in assigned:
                    dest = self._choose_dest(view, pos, assigned[wid], cargo,
                                             forbidden, can_stay, soft)
                else:
                    fb = attr - 3.0 * DIST[pos]
                    fb[pos] = -1e9
                    for b in BASE_CELLS:
                        fb[b] = -1e9   # never idle onto a base cell
                    goal = int(np.argmax(fb))
                    dest = self._choose_dest(view, pos, goal, cargo,
                                             forbidden, can_stay, soft)
            else:
                hard = {v[0] for v in view.opp_collectors.values()}
                forbidden = forbidden | hard
                if step < 22 and unknown_count > 40:
                    # exploration: maximise newly revealed cells (we see our
                    # 4-neighbours; each reveal mirrors symmetrically)
                    best, goal = -1, pos
                    for cell in [pos] + [int(c) for c in NBR4[pos]]:
                        if cell in forbidden and cell != pos:
                            continue
                        reveal = 0
                        for nb in NBR4[cell]:
                            if belief[nb] < 0:
                                reveal += 1
                            for m in MIRROR[nb]:
                                if belief[m] < 0:
                                    reveal += 1
                        if reveal > best:
                            best, goal = reveal, cell
                    dest = goal
                    if dest == pos and view.tree_at[pos] < 0 \
                            and pos not in view.my_base_cells \
                            and pos not in view.opp_base_cells \
                            and view.cash >= plant_cost:
                        # idling here would plant unintentionally (see below)
                        nbrs = [int(c) for c in NBR4[pos] if c not in forbidden]
                        dest = nbrs[0] if nbrs else pos
                else:
                    goal = p_assigned.get(wid)
                    if goal is None:
                        goal = int(np.argmax(plant_value / (1.0 + DIST[pos])))
                    dest = self._choose_dest(view, pos, goal, 0.0,
                                             forbidden, can_stay, set(),
                                             siphon_scale=0.0)
                    # Planting is implicit: a planter that ENDS the step on a
                    # plantable cell with cash >= plant_cost plants whether
                    # we want it or not. Phase A already committed intended
                    # plants, so a Phase B planter must never idle on a
                    # plantable cell -- force a move to the best neighbour.
                    if dest == pos and view.tree_at[pos] < 0 \
                            and pos not in view.my_base_cells \
                            and pos not in view.opp_base_cells \
                            and view.cash >= plant_cost:
                        dest = self._choose_dest(view, pos, goal, 0.0,
                                                 forbidden, False, set(),
                                                 siphon_scale=0.0)
            if dest is None:
                dest = pos
            act = self._action_for(pos, dest)
            dest_of[wid] = dest
            reserved.add(dest)
            if act:
                actions[wid] = act

        # ================= recruitment =================================
        arrivals = {}
        for wid, d in dest_of.items():
            if d in view.my_base_cells:
                arrivals[d] = arrivals.get(d, 0) + 1
        proj_cash = view.cash
        n_coll = len(view.my_collectors)
        n_plan = len(view.my_planters)
        n_workers = n_coll + n_plan
        stock = float(np.where(belief < 0, 11.0, belief).sum()) / 2500.0
        stock_ok = stock > P["expand_stock_ratio"] or n_workers < 6
        target_coll = P["target_collectors_late"] if step >= 150 else P["target_collectors_mid"]
        if self.switched:
            # pure harvest: fill the roster to 10 with collectors,
            # no stock gate, no planting reserve.
            target_coll = max(target_coll, 10 - n_plan)
            stock_ok = True
        top_cell = int(np.argmax(plant_value))
        top_plant_roi = plant_roi(plant_value, top_cell, plant_cost, steps_left)
        for bid, bpos in sorted(view.my_bases.items()):
            if step >= P["recruit_stop_step"] or n_workers >= 10:
                break
            if bpos in view.worker_at:
                continue  # base cell occupied: a spawn would collide
            if arrivals.get(bpos, 0) > 0 and not free_base:
                continue  # a friendly unit lands here this step
            if proj_cash < 30:
                break
            want = None
            if step == 0:
                want = "RECCOLLECTOR"
            else:
                reserve = plant_cost if (n_plan > 0 and view.my_tree_count < 6
                                         and top_plant_roi > 10) else 0.0
                if P["tree_seize_on"] and view.opp_planters \
                        and view.my_tree_count >= 1:
                    reserve = max(reserve, 20.0)   # v0.2 re-seize budget
                if self.switched:
                    reserve = 0.0
                want_planter = (not self.switched
                                and n_plan < P["target_planters"] and step >= 6
                                and n_coll >= 2 and step < 185
                                and (top_plant_roi > 12 or n_plan == 0))
                if want_planter and proj_cash >= 30 + (20 if view.my_tree_count == 0 else 0):
                    want = "RECPLANTER"
                elif n_coll < target_coll and stock_ok and proj_cash - 30 >= reserve:
                    want = "RECCOLLECTOR"
                elif n_coll < target_coll and n_workers < 6 and proj_cash - 30 >= reserve:
                    want = "RECCOLLECTOR"
            if want:
                actions[bid] = want
                proj_cash -= 30
                n_workers += 1
                if want == "RECCOLLECTOR":
                    n_coll += 1
                else:
                    n_plan += 1
        return actions


_policy = Policy()
LAST_POLICY = _policy


def make_agent():
    global LAST_POLICY
    pol = Policy()
    LAST_POLICY = pol
    return lambda observation, configuration: pol.act(observation, configuration)


def agent(observation, configuration):
    return _policy.act(observation, configuration)
