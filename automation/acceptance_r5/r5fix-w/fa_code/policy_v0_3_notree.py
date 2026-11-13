"""policy_v0.3-notree (ablation: tree warfare OFF) -- Carbon League strategy v0.3, "shoulders skeleton" rule hybrid.

v0.3 changes over v0.2 (see policy_v0_3_report.md). v0.2's paired test
came out at -5008 (statistically v1's -4857), and its ablation showed
the tree-warfare module eating its own gains: 2 planters cannot staff
planting + robbing + guarding at once, the guard's real value was
"planters staying home" rather than standing on trees, and the rob
module mostly captured tail-value trees (seizes 0.5 -> 4.1 per game,
tree income only 683 -> 720). v0.3 reworks the tree resource model
and adds a death-cargo programme driven by trace forensics on the
v0.2 games (diag_deaths_v03.py):

  a. Tree warfare resource model, redone:
     * Planter corps 2 -> 3, staged: the 3rd planter is recruited only
       once we own >= 4 trees (planter3_min_trees), so the early
       collector economy is not starved.
     * GUARDING IS RETIRED (tree_def_on defaults to 0; the v0.2 guard
       dispatch code is kept dormant for ablation continuity).
       Defence is now the RE-SEIZE RACE (tree_reseize_on): the policy
       remembers its trees step to step; when juzi seizes one (same
       tree id, owner flips), the tree becomes a priority target if
       its remaining income integral (tree_income_estimate over its
       remaining life) is >= reseize_value_min (150). The nearest
       free planter races there, hovers on a neighbouring cell while
       the enemy planter still occupies the tree (stepping in would
       be a Phase-2 collision suicide), and seizes it back the step
       the cell frees up.
     * OFFENCE QUALITY GATE: rob targets must now clear rob_value
       >= seize_value_min (150) after the race discount, on top of the
       old value > 20 + 3/travel-step gate; the Phase-A stay-seize
       uses the same 150 gate (re-seize targets exempt -- they are
       pre-gated at detection). Tail-value trees are no longer chased.
     * PLANT-SITE HOT-ZONE AVOIDANCE (plant_hot_avoid_on): a decaying
       heat map of enemy worker positions (x0.97/step, +1 on the
       cell, +0.4 on 4-neighbours) discounts plant-candidate scores,
       and candidates within 2 steps of an enemy planter are cut to
       25%; a planter already standing on a cell does not plant when
       an enemy planter is adjacent (near-certain instant re-rob).
  b. Death-cargo programme (trace forensics, diag_deaths_v03.py):
     the ledger's 727/game death cargo overstates the true loss --
     264/game is base-mouth consolidation, recovered the same step by
     the collision winner's deposit (frame-verified). The TRUE loss
     is ~463/game, 96% hunt_kill: mass duel deaths of LOW-cargo
     collectors (102 kills, mean cargo ~35; only two kills above 150)
     by emptier enemy collectors at point-blank range, steps 51-250,
     on open ground -- NOT fat collectors bleeding inside siphon
     zones (8% of true loss). Plus 112 planter deaths (14/game),
     76% of them 5+ cells from home: expedition churn.
     Fixes (cargo_guard_on / planter_guard_on):
     * the danger map (hard/soft sets) is now consulted from cargo
       >= 15 instead of >= 60, in both the Phase-A stay check and
       Phase-B movement; enemy collectors at EQUAL cargo add their
       reach to the soft set (a tie kills both).
     * _choose_dest prices hard-danger cells at +500 in BOTH pools,
       so when every option is forbidden it still prefers a
       friendly-reserved cell over an enemy-reachable one (v0.2's
       fallback treated them as equals -- traced converging kill,
       seed 1002 step 99, 213 cargo).
     * fat-cargo bubble: cargo >= 180 with a lighter enemy collector
       within 2 steps -> return home NOW, do not keep harvesting.
     * planter self-preservation: planter movement treats every
       enemy collector's reach (cell + 4-neighbours) as hard danger
       (any collector contact kills a planter), and a plant-stay is
       cancelled when an enemy collector is adjacent (it can step in
       and kill the planter in Phase 2 before the plant resolves).

Kept unchanged from v0.2: the R-configuration return-threshold group
(450/300/380/200/180, -15 distance discount, steps-left hard rule),
the opponent-model 400 calibration, and ALL freeze fixes (BFS
distance field, base-mouth open-to-returners, empty-never-parks,
yield-the-corridor rule). The v0.1 beam-search planner stays removed.
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
    # --- v0.2 tree warfare module (ablation switches) ---
    tree_def_on=0,              # v0.3: guard dispatch RETIRED (kept dormant;
                                # v0.2 ablation: its value was planters
                                # staying home, not standing on trees)
    tree_seize_on=0,            # juzi-style rob valuation + targeting
    def_value_min=80.0,         # own tree worth guarding above this value
    def_radius=2,               # enemy planter this close = threat
    def_goal_radius=5,          # ... or this close when it is his rob goal
    seize_travel_cost=3.0,      # rob gate: value > 20 + this * travel steps
    seize_stay_value=25.0,      # legacy v0.2 stay-seize gate (superseded)
    # --- v0.3 tree warfare rework ---
    target_planters_max=2,      # planter corps ceiling (was fixed 2)
    planter3_min_trees=4,       # 3rd planter only once we own >= this many
    tree_reseize_on=0,          # re-seize race replaces standing guard
    reseize_value_min=150.0,    # register a lost tree as a race target
                                # only if its remaining income >= this
    reseize_ttl_steps=40,       # race target expires after this many steps
    seize_value_min=150.0,      # offence quality gate: rob_value must
                                # clear this after the race discount
    plant_hot_avoid_on=0,       # enemy-activity heat discounts planting
    # --- v0.3 death-cargo programme (diag_deaths_v03.py) ---
    cargo_guard_on=1,           # danger map + fat-cargo bubble master switch
    guard_danger_min_cargo=15.0,  # consult danger map from this cargo
                                # (v0.2 used 60: mass low-cargo duel deaths)
    bubble_cargo=180.0,         # fat-cargo bubble: at/above this load,
    bubble_dist=2,              # a lighter enemy collector this close
                                # forces an immediate return home
    planter_guard_on=1,         # planters avoid enemy-collector reach;
                                # plant-stay cancelled if one is adjacent
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
        # v0.3 state: our trees last step (pos -> tid) for seize-flip
        # detection, live re-seize race targets (pos -> step detected),
        # and the enemy-activity heat map for plant-site avoidance.
        self.prev_my_trees = {}
        self.reseize_targets = {}
        self.opp_heat = np.zeros(N)

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
                     descent_first=False, hard=frozenset()):
        """Pick a destination cell minimising dist_weight*BFS-dist-to-goal
        + siphon exposure + soft danger, among cells not in `forbidden`
        (stay only if can_stay). Falls back to the cheapest forbidden
        cell only when no legal option exists. BFS distances (enemy
        bases blocked) replace Manhattan distance, whose plateaus next
        to enemy bases froze loaded collectors in v0.1.
        v0.3: cells in `hard` (enemy collectors' reach) cost +500 in
        BOTH pools -- in v0.2 they were merely "forbidden", so when
        every option was forbidden the fallback priced a lethal cell
        the same as a friendly-reserved one (traced converging kill:
        seed 1002 step 99, our 213-cargo collector stepped onto the
        one cell a lighter enemy collector could reach)."""
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
            if cell in hard:
                cost += 500.0
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

    # -- main ---------------------------------------------------------------
    def act(self, obs, configuration):
        step = obs["step"]
        if step == 0 or step < self.last_step:
            self.belief_state = BeliefState()
            self.opponent = OpponentModel(self.belief_state)
            self.prev_my_trees = {}
            self.reseize_targets = {}
            self.opp_heat = np.zeros(N)
        self.last_step = step
        self._bfs_cache = {}
        self.belief_state.update(obs)
        view = BoardView(obs)
        belief = self.belief_state.belief
        bv = np.where(belief < 0, P["unknown_default"], belief)
        plant_value = compute_plant_value(belief, view)
        preds = self.opponent.predict(view, plant_value)

        # ---- v0.3: enemy-activity heat map (plant-site avoidance) ----
        if P["plant_hot_avoid_on"]:
            self.opp_heat *= 0.97
            for _p, _c, _t in view.opp_workers.values():
                self.opp_heat[_p] += 1.0
                for _nb in NBR4[_p]:
                    self.opp_heat[int(_nb)] += 0.4

        # ---- v0.3: re-seize race target bookkeeping ------------------
        # A tree we owned last step (same tid) that is now enemy-owned
        # was seized this step: register it as a race target if its
        # remaining income clears the value gate. Targets die with the
        # tree, are dropped when we own it again / it is too old / its
        # value decays below 2/3 of the gate / the TTL expires.
        if P["tree_reseize_on"]:
            for tpos, tid in self.prev_my_trees.items():
                cur = view.trees.get(tpos)
                if cur is not None and cur[2] == tid and cur[0] != view.me \
                        and tpos not in self.reseize_targets:
                    val = tree_income_estimate(belief, tpos, 50 - cur[1])
                    if val >= P["reseize_value_min"]:
                        self.reseize_targets[tpos] = step
            for tpos in list(self.reseize_targets):
                cur = view.trees.get(tpos)
                if cur is None or cur[0] == view.me \
                        or 50 - cur[1] < 8 \
                        or step - self.reseize_targets[tpos] \
                        > P["reseize_ttl_steps"] \
                        or tree_income_estimate(belief, tpos, 50 - cur[1]) \
                        < (2.0 / 3.0) * P["reseize_value_min"]:
                    del self.reseize_targets[tpos]
        actions = {}
        me = view.me
        steps_left = 300 - step
        plant_cost = _plant_cost(view.my_tree_count, view.total_trees)
        endgame = step >= P["endgame_start"]
        free_base = step >= P["endgame_free_base"]

        collectors = sorted(view.my_collectors.items(), key=lambda kv: (-kv[1][1], kv[0]))
        planters = sorted(view.my_planters.items(), key=lambda kv: (-plant_value[kv[1][0]], kv[0]))

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
                elif P["cargo_guard_on"] and ecargo == cargo and cargo > 0:
                    # v0.3: an equal-cargo enemy collector kills BOTH
                    # of us on contact -- its reach is soft danger.
                    soft.add(epos)
                    for nb in NBR4[epos]:
                        soft.add(int(nb))
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
            if cargo >= P["quick_deposit_cargo"] and view.base_dist[pos] <= 3:
                ret = min(ret, P["quick_deposit_cargo"])
            r = cargo >= ret
            # v0.3 fat-cargo bubble: a fat load with a LIGHTER enemy
            # collector within bubble_dist steps heads home now -- the
            # two >200-cargo true losses in the v0.2 traces were both
            # point-blank duel kills while still harvesting.
            if P["cargo_guard_on"] and cargo >= P["bubble_cargo"]:
                for _ep, _ec, _et in view.opp_collectors.values():
                    if _ec < cargo and int(DIST[pos, _ep]) <= P["bubble_dist"]:
                        r = True
                        break
            # v0.1 steps-left hard rule: if the trip home must start now
            # for the cargo to land at all, start it (any step, any load).
            if cargo > 0 and steps_left <= int(view.base_dist[pos]) + 1:
                r = True
            if endgame and cargo > 0:
                r = True
            if step >= 292 and cargo > 0:
                r = True
            returning[wid] = r

        # v0.3: the danger map is consulted from 15 cargo, not 60 --
        # the v0.2 traces show mass duel deaths of collectors carrying
        # 15-60 (they were invisible to the old >= 60 gate).
        danger_gate = P["guard_danger_min_cargo"] if P["cargo_guard_on"] \
            else 60.0
        stayers = set()
        for wid, (pos, cargo, _) in collectors:
            if pos in view.my_base_cells and cargo > 0:
                stayers.add(wid)                       # deposit
            elif not returning[wid] and bv[pos] >= P["stay_collect_belief"] \
                    and view.tree_at[pos] < 0:
                hard, _ = danger_for(cargo) if cargo >= danger_gate \
                    else (set(), set())
                if pos not in hard:
                    stayers.add(wid)                   # stay & collect
        # ================= v0.2 tree warfare: threat map ==============
        # defend_assign: my planter wid -> own tree cell it must guard.
        # A tree is threatened when an enemy planter is within
        # def_radius steps, or within def_goal_radius when the opponent
        # model's rob goal for that planter is exactly this tree.
        defend_assign = {}
        if P["tree_def_on"] and planters and view.opp_planters:
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
                # valuation clears the v0.3 quality gate (150, was 25
                # in v0.2 -- tail-value seizures wasted planter steps);
                # a live re-seize target is pre-gated at detection and
                # is always retaken when we stand on it.
                if view.cash >= 20:
                    if P["tree_seize_on"] and rob_value(
                            belief, view, pos, 0, steps_left) \
                            >= P["seize_value_min"]:
                        stay = True                        # seize
                    elif P["tree_reseize_on"] and pos in self.reseize_targets:
                        stay = True                        # re-seize
            elif wid in defend_assign and defend_assign[wid] == pos:
                stay = True                                # guard post
            elif wid in defend_assign:
                stay = False   # must move to its guard post this step
            elif tree_here is None and pos not in view.my_base_cells \
                    and pos not in view.opp_base_cells and step < P["plant_stop_step"] \
                    and view.my_tree_count < P["max_my_trees"]:
                roi = plant_roi(plant_value, pos, plant_cost, steps_left)
                if plant_value[pos] >= 70 and roi > max(12.0, 0.5 * plant_cost) \
                        and view.cash >= plant_cost:
                    stay = True                        # plant
                    # v0.3: do not plant into immediate danger -- an
                    # adjacent enemy collector can step in and kill
                    # the planter in Phase 2 before the plant resolves
                    # (planter_guard_on), and an adjacent enemy planter
                    # means the new tree is re-robbed at once
                    # (plant_hot_avoid_on).
                    if P["planter_guard_on"] and any(
                            int(DIST[pos, _ep]) <= 1
                            for _ep, _ec, _et in view.opp_collectors.values()):
                        stay = False
                    if P["plant_hot_avoid_on"] and any(
                            int(DIST[pos, _ep]) <= 1
                            for _ep, _ec, _et in view.opp_planters.values()):
                        stay = False
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
        # Hunting: intercept the predicted path of a loaded enemy collector
        # with an empty/low own collector (least cargo wins the collision).
        hunt_target = {}
        if step < P["endgame_start"] - 10:
            foes = sorted(view.opp_collectors.items(), key=lambda kv: (-kv[1][1], kv[0]))
            for ewid, (epos, ecargo, _) in foes:
                if ecargo < P["hunt_min_enemy_cargo"]:
                    continue
                path = self.opponent.mode_path(view, plant_value, ewid, steps=3)
                best = None
                for wid, (pos, cargo, _) in collectors:
                    if wid in stayers or wid in hunt_target:
                        continue
                    if cargo > P["hunt_max_own_cargo"] or cargo > ecargo - 50:
                        continue
                    for t, cell in enumerate(path, start=1):
                        if cell in view.opp_base_cells:
                            continue
                        if DIST[pos, cell] <= t:
                            score = ecargo * (0.92 ** t) - 45 * t - 10 * DIST[pos, cell]
                            if best is None or score > best[0]:
                                best = (score, wid, cell)
                if best is not None:
                    hunt_target[best[1]] = best[2]

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
        # v0.3 re-seize race: live race targets take the NEAREST free
        # planter before any plant/rob assignment -- speed is the
        # whole point (the tree earns for juzi every step we are late,
        # and juzi's planter may still be standing on it).
        if P["tree_reseize_on"]:
            for tpos in sorted(self.reseize_targets):
                if not p_rem:
                    break
                best_w = min(p_rem, key=lambda w: (
                    int(DIST[view.my_planters[w][0], tpos]), w))
                p_assigned[best_w] = tpos
                p_rem.remove(best_w)
        # v0.3 plant-site hot-zone factors (enemy planter proximity is
        # the re-rob risk; the decaying heat map prices general enemy
        # activity around the candidate cell).
        opp_planter_cells = [v[0] for v in view.opp_planters.values()]

        def plant_factor(c):
            if not P["plant_hot_avoid_on"]:
                return 1.0
            if opp_planter_cells and min(int(DIST[c, ep])
                                         for ep in opp_planter_cells) <= 2:
                return 0.25
            return 1.0 / (1.0 + 0.5 * float(self.opp_heat[c]))

        while p_rem:
            best = None
            for wid in p_rem:
                pos = view.my_planters[wid][0]
                if step < P["plant_stop_step"] and view.my_tree_count < P["max_my_trees"]:
                    for c in plant_cands:
                        if view.tree_at[c] >= 0 or c in view.my_base_cells \
                                or c in view.opp_base_cells or c in p_assigned.values():
                            continue
                        roi = plant_roi(plant_value, c, plant_cost, steps_left)
                        if roi <= max(12.0, 0.5 * plant_cost):
                            continue
                        score = roi * 100.0 / (4.0 + DIST[pos, c]) \
                            * plant_factor(c)
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
                    if val <= 20.0 + P["seize_travel_cost"] * d:
                        continue   # must clear seize cost + travel
                    if val < P["seize_value_min"]:
                        continue   # v0.3 quality gate: no tail-value trees
                    score = val * 100.0 / (4.0 + d)
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
                # v0.3: danger map consulted from 15 cargo (was 60),
                # and the hard set is also priced into _choose_dest so
                # the all-forbidden fallback still avoids lethal cells.
                hard, soft = danger_for(cargo) if cargo >= danger_gate \
                    else (set(), set())
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
                                                 descent_first=True, hard=hard)
                    else:
                        # own base cells are never forbidden to a
                        # returner: entering an occupied own base
                        # consolidates by collision and the winner
                        # deposits the combined cargo (Phase 2 -> 3)
                        dest = self._choose_dest(view, pos, goal, cargo,
                                                 forbidden - view.my_base_cells,
                                                 can_stay, soft,
                                                 dist_weight=1000.0,
                                                 descent_first=True, hard=hard)
                    # en-route collect pause handled in phase A (rich cell
                    # stay) -- returning collectors keep moving.
                elif wid in hunt_target:
                    dest = self._choose_dest(view, pos, hunt_target[wid], cargo,
                                             forbidden, can_stay, soft,
                                             hard=hard)
                elif wid in assigned:
                    dest = self._choose_dest(view, pos, assigned[wid], cargo,
                                             forbidden, can_stay, soft,
                                             hard=hard)
                else:
                    fb = attr - 3.0 * DIST[pos]
                    fb[pos] = -1e9
                    for b in BASE_CELLS:
                        fb[b] = -1e9   # never idle onto a base cell
                    goal = int(np.argmax(fb))
                    dest = self._choose_dest(view, pos, goal, cargo,
                                             forbidden, can_stay, soft,
                                             hard=hard)
            else:
                if P["planter_guard_on"]:
                    # v0.3: ANY enemy-collector contact kills a
                    # planter, so its whole reach (cell + 4-neighbours)
                    # is hard danger (and priced +500 in _choose_dest).
                    hard = set()
                    for _ep, _ec, _et in view.opp_collectors.values():
                        hard.add(_ep)
                        for _nb in NBR4[_ep]:
                            hard.add(int(_nb))
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
                    if wid in p_assigned:
                        # v0.3: a rob / re-seize target still OCCUPIED
                        # by an enemy worker must not be entered
                        # (Phase-2 collision, planter dies, no seize):
                        # forbid the cell itself so _choose_dest hovers
                        # on a neighbouring descent cell until it frees.
                        _w = view.worker_at.get(goal)
                        _t = view.trees.get(goal)
                        if _w is not None and not _w[1] \
                                and _t is not None and _t[0] != me:
                            forbidden = forbidden | {goal}
                    dest = self._choose_dest(view, pos, goal, 0.0,
                                             forbidden, can_stay, set(),
                                             siphon_scale=0.0, hard=hard)
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
                                                 siphon_scale=0.0, hard=hard)
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
                if P["tree_reseize_on"] and self.reseize_targets:
                    reserve = max(reserve, 20.0)   # v0.3 race budget
                # v0.3 staged corps: the 3rd planter unlocks once the
                # tree economy is real (>= planter3_min_trees trees),
                # so early collector growth is never starved for it.
                eff_target_planters = P["target_planters_max"] \
                    if view.my_tree_count >= P["planter3_min_trees"] \
                    else P["target_planters"]
                want_planter = (n_plan < eff_target_planters and step >= 6
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
        # v0.3: remember our trees for next step's seize-flip detection
        self.prev_my_trees = {pos: tid for pos, (owner, age, tid)
                              in view.trees.items() if owner == me}
        return actions


_policy = Policy()


def make_agent():
    pol = Policy()
    return lambda observation, configuration: pol.act(observation, configuration)


def agent(observation, configuration):
    return _policy.act(observation, configuration)
