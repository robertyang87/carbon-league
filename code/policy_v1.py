"""policy_v1 -- Carbon League strategy v1, "shoulders skeleton" rule hybrid.

Design sources (ideas re-implemented, no code copied):
- juzi reference strategy (reference_juzi.py, human finalist, site-published):
  * four-fold-symmetric belief carbon map: every observed cell is mirrored
    into its 3 symmetric counterparts; unobserved cells regrow x1.03/step;
    an enemy collector that stays reveals its cell's carbon through its
    cargo increment (inc / 0.3 * 0.7 after a x0.7 decay)   [juzi: ObservationParser]
  * plant value map = sum over 4-neighbours of carbon / (1 + trees around
    that neighbour), boosted x1.5 where enemy trees dominate, /1.5 where own
    trees dominate                                          [juzi: plant_value_map]
  * rob/seize valuation = neighbour carbon integrated over the tree's
    remaining life at the 3.75% absorption rate             [juzi: rob_feature]
  * plant decision: value * (1 - (1-0.0375)^min(16, steps left)) > price
  * return-home cargo thresholds scaled by cash / endgame  [juzi: collector_feature]
  * global no-self-collision via priority order + reserved destinations,
    released near the end so the base mouth can consolidate [juzi: obs_transform]
- reCurse (Halite III #3 post-mortem; Halite shares this game's collision
  rule verbatim) -- see shoulders.md section 4.1:
  * attraction field over the resource map for global harvesting
  * candidate-pool global assignment with destination reservation
    (Hungarian-style target layer + path-layer reservation). v1 collision
    discipline is two-phase: stationary units (deposit / stay-collect /
    plant / seize / defend) commit and reserve their cells FIRST, then
    movers pick destinations that are neither reserved nor still occupied
    by a not-yet-committed friendly unit.
  * forbidden moves from local dominance / enemy reachability
  * endgame: Dijkstra (torus shortest-path) return scheduling, and in the
    last ~20 steps the base cell is EXEMPT from anti-collision so cargo
    consolidation by collision can run
  * minimal recruiting: cash check + steps-left check + global resource
    stock > 33% before expanding
- alpha_analysis.md section 8 (AlphaCarbon-2 dissection): opening template
  (step-1 double collector, first planter ~step 7, first tree steps 12-28,
  ~7 collectors / 2 planters mid-game composition), cargo-protection
  discipline (its death-carbon median is 57 vs opponents' 210), endgame
  base-mouth collision consolidation worth ~+220, and its proven loss
  pattern (post-step-150 pure-harvest deposit throughput), hence our late
  game tilts to 9 collectors and stops planting on a payback horizon.
- User-directed combat module: hunt loaded enemy collectors with empty /
  low collectors (collision rule: least cargo wins and captures), seize
  high-value enemy trees, protect own cargo from enemy trees' 10% siphon
  and from enemy low collectors, race-awareness on tree seizes.
- opponent_model: reconstructs the enemy's approximate belief state (their
  visible cells are deterministic given their unit positions) and scores
  intent hypotheses per enemy unit into a <=5-cell next-position
  distribution; used by hunting (1-3 step interception), cargo protection
  (enemy low collectors' predicted cells are costly), coordination
  (predicted enemy destinations penalised) and seize racing (arrival-step
  comparison).

The agent is a per-instance closure from make_agent(); the module-level
agent() uses a singleton, as the league runner expects. State resets when
observation step goes back to 0.
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
    return_cargo=200.0,         # normal return-home cargo threshold
    return_cargo_lowcash=110.0, # when cash < 150, cycle back earlier (juzi)
    return_cargo_late=160.0,    # step > 250
    early_return_cargo=100.0,   # step < 50: fast cycling funds expansion (juzi-like)
    quick_deposit_cargo=120.0,  # return anyway if this loaded and base <= 3 away
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
                elif cargo >= 380 or (cargo >= 180 and view.step >= 250):
                    # returning: shortest-path first steps toward nearest base.
                    # Threshold calibrated on replay validation (see report):
                    # top players keep collecting until ~400 cargo, so a
                    # 150-380 cargo collector is usually still harvesting.
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

    @staticmethod
    def _action_for(pos, dest):
        if dest is None or dest == pos:
            return None
        for k in range(4):
            if int(NBR4[pos, k]) == dest:
                return ACTIONS[k]
        return None

    def _choose_dest(self, view, pos, goal, cargo, forbidden, can_stay,
                     soft, siphon_scale=1.0):
        """Pick a destination cell minimising 10*dist-to-goal + siphon
        exposure + soft danger, among cells not in `forbidden` (stay only
        if can_stay). Falls back to the cheapest forbidden cell only when
        no legal option exists."""
        options = []
        cells = [int(c) for c in NBR4[pos]]
        if can_stay:
            cells.append(pos)
        legal, illegal = [], []
        for cell in cells:
            if cell in view.opp_base_cells and cell != pos:
                continue  # never step on an enemy base (cargo donates / death)
            cost = 10.0 * DIST[cell, goal]
            cost += siphon_scale * float(view.opp_tree_near[cell]) * cargo * 0.08
            if cell in soft:
                cost += 30.0
            (illegal if cell in forbidden else legal).append((cost, cell))
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
        self.last_step = step
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
            if cargo >= P["quick_deposit_cargo"] and view.base_dist[pos] <= 3:
                ret = min(ret, P["quick_deposit_cargo"])
            r = cargo >= ret
            if endgame and cargo > 0:
                r = True
            if step >= 292 and cargo > 0:
                r = True
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
        planter_stay = {}
        for wid, (pos, cargo, _) in planters:
            tree_here = view.trees.get(pos)
            stay = False
            if tree_here is not None and tree_here[0] != me and view.cash >= 20:
                remain = 50 - tree_here[1]
                if plant_roi(plant_value, pos, 20.0, remain) > 10.0:
                    stay = True                        # seize
            elif tree_here is None and pos not in view.my_base_cells \
                    and pos not in view.opp_base_cells and step < P["plant_stop_step"] \
                    and view.my_tree_count < P["max_my_trees"]:
                roi = plant_roi(plant_value, pos, plant_cost, steps_left)
                if plant_value[pos] >= 70 and roi > max(12.0, 0.5 * plant_cost) \
                        and view.cash >= plant_cost:
                    stay = True                        # plant
            elif tree_here is not None and tree_here[0] == me:
                val = tree_income_estimate(belief, pos, 50 - tree_here[1])
                threat = any(DIST[view.opp_planters[w][0], pos] <= 1
                             for w in view.opp_planters)
                if val > 120 and threat:
                    stay = True                        # defend
            planter_stay[wid] = stay
            if stay:
                stayers.add(wid)
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
        # per-target load discount (reCurse / juzi share-discounting).
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

        # Planter targets: plant cells / rob targets, globally assigned.
        plant_cands = [int(c) for c in np.argsort(-plant_value, kind="stable")[:24]]
        rob_cands = sorted(tpos for tpos, (owner, age, tid) in view.trees.items()
                           if owner != me and 50 - age >= 13)
        p_assigned = {}
        p_rem = [w for w, _ in planters if w not in stayers]
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
                        score = roi * 100.0 / (4.0 + DIST[pos, c])
                        if best is None or score > best[0]:
                            best = (score, wid, c)
                for c in rob_cands:
                    if c in p_assigned.values():
                        continue
                    d = DIST[pos, c]
                    remain = 50 - view.trees[c][1]
                    if remain - d < 8:
                        continue
                    roi = plant_roi(plant_value, c, 20.0, remain - d)
                    for ewid, (epos, _, _) in view.opp_planters.items():
                        if DIST[epos, c] <= d:
                            roi *= 0.5     # race discount (juzi: defender halves)
                            break
                    if roi <= 10.0:
                        continue
                    score = roi * 100.0 / (4.0 + d)
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
                if returning[wid]:
                    goal = view.nearest_base[pos]
                    if free_base:
                        # base mouth exempt: consolidation collisions allowed
                        dest = self._choose_dest(view, pos, goal, cargo,
                                                 forbidden - view.my_base_cells,
                                                 can_stay, soft, siphon_scale=0.5)
                    else:
                        dest = self._choose_dest(view, pos, goal, cargo,
                                                 forbidden, can_stay, soft)
                    # en-route collect pause handled in phase A (rich cell
                    # stay) -- returning collectors keep moving.
                elif wid in hunt_target:
                    dest = self._choose_dest(view, pos, hunt_target[wid], cargo,
                                             forbidden, can_stay, soft)
                elif wid in assigned:
                    dest = self._choose_dest(view, pos, assigned[wid], cargo,
                                             forbidden, can_stay, soft)
                else:
                    goal = int(np.argmax(attr - 3.0 * DIST[pos]))
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
                want_planter = (n_plan < P["target_planters"] and step >= 6
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


def make_agent():
    pol = Policy()
    return lambda observation, configuration: pol.act(observation, configuration)


def agent(observation, configuration):
    return _policy.act(observation, configuration)
