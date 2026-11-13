"""FastCarbon: a fast, numpy/stdlib re-implementation of the Carbon League
(zerosum_env "carbon") board transition, semantically aligned step-by-step
with `Board.next()` in zerosum_env/envs/carbon/helpers.py and with
`populate_board()` in zerosum_env/envs/carbon/carbon.py.

Design notes (all verified against the original source, not from memory):

* Positions are stored in observation index space: index = row * size + col,
  matching the flat `carbon` list.  Point(x, y) in the original maps to
  index (size - y - 1) * size + x, so the worker action offsets are
  UP=(0,+1) -> index-size, RIGHT=(+1,0), DOWN=(0,-1) -> index+size,
  LEFT=(-1,0), with per-coordinate wraparound (torus).
* The step pipeline mirrors Board.next() phase by phase, in the same order,
  with the same iteration orders (players 0 then 1; per-player entity order;
  brand-new recruits appended to the global worker order as they are created;
  collision groups in board-worker order; trees in board-tree order;
  absorption neighbours in ConnectedField8 order) and with plain Python
  float arithmetic in the same accumulation order, so results are bit-equal
  to the original in the parity harness (see test_fastsim_parity.py).
* Quirks reproduced deliberately:
  - Recruitment costs are computed from the *global* collector/planter
    counts (both players), exactly as in Board.next().
  - Plant price: a player with <= playerTreeProtectiveNumber trees pays the
    flat plantCost (20); everyone else pays
    plantCost + plantCostInflationRatio * plantCostInflationBase ** A,
    where A = number of *alive* (age < treeLifespan) trees on the whole
    board, evaluated once before any withering this step.
  - Collision: the unique collector carrying the least carbon wins and
    captures every deleted worker's carbon; a tie for least (or no
    collector at all in the group) deletes everyone; a deleted worker's
    carbon goes to the tree owner if a tree stands on the cell, else to
    the base owner if a base stands there, else onto the cell (capped).
  - Base deposit also fires on *enemy* bases: an enemy collector stepping
    on your base with carbon > smeltCost hands you its carbon; with
    carbon <= smeltCost it is deleted.
  - Seizing a tree calls Board._delete_tree(), whose side effect clears
    the seizing planter's cell worker marker (cell._worker_id = None).
    That marker is used by the absorption/regen phases, so we reproduce it.
  - _delete_worker() nulls the creator field of every trees_dict entry
    that names the deleted worker; obs["trees"] therefore shows None for
    trees whose planting worker has died.
  - Newborn trees do not absorb on the step they are planted, but they do
    take part in the quad-surround flag construction that suppresses
    carbon regeneration on neighbouring cells.
  - Cells in the quad-surround flag dict (4-neighbours of any tree) never
    regenerate this step, even when their flag set is empty.
  - Every 4-neighbour of every tree is a flag-dict key, so tree-adjacent
    cells never regenerate.
  - Cash is rounded with Python round(x, 2) and cell carbon with
    round(x, 3) at the end of the step; worker cargo is not rounded.
* Out of scope (env-core concerns, not Board.next()): agent timeouts,
  remainingOverageTime bookkeeping (passed through unchanged), and the
  elimination/DONE logic in carbon.py compute_next_board().
"""

import math

import numpy as np

# Default configuration: the defaults from zerosum_env/envs/carbon/carbon.json
DEFAULT_CONFIG = {
    "episodeSteps": 300,
    "startingCarbon": 2500,
    "size": 15,
    "carbonCoverage": 45,
    "numberOfBases": 2,
    "startPosOffset": 1,
    "startingCellCarbon": 50,
    "maxCellCarbon": 100,
    "recCollectorCost": 30,
    "recCollectorIncreaseCost": 0,
    "recPlanterCost": 30,
    "recPlanterIncreaseCost": 0,
    "moveCost": 0,
    "smeltCost": 0,
    "workerLimit": 10,
    "planterLimit": 1,
    "collectorLimit": 2,
    "plantCost": 20,
    "plantCostInflationRatio": 5,
    "plantCostInflationBase": 1.235,
    "playerTreeProtectiveNumber": 0,
    "co2FrmWithered": 30,
    "seizeCost": 20,
    "initialCollectRate": 0.3,
    "collectDecreaseRate": 0.0,
    "regenRate": 0.03,
    "treeLifespan": 50,
    "cellAbsorptionRate": 0.0375,
    "collectorAbsorptionRate": 0.1,
    "randomSeed": None,
    "agentTimeout": 60,
    "actTimeout": 1,
    "runTimeout": 9600,
    "remainingOverageTime": 60,
}

_MOVE_OFFSETS = {  # action -> Point offset (dx, dy), as in WorkerAction.to_point()
    "UP": (0, 1),
    "RIGHT": (1, 0),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
}
# ConnectedField4 order in helpers.py: (0,1), (1,0), (0,-1), (-1,0)
_FIELD4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# ConnectedField8 order in helpers.py
_FIELD8 = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]


class _Worker:
    __slots__ = ("id", "player", "pos", "carbon", "is_collector", "action")

    def __init__(self, wid, player, pos, carbon, is_collector):
        self.id = wid
        self.player = player
        self.pos = pos
        self.carbon = carbon
        self.is_collector = is_collector
        self.action = None


class _Tree:
    __slots__ = ("id", "player", "pos", "age")

    def __init__(self, tid, player, pos, age):
        self.id = tid
        self.player = player
        self.pos = pos
        self.age = age


class FastCarbon:
    """Fast drop-in replacement for the original Board transition chain."""

    def __init__(self, config=None):
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg
        self.size = int(cfg["size"])
        n = self.size
        self.n_cells = n * n
        # Neighbour tables in index space.  For index i: x = i % n,
        # y = n - 1 - i // n (Point coords).  Offset (dx, dy) wraps per axis.
        self._nbr4 = []
        self._nbr8 = []
        self._move_nbr = {}
        tables = {}
        for name, offs in (("f4", _FIELD4), ("f8", _FIELD8),
                           ("UP", [(0, 1)]), ("RIGHT", [(1, 0)]),
                           ("DOWN", [(0, -1)]), ("LEFT", [(-1, 0)])):
            tab = []
            for i in range(self.n_cells):
                x, y = i % n, n - 1 - i // n
                row = []
                for dx, dy in offs:
                    nx, ny = (x + dx) % n, (y + dy) % n
                    row.append((n - 1 - ny) * n + nx)
                tab.append(row[0] if len(row) == 1 else row)
            tables[name] = tab
        self._nbr4 = tables["f4"]
        self._nbr8 = tables["f8"]
        self._move_nbr = {k: tables[k] for k in ("UP", "RIGHT", "DOWN", "LEFT")}

        self.step_count = 0
        self.carbon = [0.0] * self.n_cells
        self.cash = [80, 80]
        self.bases = {0: [], 1: []}          # player -> [(base_id, pos), ...]
        self.base_at = {}                    # pos -> (player, base_id)
        self.workers = {}                    # wid -> _Worker (insertion ordered)
        self.player_workers = {0: [], 1: []}
        self.trees = {}                      # tid -> _Tree (insertion ordered)
        self.player_trees = {0: [], 1: []}
        self.trees_dict = {}                 # tid -> [creator_wid|None, last_absorption]
        self._wid_counter = [0, 0]
        self._tid_counter = [0, 0]
        self.remaining_overage = cfg.get("remainingOverageTime", 60)

    # ------------------------------------------------------------------ ids
    def _new_worker_id(self, player):
        wid = f"player-{player}-worker-{self._wid_counter[player]}"
        self._wid_counter[player] += 1
        return wid

    def _new_tree_id(self, player):
        tid = f"player-{player}-tree-{self._tid_counter[player]}"
        self._tid_counter[player] += 1
        return tid

    @property
    def worker_counters(self):
        return list(self._wid_counter)

    @property
    def tree_counters(self):
        return list(self._tid_counter)

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, full_carbon=None, cash=80):
        """Reset to the populate_board() initial state.

        seed: if given, the map is generated exactly like populate_board()
        after np.random.seed(seed) -- i.e. one generation from that seed.
        (The original never seeds the global RNG itself; populate_board
        consumes the ambient np.random stream: one uniform() call then one
        choice() call.  If seed is None and full_carbon is None, the
        ambient np.random stream is used as-is, again like the original.)
        full_carbon: explicit flat map (list/array of size*size), overrides
        generation.
        """
        cfg = self.config
        size = self.size
        self.step_count = 0
        self.workers = {}
        self.player_workers = {0: [], 1: []}
        self.trees = {}
        self.player_trees = {0: [], 1: []}
        self.trees_dict = {}
        self._wid_counter = [0, 0]
        self._tid_counter = [0, 0]
        self.cash = [cash, cash]

        if full_carbon is not None:
            assert len(full_carbon) == self.n_cells
            self.carbon = [float(v) for v in full_carbon]
        else:
            if seed is not None:
                state = np.random.get_state()
                np.random.seed(seed)
                try:
                    self.carbon = self._generate_map()
                finally:
                    np.random.set_state(state)
            else:
                self.carbon = self._generate_map()

        # Starting base positions (index space), exactly as populate_board.
        off = cfg["startPosOffset"]
        num_bases = cfg["numberOfBases"]
        starting_positions = [0] * 2 * num_bases
        starting_positions[0] = size * (size // 4 + off) + size // 4 + off
        starting_positions[1] = size * (3 * size // 4 - off) + 3 * size // 4 - off
        if num_bases * 2 > 2:
            starting_positions[2] = size * (size // 4 + off) + 3 * size // 4 - off
            starting_positions[3] = size * (3 * size // 4 - off) + size // 4 + off
        self.bases = {0: [], 1: []}
        self.base_at = {}
        for p in range(2):
            for j in range(num_bases):
                pos = starting_positions[p * num_bases + j]
                bid = f"player-{p}-recrtCenter-{j}"
                self.bases[p].append((bid, pos))
                self.base_at[pos] = (p, bid)
                self.carbon[pos] = 0
        return self

    def _generate_map(self):
        """Byte-for-byte replica of populate_board()'s carbon distribution,
        consuming the ambient np.random stream in the same order."""
        cfg = self.config
        size = self.size
        half = math.ceil(size / 2)
        min_carbon_cells = int(math.ceil(
            cfg["startingCarbon"] / cfg["startingCellCarbon"] / 4))
        n_carbon_cells = int(math.ceil(half * half * cfg["carbonCoverage"] / 100))
        n_carbon_cells = max(n_carbon_cells, min_carbon_cells)
        carbon_mean = min(cfg["startingCarbon"] / 4 / n_carbon_cells,
                          cfg["startingCellCarbon"])
        half_grid_carbon = np.random.uniform(2 * carbon_mean, size=n_carbon_cells)
        half_grid_indices = np.random.choice((half * half), size=n_carbon_cells,
                                             replace=False)
        half_grid = np.zeros(half * half, np.float16)
        half_grid[half_grid_indices] = half_grid_carbon
        half_grid = half_grid.reshape((half, half))
        full = [0] * (size * size)
        for r, row in enumerate(half_grid):
            for c, val in enumerate(row):
                val = min(int(val), cfg["startingCellCarbon"])
                full[size * r + c] = val
                full[size * r + (size - c - 1)] = val
                full[size * (size - 1) - (size * r) + c] = val
                full[size * (size - 1) - (size * r) + (size - c - 1)] = val
        return [float(v) for v in full]

    # ------------------------------------------------------- state loading
    def load_observation(self, full_obs):
        """Load state from a *full* (unfiltered) observation dict, i.e. the
        format Board.observation produces / populate_board leaves in
        state[0].observation (with 'full_carbon' or full 'carbon')."""
        size = self.size
        carbon = full_obs.get("full_carbon", full_obs["carbon"])
        self.reset(full_carbon=list(carbon))
        self.step_count = int(full_obs.get("step", 0))
        self.cash = [full_obs["players"][0][0], full_obs["players"][1][0]]
        # Bases: reset() already created the standard base layout; rebuild
        # from the observation to honour crafted layouts.
        self.bases = {0: [], 1: []}
        self.base_at = {}
        max_rc = [0, 0]
        for p in range(2):
            for bid, pos in full_obs["players"][p][1].items():
                self.bases[p].append((bid, pos))
                self.base_at[pos] = (p, bid)
                self.carbon[pos] = 0
                max_rc[p] = max(max_rc[p], _id_suffix(bid) + 1)
        self.workers = {}
        self.player_workers = {0: [], 1: []}
        self.trees = {}
        self.player_trees = {0: [], 1: []}
        wc = [0, 0]
        tc = [0, 0]
        for p in range(2):
            for tid, (pos, age) in full_obs["players"][p][3].items():
                self.trees[tid] = _Tree(tid, p, pos, age)
                self.player_trees[p].append(tid)
                # Tree ids embed the *creator* ("player-<c>-tree-<n>"), and a
                # seized tree keeps its creator id while changing owner. The
                # per-player id counter is the creator's high-water mark, so
                # restore it under the creator parsed from the id -- not under
                # the current owner. Crediting the owner left the creator's
                # counter at 0 after a seizure, and the creator's next planted
                # tree reused the seized tree's live id and overwrote it
                # (REVIEW_FULL_20261113 P3 snapshot defect).
                tc[_id_creator(tid, p)] = max(tc[_id_creator(tid, p)],
                                              _id_suffix(tid) + 1)
            for wid, (pos, cargo, occ) in full_obs["players"][p][2].items():
                self.workers[wid] = _Worker(wid, p, pos, cargo, occ == "COLLECTOR")
                self.player_workers[p].append(wid)
                wc[p] = max(wc[p], _id_suffix(wid) + 1)
        self._wid_counter = wc
        self._tid_counter = tc
        src_trees_dict = full_obs.get("trees", {})
        self.trees_dict = {}
        for p in range(2):
            for tid in self.player_trees[p]:
                entry = src_trees_dict.get(tid, [None, 0])
                self.trees_dict[tid] = [entry[0], entry[1]]
        self.remaining_overage = full_obs.get(
            "remainingOverageTime", self.config.get("remainingOverageTime", 60))
        return self

    # ------------------------------------------------------------ obs/state
    def _players_obs(self):
        players = []
        for p in range(2):
            bases = {bid: pos for bid, pos in self.bases[p]}
            workers = {}
            for wid in self.player_workers[p]:
                w = self.workers[wid]
                workers[wid] = [w.pos, w.carbon,
                                "COLLECTOR" if w.is_collector else "PLANTER"]
            trees = {}
            for tid in self.player_trees[p]:
                t = self.trees[tid]
                trees[tid] = [t.pos, t.age]
            players.append([self.cash[p], bases, workers, trees])
        return players

    def full_observation(self, player=0):
        """Unfiltered observation (full carbon map), Board.observation format."""
        return {
            "carbon": list(self.carbon),
            "players": self._players_obs(),
            "player": player,
            "step": self.step_count,
            "trees": {tid: [v[0], v[1]] for tid, v in self.trees_dict.items()},
            "remainingOverageTime": self.remaining_overage,
        }

    def obs(self, player_index):
        """Per-player observation identical in format to the original
        interpreter's per-agent observation: carbon is zeroed everywhere
        except cells visible to that player's workers (a collector sees its
        own cell; a planter sees its 4-neighbour cells -- not its own)."""
        visible = [0] * self.n_cells
        for wid in self.player_workers[player_index]:
            w = self.workers[wid]
            if w.is_collector:
                visible[w.pos] = self.carbon[w.pos]
            else:
                for nb in self._nbr4[w.pos]:
                    visible[nb] = self.carbon[nb]
        return {
            "carbon": visible,
            "players": self._players_obs(),
            "player": player_index,
            "step": self.step_count,
            "trees": {tid: [v[0], v[1]] for tid, v in self.trees_dict.items()},
            "remainingOverageTime": self.remaining_overage,
        }

    def get_state(self):
        """Compact state export for comparisons / training pipelines."""
        return {
            "step": self.step_count,
            "cash": list(self.cash),
            "carbon": list(self.carbon),
            "workers": {
                wid: {"player": w.player, "pos": w.pos, "carbon": w.carbon,
                      "type": "COLLECTOR" if w.is_collector else "PLANTER"}
                for wid, w in self.workers.items()
            },
            "trees": {
                tid: {"player": t.player, "pos": t.pos, "age": t.age}
                for tid, t in self.trees.items()
            },
            "bases": {p: list(self.bases[p]) for p in range(2)},
            "trees_dict": {tid: [v[0], v[1]] for tid, v in self.trees_dict.items()},
        }

    # ----------------------------------------------------------------- step
    def _kill_worker(self, w, cell_worker):
        """Mirror of Board._delete_worker incl. trees_dict creator nulling."""
        if cell_worker.get(w.pos) == w.id:
            del cell_worker[w.pos]
        self.player_workers[w.player].remove(w.id)
        del self.workers[w.id]
        for entry in self.trees_dict.values():
            if entry[0] == w.id:
                entry[0] = None

    def step(self, actions_a, actions_b):
        cfg = self.config
        actions = [actions_a or {}, actions_b or {}]
        size = self.size
        carbon = self.carbon

        # ---- Board construction order: players 0,1; their workers/trees.
        # (self.workers / self.trees are re-ordered to construction order.)
        ordered_workers = {}
        for p in range(2):
            for wid in self.player_workers[p]:
                ordered_workers[wid] = self.workers[wid]
        self.workers = ordered_workers
        ordered_trees = {}
        for p in range(2):
            for tid in self.player_trees[p]:
                ordered_trees[tid] = self.trees[tid]
        self.trees = ordered_trees
        ordered_td = {}
        for p in range(2):
            for tid in self.player_trees[p]:
                ordered_td[tid] = self.trees_dict[tid]
        self.trees_dict = ordered_td
        # Board construction zeroes the carbon of every base cell.
        for pos in self.base_at:
            carbon[pos] = 0
        # Attach actions (unknown ids / unknown strings are ignored, as in
        # Board.__init__ where a raw action not in the Enum members -> None).
        for w in self.workers.values():
            a = actions[w.player].get(w.id)
            w.action = a if a in _MOVE_OFFSETS else None

        # ---- Phase 0: prices (Board.next() preamble)
        total_collectors = sum(1 for w in self.workers.values() if w.is_collector)
        total_planters = len(self.workers) - total_collectors
        rec_collector_cost = (cfg["recCollectorCost"]
                              + total_collectors * cfg["recCollectorIncreaseCost"])
        rec_planter_cost = (cfg["recPlanterCost"]
                            + total_planters * cfg["recPlanterIncreaseCost"])
        lifespan = cfg["treeLifespan"]
        alive_tree_count = sum(1 for t in self.trees.values() if t.age < lifespan)
        plant_market_price = (cfg["plantCost"] + cfg["plantCostInflationRatio"]
                              * (cfg["plantCostInflationBase"] ** alive_tree_count))
        threshold = cfg["playerTreeProtectiveNumber"]
        player_plant_cost = {}
        for p in range(2):
            player_plant_cost[p] = (cfg["plantCost"]
                                    if len(self.player_trees[p]) <= threshold
                                    else plant_market_price)

        # ---- Phase 1: per player -- recruit, move, age trees
        disappeared = set()
        tree_at = {t.pos: t for t in self.trees.values()}
        for p in range(2):
            act_p = actions[p]
            for bid, bpos in self.bases[p]:
                a = act_p.get(bid)
                if a == "RECCOLLECTOR" and self.cash[p] >= rec_collector_cost \
                        and len(self.player_workers[p]) < cfg["workerLimit"]:
                    self.cash[p] = max(self.cash[p] - rec_collector_cost, 0)
                    wid = self._new_worker_id(p)
                    w = _Worker(wid, p, bpos, 0, True)
                    self.workers[wid] = w
                    self.player_workers[p].append(wid)
                elif a == "RECPLANTER" and self.cash[p] >= rec_planter_cost \
                        and len(self.player_workers[p]) < cfg["workerLimit"]:
                    self.cash[p] = max(self.cash[p] - rec_planter_cost, 0)
                    wid = self._new_worker_id(p)
                    w = _Worker(wid, p, bpos, 0, False)
                    self.workers[wid] = w
                    self.player_workers[p].append(wid)
            for wid in list(self.player_workers[p]):
                w = self.workers[wid]
                if w.action is not None:
                    w.pos = self._move_nbr[w.action][w.pos]
                    w.carbon = w.carbon * (1 - cfg["moveCost"])
            for tid in list(self.player_trees[p]):
                t = self.trees[tid]
                if t.age >= lifespan:
                    disappeared.add(t.pos)
                    if tree_at.get(t.pos) is t:
                        del tree_at[t.pos]
                    carbon[t.pos] = cfg["co2FrmWithered"]
                    self.player_trees[p].remove(tid)
                    del self.trees[tid]
                    del self.trees_dict[tid]
                else:
                    t.age += 1

        # ---- Phase 2: collisions
        self.last_collisions = []           # read-only log: (worker id, player, carbon, who got the carbon: player or None)
        cell_worker = {}
        collisions_flag = set()
        groups = {}
        for w in self.workers.values():
            groups.setdefault(w.pos, []).append(w)
        for pos, group in groups.items():
            if len(group) == 1:
                cell_worker[pos] = group[0].id
                continue
            collectors_by_carbon = {}
            for w in group:
                if w.is_collector:
                    collectors_by_carbon.setdefault(w.carbon, []).append(w)
            winner = None
            if collectors_by_carbon:
                smallest = collectors_by_carbon[min(collectors_by_carbon)]
                if len(smallest) == 1:
                    winner = smallest[0]
            deleted = [w for w in group if w is not winner]
            if deleted:
                collisions_flag.add(pos)
            if winner is not None:
                cell_worker[pos] = winner.id
            winner_is_collector = winner is not None and winner.is_collector
            for w in deleted:
                if winner_is_collector:
                    winner.carbon += w.carbon; gets = winner.player
                elif pos in tree_at:
                    self.cash[tree_at[pos].player] += w.carbon; gets = tree_at[pos].player
                elif pos in self.base_at:
                    self.cash[self.base_at[pos][0]] += w.carbon; gets = self.base_at[pos][0]
                else:
                    carbon[pos] = min(carbon[pos] + w.carbon,
                                      cfg["maxCellCarbon"]); gets = None
                self.last_collisions.append((w.id, w.player, float(w.carbon), gets,
                                             winner.player if winner_is_collector else None))   # 5th: winning collector's player
                self._kill_worker(w, cell_worker)

        # ---- Phase 3: base deposit (incl. enemy-base rules)
        smelt = cfg["smeltCost"]
        for p in range(2):
            for bid, bpos in self.bases[p]:
                wid = cell_worker.get(bpos)
                if wid is None:
                    continue
                w = self.workers[wid]
                if not w.is_collector:
                    continue
                if w.player != p and w.carbon <= smelt:
                    self._kill_worker(w, cell_worker)
                elif w.carbon >= smelt:
                    self.cash[p] += w.carbon - smelt
                    w.carbon = 0

        # ---- Phase 4: stay actions -- collect / plant / seize
        collect_rate = {}
        for p in range(2):
            n_coll = sum(1 for wid in self.player_workers[p]
                         if self.workers[wid].is_collector)
            collect_rate[p] = max(
                cfg["initialCollectRate"] - n_coll * cfg["collectDecreaseRate"], 0)
        new_born = set()
        for w in list(self.workers.values()):
            if w.action is not None:
                continue
            pos = w.pos
            has_tree = pos in tree_at
            has_base = pos in self.base_at
            if not has_tree and not has_base:
                if not w.is_collector and self.cash[w.player] >= player_plant_cost[w.player]:
                    self.cash[w.player] -= player_plant_cost[w.player]
                    tid = self._new_tree_id(w.player)
                    t = _Tree(tid, w.player, pos, 1)
                    self.trees[tid] = t
                    self.player_trees[w.player].append(tid)
                    self.trees_dict[tid] = [w.id, 0]
                    tree_at[pos] = t
                    carbon[pos] = 0
                    new_born.add(tid)
                elif w.is_collector:
                    delta = carbon[pos] * collect_rate[w.player]
                    if delta > 0:
                        carbon[pos] = max(carbon[pos] - delta, 0)
                        w.carbon += delta
            elif has_tree and not has_base:
                t = tree_at[pos]
                if t.player != w.player and not w.is_collector \
                        and self.cash[w.player] >= cfg["seizeCost"]:
                    self.cash[w.player] -= cfg["seizeCost"]
                    tid, age = t.id, t.age
                    # Board._delete_tree(): removes tree; also clears the
                    # cell worker marker (the seizing planter's) -- replicated
                    # via cell_worker below.
                    self.player_trees[t.player].remove(tid)
                    del self.trees[tid]
                    del self.trees_dict[tid]
                    if cell_worker.get(pos) == w.id:
                        del cell_worker[pos]
                    nt = _Tree(tid, w.player, pos, age)
                    self.trees[tid] = nt
                    self.player_trees[w.player].append(tid)
                    self.trees_dict[tid] = [w.id, 0]
                    tree_at[pos] = nt

        # ---- Phase 5: quad-surround flags for tree absorption / regen
        quad_flag = {}
        for t in self.trees.values():
            for nb in self._nbr4[t.pos]:
                if nb not in quad_flag:
                    quad_flag[nb] = set()
                if nb in tree_at:
                    continue
                wid = cell_worker.get(nb)
                if wid is None:
                    quad_flag[nb].add(t.id)
                else:
                    w = self.workers[wid]
                    if not w.is_collector or w.action is not None:
                        quad_flag[nb].add(t.id)

        # ---- Phase 6: tree absorption
        cell_rate = cfg["cellAbsorptionRate"]
        coll_rate = cfg["collectorAbsorptionRate"]
        board_reduction = {}
        worker_reduction = {}
        for t in list(self.trees.values()):
            if t.id in new_born:
                continue
            tree_carbon = 0
            wid = cell_worker.get(t.pos)
            if wid is not None:
                w = self.workers[wid]
                if w.is_collector and w.player != t.player:
                    absorbed = w.carbon * coll_rate
                    tree_carbon += absorbed
                    worker_reduction[wid] = worker_reduction.get(wid, 0) + absorbed
            for nb in self._nbr8[t.pos]:
                wid = cell_worker.get(nb)
                if wid is not None:
                    w = self.workers[wid]
                    if w.is_collector and w.player != t.player:
                        absorbed = w.carbon * coll_rate
                        tree_carbon += absorbed
                        worker_reduction[wid] = worker_reduction.get(wid, 0) + absorbed
                if carbon[nb] > 0 and nb in quad_flag and t.id in quad_flag[nb]:
                    absorbed = carbon[nb] * cell_rate
                    tree_carbon += absorbed
                    board_reduction[nb] = board_reduction.get(nb, 0) + absorbed
            self.cash[t.player] += tree_carbon
            creator = self.trees_dict[t.id][0]
            del self.trees_dict[t.id]
            self.trees_dict[t.id] = [creator, tree_carbon]

        for pos, flag in quad_flag.items():
            if carbon[pos] > 0 and flag:
                carbon[pos] = max(carbon[pos] - board_reduction.get(pos, 0), 0)
        for wid, red in worker_reduction.items():
            if red > 0:
                w = self.workers[wid]
                w.carbon = max(w.carbon - red, 0)

        # ---- Phase 7: regeneration of unaffected cells
        regen_factor = 1 + cfg["regenRate"]
        max_c = cfg["maxCellCarbon"]
        for pos in range(self.n_cells):
            if pos in collisions_flag or pos in tree_at or pos in disappeared \
                    or pos in quad_flag:
                continue
            wid = cell_worker.get(pos)
            if wid is not None and self.workers[wid].action is None:
                continue
            v = carbon[pos] * regen_factor
            carbon[pos] = max_c if v > max_c else v

        # ---- Phase 8: clear actions, rounding, step counter
        for w in self.workers.values():
            w.action = None
        self.cash[0] = round(self.cash[0], 2)
        self.cash[1] = round(self.cash[1], 2)
        self.carbon = [round(v, 3) for v in carbon]
        self.step_count += 1
        return self


def _id_suffix(entity_id):
    try:
        return int(str(entity_id).rsplit("-", 1)[1])
    except (ValueError, IndexError):
        return -1


def _id_creator(entity_id, fallback):
    """Creator player index embedded in an entity id ("player-<c>-...").

    Falls back to `fallback` for ids that do not carry a valid creator."""
    parts = str(entity_id).split("-")
    if len(parts) >= 3 and parts[0] == "player":
        try:
            c = int(parts[1])
        except ValueError:
            return fallback
        if c in (0, 1):
            return c
    return fallback
