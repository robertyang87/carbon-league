"""One-step defensive features for a worker considering a world-frame action.

Only the public player lists and the game configuration are read. Reach means an
opponent *can* arrive, without predicting its action. Friendly collisions and
reservations belong to the joint decoder. No probability or utility is inferred.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import inf


ACTIONS = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")
_INDEX = {action: index for index, action in enumerate(ACTIONS)}


@lru_cache(maxsize=8)
def _geometry(size):
    moves, neighborhoods = [], []
    for cell in range(size * size):
        row, col = divmod(cell, size)
        def shifted(dr, dc):
            return ((row + dr) % size) * size + (col + dc) % size

        moves.append((cell, shifted(-1, 0), shifted(0, 1), shifted(1, 0), shifted(0, -1)))
        neighborhoods.append({shifted(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)})
    return moves, neighborhoods


@dataclass(frozen=True)
class ActionDefense:
    destination: int
    collision_loss_possible: bool
    home_contested: bool
    opponent_base: bool
    enemy_tree_count: int
    can_screen_tree: bool


class DefenseBoard:
    """Build once for the current player, then query any of its workers.

    A contested home is reachable by an existing enemy worker. A threatened
    tree has a live enemy planter already standing on it: moving onto a tree
    does not seize it until a later stay. Affordability is resolved later in
    the step (after deposits), so this flag reports an opportunity to seize.
    """

    def __init__(self, observation, configuration=None):
        cfg = configuration or {}
        self.size = int(cfg.get("size", 15))
        self._moves, neighborhoods = _geometry(self.size)
        self._move_factor = 1 - cfg.get("moveCost", 0)
        lifespan = cfg.get("treeLifespan", 50)
        players = observation["players"]
        me = observation["player"]
        _, my_bases, self._workers, my_trees = players[me]
        cash, enemy_bases, enemies, enemy_trees = players[1 - me]
        self._enemy_bases = set(enemy_bases.values())
        live_tree_cells = {cell for cell, age in my_trees.values() if age < lifespan}
        self.threatened_trees = {
            pos for pos, _, role in enemies.values() if role == "PLANTER" and pos in live_tree_cells
        }

        cells = self.size * self.size
        self._minimum_collector = [inf] * cells
        self._enemy_reachable = [False] * cells
        for pos, cargo, role in enemies.values():
            for move, destination in enumerate(self._moves[pos]):
                self._enemy_reachable[destination] = True
                if role == "COLLECTOR":
                    arrival_cargo = cargo * self._move_factor if move else cargo
                    self._minimum_collector[destination] = min(self._minimum_collector[destination], arrival_cargo)
        self.contested_bases = {pos for pos in my_bases.values() if self._enemy_reachable[pos]}

        # Recruitment happens before collision. Any single affordable spawn
        # can be chosen instead of the other bases' actions.
        if len(enemies) < cfg.get("workerLimit", 10):
            collectors = sum(w[2] == "COLLECTOR" for p in players for w in p[2].values())
            planters = len(self._workers) + len(enemies) - collectors
            collector_cost = cfg.get("recCollectorCost", 30) + collectors * cfg.get("recCollectorIncreaseCost", 0)
            planter_cost = cfg.get("recPlanterCost", 30) + planters * cfg.get("recPlanterIncreaseCost", 0)
            for pos in self._enemy_bases:
                if cash >= min(collector_cost, planter_cost):
                    self._enemy_reachable[pos] = True
                if cash >= collector_cost:
                    self._minimum_collector[pos] = min(0.0, self._minimum_collector[pos])

        self._tree_count = [0] * cells
        for cell, age in enemy_trees.values():
            if age < lifespan:
                for neighbor in neighborhoods[cell]:
                    self._tree_count[neighbor] += 1

    def assess(self, worker_id, action):
        """Describe a single action; NONE or None means stay.

        Collision loss is possible against an enemy existing worker or spawn
        acting alone. A collector loses against a lighter collector or a tie;
        any enemy meeting a planter kills it. A collector can remove a tree
        intruder by reaching its cell if it stays, provided the collector is
        safe from other enemy collisions. Tree count measures pre-existing
        enemy-tree exposure around this cell; actual ownership can change.
        """
        move = _INDEX["NONE" if action is None else action]
        pos, cargo, role = self._workers[worker_id]
        dst = self._moves[pos][move]
        collector = role == "COLLECTOR"
        arrival_cargo = cargo * self._move_factor if move else cargo
        loses = self._minimum_collector[dst] <= arrival_cargo if collector else self._enemy_reachable[dst]
        return ActionDefense(
            destination=dst,
            collision_loss_possible=loses,
            home_contested=dst in self.contested_bases,
            opponent_base=dst in self._enemy_bases,
            enemy_tree_count=self._tree_count[dst],
            can_screen_tree=collector and dst in self.threatened_trees and not loses,
        )
