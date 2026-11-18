"""Run with python reports/master_family/tactical_rules/test_defense.py."""

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from fastsim import FastCarbon

from defense import ACTIONS, DefenseBoard


def scenario(own, enemy, *, homes=(0, 24), trees=({}, {}), cash=(80, 80), **config):
    sim = FastCarbon({"size": 5, **config})
    players = []
    for side, workers in enumerate((own, enemy)):
        workers_by_id = {f"player-{side}-worker-{i}": w for i, w in enumerate(workers)}
        players.append([cash[side], {f"base-{side}": homes[side]}, workers_by_id, trees[side]])
    sim.load_observation({"players": players, "carbon": [0.] * 25, "step": 14})
    return sim


A, B, T = "player-0-worker-0", "player-1-worker-0", "player-0-tree-0"
C, P = "COLLECTOR", "PLANTER"


class DefenseTest(unittest.TestCase):
    def test_reachable_kill_matches_all_enemy_actions(self):
        # Includes ties, strictly heavier opponents, a moving collector's
        # cargo cost, toroidal reach, and both possible observation seats.
        for role, other, move_cost, seat in itertools.product((C, P), (C, P), (0, 0.25), (0, 1)):
            for their_cargo in ((12, 16, 20) if other == C else (0,)):
                workers = [((5, 16 if role == C else 0, role),), ((9, their_cargo, other),)]
                if seat:
                    workers.reverse()
                sim = scenario(*workers, homes=(17, 22), moveCost=move_cost)
                state = sim.get_state()
                own_id, enemy_id = (B, A) if seat else (A, B)
                board = DefenseBoard(sim.obs(seat), sim.config)
                for move in ACTIONS:
                    prediction = board.assess(own_id, move)
                    deaths = []
                    for reply in ACTIONS:
                        sim.set_state(state)
                        joint = [{}, {}]
                        joint[seat][own_id], joint[1-seat][enemy_id] = move, reply
                        sim.step(*joint)
                        deaths.append(own_id not in sim.workers)
                    self.assertEqual(any(deaths), prediction.collision_loss_possible,
                                     (role, other, move_cost, seat, their_cargo, move))

    def test_collector_intercepts_only_if_planter_stays(self):
        sim = scenario([(11, 15, C)], [(12, 0, P)], trees=({T: [12, 6]}, {}))
        first = sim.get_state()
        for action, seized in (("NONE", True), ("RIGHT", False)):
            sim.set_state(first)
            board = DefenseBoard(sim.obs(0), sim.config)
            self.assertTrue(board.assess(A, "RIGHT").can_screen_tree)
            sim.step({A: action}, {})
            self.assertEqual(sim.trees[T].player == 1, seized)
        sim = scenario([(11, 15, C)], [(12, 0, P), (13, 0, C)], trees=({T: [12, 6]}, {}))
        self.assertFalse(DefenseBoard(sim.obs(0), sim.config).assess(A, "RIGHT").can_screen_tree)
        sim.step({A: "RIGHT"}, {"player-1-worker-1": "LEFT"})
        self.assertNotIn(A, sim.workers)

    def test_tree_approach_and_unfunded_stay(self):
        sim = scenario([(4, 0, C)], [(13, 0, P)], trees=({T: [12, 6]}, {}), cash=(0, 1))
        self.assertFalse(DefenseBoard(sim.obs(0), sim.config).threatened_trees)
        sim.step({}, {B: "LEFT"})
        self.assertEqual(sim.trees[T].player, 0)
        self.assertIn(12, DefenseBoard(sim.obs(0), sim.config).threatened_trees)
        sim.step({}, {})
        self.assertEqual(sim.trees[T].player, 0)

    def test_expiring_tree_is_removed_before_collision(self):
        sim = scenario([(11, 15, C)], [(12, 0, P)], trees=({T: [12, 50]}, {}))
        self.assertFalse(DefenseBoard(sim.obs(0), sim.config).assess(A, "RIGHT").can_screen_tree)
        sim.step({A: "RIGHT"}, {})
        self.assertNotIn(T, sim.trees)

    def test_contested_base_still_belongs_to_its_owner(self):
        sim = scenario([(11, 25, C)], [(13, 5, C)], homes=(12, 24))
        risk = DefenseBoard(sim.obs(0), sim.config).assess(A, "RIGHT")
        self.assertTrue(risk.home_contested and risk.collision_loss_possible)
        sim.step({A: "RIGHT"}, {B: "LEFT"})
        self.assertNotIn(A, sim.workers)
        self.assertEqual(sim.cash, [110, 80])  # winner brings both cargoes to our base

    def test_opposing_base_spawn_before_deposit(self):
        for balance, can_spawn in ((36, False), (37, True)):
            sim = scenario([(11, 20, C)], [], homes=(0, 12), cash=(0, balance),
                           recCollectorIncreaseCost=7, recPlanterCost=80)
            risk = DefenseBoard(sim.obs(0), sim.config).assess(A, "RIGHT")
            self.assertTrue(risk.opponent_base)
            self.assertEqual(risk.collision_loss_possible, can_spawn)
            sim.step({A: "RIGHT"}, {"base-1": "RECCOLLECTOR"})
            self.assertEqual(A not in sim.workers, can_spawn)
            self.assertEqual(sim.cash[1], balance + 20 - (37 if can_spawn else 0))

    def test_tree_drain_counts_diagonals_but_not_expired_trees(self):
        sim = scenario([(11, 100, C)], [], trees=({}, {"e-0": [8, 3], "e-1": [17, 50]}))
        obs = sim.obs(0)
        obs.pop("carbon")  # layer does not require even the masked carbon plane
        self.assertEqual(DefenseBoard(obs, sim.config).assess(A, "RIGHT").enemy_tree_count, 1)
        sim.step({A: "RIGHT"}, {})
        self.assertAlmostEqual(sim.workers[A].carbon, 90)


if __name__ == "__main__":
    unittest.main()
