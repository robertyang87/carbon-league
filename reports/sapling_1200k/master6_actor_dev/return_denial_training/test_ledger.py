"""One public transition per fixture. Both engines must agree before constructing each label."""

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tree_response_training/planter_survival"))
from extract import FastCarbon, step
from ledger import one


def uid(seat, i):
    return f"player-{seat}-worker-{i}"


class LedgerTest(unittest.TestCase):
    def transition(self, workers, orders=(), trees=(), tiles=(), config=None, focal=(0, 0), perspective=0):
        sim = FastCarbon(config)
        sim.reset(full_carbon=[0.0] * sim.n_cells)
        state = sim.full_observation()
        for pos, value in tiles:
            state["carbon"][pos] = value
        for p, i, pos, load, *role in workers:
            state["players"][p][2][uid(p, i)] = [pos, load, role[0] if role else "COLLECTOR"]
        for p, i, pos, age in trees:
            tid = f"player-{p}-tree-{i}"
            state["players"][p][3][tid] = [pos, age]
            state["trees"][tid] = [uid(p, 0), 0]
        sim.load_observation(state)
        before = sim.full_observation()
        joint = [{}, {}]
        for p, i, act in orders:
            joint[p][uid(p, i)] = act
        after, cash = step(sim, before, sim.config, joint)
        return (
            one(
                before,
                after,
                sim,
                cash,
                perspective,
                uid(perspective, focal[0]),
                uid(1 - perspective, focal[1]),
                joint[perspective],
                joint[1 - perspective],
                sim.config,
            ),
            after,
        )

    def test_no_collision_stationary_collectors(self):
        r, after = self.transition(
            [(0, 0, 50, 3), (1, 0, 100, 2)],
            tiles=[(50, 20), (100, 30)],
            config={"initialCollectRate": 0.25, "collectDecreaseRate": 0.0},
        )
        self.assertEqual(r["transfer_margin"], 0.0)
        self.assertEqual(r["real_new_harvest"], [5.0, 7.5])
        self.assertEqual(r["cash_change"], [0.0, 0.0])
        self.assertEqual(r["collision_cross_received"], [0.0, 0.0])
        self.assertFalse(r["own_died"] or r["enemy_died"])
        self.assertEqual(after["players"][0][2][uid(0, 0)][1], 8.0)

    def test_simultaneous_counter_transfer_and_friendly_cargo(self):
        r, after = self.transition(
            [(0, 0, 50, 0), (0, 1, 49, 20), (0, 2, 99, 10), (1, 0, 51, 40), (1, 1, 100, 1)],
            [(0, 1, "RIGHT"), (0, 2, "RIGHT"), (1, 0, "LEFT")],
            config={"moveCost": 0.1},
        )
        self.assertEqual(r["cross_to_worker"], [36.0, 9.0])
        self.assertEqual(r["transfer_margin"], 27.0)
        self.assertEqual(r["friendly_collision_retained"], [18.0, 0.0])
        self.assertEqual(r["move_cost_burn"], [3.0, 4.0])
        self.assertEqual(r["cash_change"], [0.0, 0.0])
        self.assertEqual(r["real_new_harvest"], [0.0, 0.0])
        self.assertEqual(r["enemy_collision"]["cargo"], 36.0)
        self.assertEqual(after["players"][0][2][uid(0, 0)][1], 54.0)

    def test_ground_cap_and_expired_tree(self):
        for expired in (False, True):
            with self.subTest(expired=expired):
                r, after = self.transition(
                    [(0, 0, 49, 20), (0, 1, 51, 20), (1, 0, 35, 20)],
                    [(0, 0, "RIGHT"), (0, 1, "LEFT"), (1, 0, "DOWN")],
                    trees=[(0, 0, 50, 50)] if expired else [],
                    tiles=[(50, 95 if not expired else 0)],
                    config={
                        "moveCost": 0.0,
                        "maxCellCarbon": 100.0,
                        "co2FrmWithered": 70.0,
                        "treeLifespan": 50,
                    },
                )
                admitted = 30.0 if expired else 5.0
                self.assertEqual(r["ground_collision_admitted"], [admitted, 0.0])
                self.assertEqual(r["ground_collision_overflow"], [40.0 - admitted, 20.0])
                self.assertTrue(r["own_died"] and r["enemy_died"])
                self.assertEqual(r["transfer_margin"], 0.0)
                self.assertEqual(after["carbon"][50], 100.0)

    def test_collision_direct_cash_on_tree_and_base(self):
        b1 = FastCarbon().reset(seed=0).bases[1][0][1]
        r, _ = self.transition(
            [(0, 0, 49, 20), (1, 0, 51, 20), (0, 1, b1 - 1, 6), (1, 1, b1 + 1, 6)],
            [(0, 0, "RIGHT"), (1, 0, "LEFT"), (0, 1, "RIGHT"), (1, 1, "LEFT")],
            trees=[(0, 0, 50, 15)],
            config={"moveCost": 0.0},
            perspective=1,
        )
        self.assertEqual(r["cross_to_tree_cash"], [20.0, 0.0])
        self.assertEqual(r["cross_to_base_cash"], [0.0, 6.0])
        self.assertEqual(r["cash_change"], [40.0, 12.0])
        self.assertEqual(r["transfer_margin"], -14.0)
        self.assertEqual(r["friendly_collision_retained"], [20.0, 6.0])
        self.assertEqual(r["own_collision"]["destination"], "tree_cash")

    def test_capture_then_immediate_deposit_count_once(self):
        base = FastCarbon().reset(seed=0).bases[0][0][1]
        r, _ = self.transition(
            [(0, 0, base, 2), (1, 0, base + 1, 50)],
            [(1, 0, "LEFT")],
            config={"moveCost": 0.0, "smeltCost": 1.0},
        )
        self.assertEqual(r["transfer_margin"], 50.0)
        self.assertEqual(r["cross_to_worker"], [50.0, 0.0])
        self.assertEqual(r["gross_cargo_unloaded_by_worker_owner"], [52.0, 0.0])
        self.assertEqual(r["cash_change"], [51.0, 0.0])
        self.assertEqual(r["smelt_fee"], [1.0, 0.0])
        self.assertEqual(r["real_new_harvest"], [0.0, 0.0])

    def test_hostile_tree_tax_and_enemy_delivery_stay_off_target(self):
        base = FastCarbon().reset(seed=0).bases[1][0][1]
        r, _ = self.transition(
            [(0, 0, 50, 40), (1, 0, 150, 0), (0, 1, base - 1, 70)],
            [(0, 1, "RIGHT")],
            trees=[(1, 0, 51, 4)],
            config={"moveCost": 0.0, "collectorAbsorptionRate": 0.15, "smeltCost": 2.0},
        )
        self.assertEqual(r["transfer_margin"], 0.0)
        self.assertAlmostEqual(r["tree_tax_from_enemy_cargo"][1], 6.0)
        self.assertEqual(r["cash_change"], [0.0, 74.0])
        self.assertEqual(r["all_phases_cross_flow_margin"], -74.0)


if __name__ == "__main__":
    unittest.main()
