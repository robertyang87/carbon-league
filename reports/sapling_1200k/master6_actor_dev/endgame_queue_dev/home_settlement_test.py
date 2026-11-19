"""Synthetic one- and two-turn settlement tests. No agents, map draws or game tapes."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[4] / "code"
ENGINE = CODE / "env_src/carbon_challenge-master"
sys.path[:0] = [str(CODE), str(ENGINE)]
from fastsim import FastCarbon
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.carbon import compute_next_board
from zerosum_env.utils import structify


def worker_id(seat, index):
    return f"player-{seat}-worker-{index}"


class HomeSettlement(unittest.TestCase):
    def setUp(self):
        self.reset_board()

    def reset_board(self):
        self.sim = FastCarbon().reset(full_carbon=[0] * 225)
        self.obs = self.sim.full_observation()
        self.orders = [{}, {}]

    def incoming(self, destination, workers):
        """Place each (seat, cargo) on a distinct orthogonal approach."""
        approaches = [(-1, "RIGHT"), (1, "LEFT"), (-15, "DOWN"), (15, "UP")]
        assert len(workers) <= len(approaches)
        for index, ((offset, action), (seat, cargo)) in enumerate(zip(approaches, workers)):
            uid = worker_id(seat, index)
            self.obs["players"][seat][2][uid] = [destination + offset, cargo, "COLLECTOR"]
            self.orders[seat][uid] = action

    def transition(self):
        """Compare full board state, then read rewards from the official interpreter."""
        sim = self.sim.load_observation(self.obs)
        counters = {f"player-{p}-{role}-": values[p] for p in (0, 1)
                    for role, values in (("worker", sim.worker_counters), ("tree", sim.tree_counters))}
        public = dict(self.obs, full_carbon=self.obs["carbon"])
        states = structify([{"status": "ACTIVE", "reward": pl[0], "action": self.orders[p],
                             "observation": public if p == 0 else {"player": 1}}
                            for p, pl in enumerate(self.obs["players"])])
        with patch.dict(idgen._id_dict, counters, clear=True):
            official = compute_next_board(states, structify({"configuration": sim.config}))
        sim.step(*self.orders)
        after = sim.full_observation()
        observed = official[0].observation
        self.assertEqual(after["carbon"], observed.full_carbon)
        for field in ("players", "trees", "step"):
            self.assertEqual(after[field], observed[field], field)
        self.obs = after
        self.orders = [{}, {}]
        return [p.status for p in official], [p.reward for p in official]

    def test_source_defaults(self):
        specification = json.loads((ENGINE / "zerosum_env/envs/carbon/carbon.json").read_text())
        values = {"moveCost": 0, "smeltCost": 0, "numberOfBases": 2, "recCollectorCost": 30,
                  "recPlanterCost": 30, "recCollectorIncreaseCost": 0, "recPlanterIncreaseCost": 0,
                  "collectorAbsorptionRate": 0.1, "workerLimit": 10}
        for key, value in values.items():
            self.assertEqual(specification["configuration"][key]["default"], value, key)
            self.assertEqual(self.sim.config[key], value, key)
        self.assertEqual([len(self.sim.bases[p]) for p in (0, 1)], [2, 2])

    def test_home_collision_deposits_and_surviving_workers(self):
        # Entries are relative to the base owner. The last field identifies the survivor by input index.
        scenarios = [
            ("two unequal", [(0, 70), (0, 20)], 1),
            ("two equal", [(0, 40), (0, 40)], None),
            ("three unequal", [(0, 70), (0, 30), (0, 10)], 2),
            ("three equal", [(0, 40), (0, 40), (0, 40)], None),
            ("three tied for minimum", [(0, 55), (0, 20), (0, 20)], None),
            ("enemy lightest", [(0, 60), (0, 25), (1, 5)], 2),
            ("enemy ties minimum", [(0, 60), (0, 25), (1, 25)], None),
            ("enemy heaviest", [(0, 60), (0, 25), (1, 80)], 1),
            ("all four approach", [(0, 60), (0, 40), (0, 20), (1, 0)], 3),
            ("enemy alone and empty", [(1, 0)], None),
        ]
        for owner, base_index in ((0, 0), (1, 1)):
            for label, relative, survivor in scenarios:
                with self.subTest(owner=owner, case=label):
                    self.reset_board()
                    workers = [(owner ^ side, amount) for side, amount in relative]
                    base = self.sim.bases[owner][base_index][1]
                    self.incoming(base, workers)
                    self.transition()
                    self.assertEqual(self.sim.cash[owner], 80 + sum(amount for _, amount in workers))
                    self.assertEqual(self.sim.cash[1 - owner], 80)
                    live = set() if survivor is None else {worker_id(workers[survivor][0], survivor)}
                    self.assertEqual(set(self.sim.workers), live)
                    self.assertTrue(all(w.pos == base and w.carbon == 0 for w in self.sim.workers.values()))

    def test_recruitment_charges_and_identity(self):
        for owner in (0, 1):
            for command, own_money, survivor_index in (
                    ("RECCOLLECTOR", 30, 2), ("RECPLANTER", 30, 1), ("RECCOLLECTOR", 29.99, 1)):
                with self.subTest(owner=owner, command=command, cash=own_money):
                    self.reset_board()
                    center, home = self.sim.bases[owner][0]
                    self.obs["players"][owner][0] = own_money
                    self.incoming(home, [(owner, 70), (owner, 20)])
                    self.orders[owner][center] = command
                    # Opponent may hire at its own base, but cannot recruit at this contested center.
                    other_center, other_home = self.sim.bases[1 - owner][0]
                    self.orders[1 - owner].update({other_center: "RECCOLLECTOR", center: "RECCOLLECTOR"})
                    self.transition()
                    paid = 30 if own_money >= 30 else 0
                    self.assertAlmostEqual(self.sim.cash[owner], own_money + 90 - paid)
                    self.assertEqual(self.sim.cash[1 - owner], 50)
                    survivor = worker_id(owner, survivor_index)
                    opponent_recruit = worker_id(1 - owner, 0)
                    self.assertEqual(set(self.sim.workers), {survivor, opponent_recruit})
                    self.assertEqual((self.sim.workers[survivor].pos, self.sim.workers[survivor].carbon), (home, 0))
                    self.assertEqual(self.sim.workers[opponent_recruit].pos, other_home)

    def test_off_base_collision_transfers_cargo_or_drops_it(self):
        cell = self.sim.bases[0][0][1] + 2
        for loads, survivor, carried, ground in [
                ((80, 20), worker_id(1, 1), 100, 0), ((65, 65), None, 0, 100)]:
            with self.subTest(loads=loads):
                self.reset_board()
                self.incoming(cell, [(0, loads[0]), (1, loads[1])])
                self.transition()
                self.assertEqual(self.sim.cash, [80, 80])
                self.assertEqual(set(self.sim.workers), {survivor} if survivor else set())
                self.assertEqual(sum(w.carbon for w in self.sim.workers.values()), carried)
                self.assertEqual(self.sim.carbon[cell], ground)

    def test_arrivals_on_enemy_recruiting_center_pay_the_enemy(self):
        center, enemy_home = self.sim.bases[1][0]
        self.incoming(enemy_home, [(0, 70), (0, 20)])
        self.orders[1][center] = "RECCOLLECTOR"
        self.transition()
        self.assertEqual(self.sim.cash, [80, 80 - 30 + 90])
        recruit = worker_id(1, 0)
        self.assertEqual(set(self.sim.workers), {recruit})
        self.assertEqual((self.sim.workers[recruit].pos, self.sim.workers[recruit].carbon), (enemy_home, 0))

    def test_enemy_tree_tax_precedes_only_later_turns_deposit(self):
        home = self.sim.bases[0][0][1]
        uid, tid = worker_id(0, 0), "player-1-tree-0"
        self.obs["players"][0][2][uid] = [home + 2, 100, "COLLECTOR"]
        self.obs["players"][1][3][tid] = [home + 16, 6]
        self.obs["trees"][tid] = [None, 0]
        self.orders[0][uid] = "LEFT"
        self.transition()
        self.assertEqual((self.sim.cash, self.sim.workers[uid].carbon), ([80, 90], 90))
        self.orders[0][uid] = "LEFT"
        self.transition()
        self.assertEqual((self.sim.cash, self.sim.workers[uid].carbon), ([170, 90], 0))

    def test_official_extinction_scores_and_recruitment_boundary(self):
        home = self.sim.bases[0][0][1]
        for rival_cash, rival_tree, own_start, expected_cash, statuses, rewards in [
            (29.99, False, 20, [40, 29.99], ["DONE", "DONE"], [20, -3]),
            (30, False, 20, [40, 30], ["ACTIVE", "ACTIVE"], [40, 30]),
            (29.99, True, 20, [40, 29.99], ["ACTIVE", "ACTIVE"], [40, 29.99]),
            (29.99, False, 5, [25, 29.99], ["DONE", "DONE"], [-3, -3]),
        ]:
            with self.subTest(rival_cash=rival_cash, tree=rival_tree, own_start=own_start):
                self.reset_board()
                self.obs["step"] = 297
                self.obs["players"][0][0] = own_start
                self.obs["players"][1][0] = rival_cash
                if rival_tree:
                    self.obs["players"][1][3]["player-1-tree-0"] = [5, 8]
                    self.obs["trees"]["player-1-tree-0"] = [None, 0]
                self.incoming(home, [(0, 10), (1, 10)])
                self.assertEqual(self.transition(), (statuses, rewards))
                self.assertEqual(self.sim.cash, expected_cash)
                self.assertEqual(self.sim.workers, {})

    def test_transfer_formula_is_configuration_specific(self):
        home = self.sim.bases[0][0][1]
        for loads, credit in (((10, 90), 80), ((50, 50), 90)):
            with self.subTest(loads=loads):
                self.reset_board()
                self.sim.config.update(moveCost=0.1, smeltCost=10)
                self.incoming(home, [(0, loads[0]), (0, loads[1])])
                self.transition()
                self.assertEqual(self.sim.cash, [80 + credit, 80])


if __name__ == "__main__":
    unittest.main()
