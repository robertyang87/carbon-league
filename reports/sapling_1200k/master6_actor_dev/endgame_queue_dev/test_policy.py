"""Construct positions from public observations; no source games, no fitted parameters."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "code"))
from fastsim import FastCarbon
from policy import choose, orders


class QueueDecision(unittest.TestCase):
    def at(self, seat, steps_left, workers):
        sim = FastCarbon().reset(full_carbon=[0] * 225)
        obs = sim.full_observation(seat)
        home = sim.bases[seat][0][1]
        ids = []
        for i, (offset, cargo, side) in enumerate(workers):
            uid = f"player-{side}-worker-{i}"
            ids.append(uid)
            obs["players"][side][2][uid] = [home + offset, cargo, "COLLECTOR"]
        obs["step"] = 299 - steps_left
        return sim.load_observation(obs).obs(seat), sim.config, home, ids

    def test_both_loaded_collectors_credited_at_one_bank(self):
        for seat in (0, 1):
            obs, cfg, home, (a, b) = self.at(seat, 1, [(-1, 70, seat), (1, 20, seat)])
            chosen, audit = choose(obs, {}, {}, cfg)
            self.assertEqual(chosen, {a: "RIGHT", b: "LEFT"})
            self.assertEqual(audit["per_scenario_gain"], [90] * 3)
            self.assertEqual(audit["selected_shared_bank"][0]["cell"], home)
            self.assertEqual(audit["selected_shared_bank"][0]["inbound_loaded"], 2)

    def test_no_new_zero_gain_or_below_threshold_edit(self):
        obs, cfg, _, (a, b) = self.at(0, 1, [(-1, 3, 0), (1, 1, 0)])
        self.assertEqual(choose(obs, {}, {}, cfg)[0], {})
        own = {a: "RIGHT", b: "LEFT"}
        self.assertEqual(choose(obs, {}, own, cfg)[0], own)

    def test_two_step_route_and_hidden_information_invariance(self):
        obs, cfg, _, (a,) = self.at(0, 2, [(-2, 93, 0)])
        expected, audit = choose(obs, {}, {}, cfg)
        self.assertEqual(expected, {a: "RIGHT"})
        self.assertEqual(audit["per_scenario_gain"], [93] * 3)
        leaked = copy.deepcopy(obs)
        for i in range(225):
            if i != obs["players"][0][2][a][0]:
                leaked["carbon"][i] = 99
        leaked["full_carbon"] = [99] * 225
        future = {298: dict(obs, step=298), 312: {}}
        altered, alt_audit = choose(leaked, future, {}, dict(cfg, randomSeed=937))
        self.assertEqual(altered, expected)
        self.assertEqual(alt_audit["per_scenario_gain"], audit["per_scenario_gain"])

    def test_terminal_orders_preserve_all_other_workers_and_recruit(self):
        obs, cfg, home, (a, b) = self.at(0, 1, [(-1, 90, 0), (2, 0, 0)])
        center = next(k for k,v in obs["players"][0][1].items() if v == home)
        offered = {b: "DOWN", center: "RECCOLLECTOR"}
        chosen, _ = choose(obs, {}, offered, cfg)
        self.assertEqual(chosen, {**offered, a: "RIGHT"})
        self.assertEqual(offered, {b: "DOWN", center: "RECCOLLECTOR"})

    def test_outside_enemy_home_blocker_must_pay_in_margin(self):
        # Worker is beside the only staging cell on the rival's two-step route.
        obs, cfg, home, _ = self.at(1, 2, [])
        obs["player"] = 0
        a, e = "player-0-worker-0", "player-1-worker-0"
        obs["players"][0][2][a] = [home + 16, 2, "COLLECTOR"]
        obs["players"][1][2][e] = [home + 2, 203, "COLLECTOR"]
        chosen, audit = choose(obs, {}, {a: "DOWN"}, cfg)
        self.assertEqual(audit["blocker"], a)
        self.assertEqual(audit["enemy_target"], e)
        # This is an attractive collision, but a simulated surviving interceptor
        # carries the prize across the rival bank on its final route. Veto it.
        self.assertEqual(chosen[a], "DOWN")
        self.assertEqual(audit["reason"], "below_gain_gate")
        obs["players"][1][2][e][0] = home + 30
        chosen, audit = choose(obs, {}, {}, cfg)
        self.assertEqual(chosen[a], "LEFT")
        self.assertEqual(audit["per_scenario_gain"], [203] * 3)
        # On the next tick there is no outside-bank interception left.
        obs["step"] = 298
        obs["players"][1][2][e][0] = home + 1
        self.assertIsNone(choose(obs, {}, {}, cfg)[1]["blocker"])

    def test_foreign_and_future_terrain_do_not_enter_motion_forecasts(self):
        obs, cfg, _, (a,) = self.at(0, 2, [(-1, 55, 0)])
        future = {298: dict(obs, step=298)}
        self.assertEqual(orders(obs, {}, cfg, 1), orders(obs, future, cfg, 1))

if __name__ == "__main__":
    unittest.main()
