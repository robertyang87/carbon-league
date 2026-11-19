import unittest
from chooser import choose, by_cell


class Lane(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(
            size=15,
            episodeSteps=300,
            treeLifespan=50,
            moveCost=0,
            initialCollectRate=0.3,
            collectDecreaseRate=0.0,
            workerLimit=10,
            recCollectorCost=20,
            recCollectorIncreaseCost=0,
        )
        self.obs = dict(
            step=20,
            player=0,
            carbon=[0.0] * 225,
            players=[
                [300, {"base": 51}, {"f": [51, 0, "COLLECTOR"]}, {}],
                [200, {"eb": 59}, {"e": [52, 200, "COLLECTOR"]}, {}],
            ],
        )
        self.fore = {"e": {3: {54: 0.60, "OTHER": 0.4}, 5: {55: 0.70, "OTHER": 0.3}}}

    def run_policy(self, old="LEFT"):
        return choose(self.obs, {"f": old}, self.cfg, self.fore)

    def test_partial_baseline_and_earlier_alternative(self):
        self.obs["players"][0][2]["f"][0] = 52
        self.obs["players"][0][1]["base"] = 45
        got = self.run_policy()
        self.assertTrue(got["triggered"])
        self.assertEqual(got["commands"], {"f": "RIGHT"})
        self.assertEqual(got["candidate"]["reachable_alt"], [54, 55])
        self.assertEqual(got["candidate"]["p_route"], [0.6, 0.7])
        # The order can cover both times geometrically; no captured cargo is credited in this function.
        self.obs["players"][0][2]["f"][0] = 53
        got = self.run_policy()
        self.assertEqual(got["candidate"]["reachable_v1"], [55])
        self.assertEqual(got["candidate"]["score_v1"], 0.35)

    def test_actual_bank_guard_and_friendly_block(self):
        self.obs["players"][0][2]["f"] = [52, 10, "COLLECTOR"]
        self.assertFalse(self.run_policy()["triggered"])
        self.obs["players"][0][1]["base"] = 45
        self.obs["players"][0][2]["g"] = [53, 0, "PLANTER"]
        self.assertFalse(self.run_policy()["triggered"])

    def test_cash_opportunity_and_mass(self):
        self.obs["players"][0][2]["f"][0] = 52
        self.obs["players"][0][1]["base"] = 45
        self.obs["carbon"][52] = 100.0
        self.fore = {"e": {3: {54: 0.1, "OTHER": 0.9}, 5: {55: 0.1, "OTHER": 0.9}}}
        got = self.run_policy(old="NONE")
        self.assertFalse(got["triggered"])
        self.assertEqual(got["candidate"]["harvest_opportunity_cost"], 30.0)
        self.assertEqual(got["commands"], {"f": "NONE"})
        with self.assertRaises(ValueError):
            by_cell({3: {54: 0.6, "OTHER": 0.6}}, 3)

    def test_cash_no_five_turn_horizon(self):
        self.obs["step"] = 295
        self.assertIsNone(self.run_policy()["candidate"])


if __name__ == "__main__":
    unittest.main()
