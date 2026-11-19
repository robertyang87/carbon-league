import unittest
from scan import summary


class EliminationCases(unittest.TestCase):
    def test_workers_trees_bases_and_cash(self):
        cfg = {"recCollectorCost":30, "recCollectorIncreaseCost":0, "recPlanterCost":30}
        def state(cash, workers=None, trees=None, bases=True):
            player = [cash, {"b0":2, "b1":37} if bases else {}, workers or {}, trees or {}]
            return summary([player, [80, {}, {}, {}]],0,cfg)
        self.assertFalse(state(13.95, {"planter":[5,0,"PLANTER"]})["official_extinct"])
        self.assertFalse(state(0, {"collector":[5,0,"COLLECTOR"]})["official_extinct"])
        self.assertFalse(state(0, trees={"tree":[4,12]})["official_extinct"])
        self.assertFalse(state(30)["official_extinct"])
        self.assertTrue(state(13.95)["official_extinct"])
        self.assertTrue(state(100,bases=False)["official_extinct"])


if __name__ == "__main__":
    unittest.main()
