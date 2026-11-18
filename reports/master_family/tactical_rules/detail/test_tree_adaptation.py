"""Regression cases for causal ordering, censored actions, and match isolation."""
import unittest

import numpy as np

from tree_adaptation import AdaptiveTreeIntent, HALF_LIFE, PRIOR_WEIGHT

CFG = {"size": 15, "seizeCost": 20, "treeLifespan": 50}
FROZEN = {"p": np.array([.4, .1, .2, .1, .2])}


def scene(t, pos=64, guard=63):
    enemies = {} if pos is None else {"p": [pos, 0, "PLANTER"]}
    return {"player": 0, "step": t, "players": [
        [100, {}, {"c": [guard, 0, "COLLECTOR"]}, {"tree": [64, 6]}],
        [100, {}, enemies, {}]]}


class TestAdaptiveTreeIntent(unittest.TestCase):
    def test_stay_and_two_queries(self):
        memory = AdaptiveTreeIntent()
        first = scene(0)
        memory.observe(first, CFG)
        np.testing.assert_array_equal(memory.probabilities(first, CFG, FROZEN)["p"], FROZEN["p"])
        current = scene(1)
        memory.observe(current, CFG)
        memory.observe(current, CFG)
        self.assertEqual(memory.learned, 1)
        p = memory.probabilities(current, CFG, FROZEN)["p"]
        self.assertAlmostEqual(p[0], (PRIOR_WEIGHT * .4 + 1) / (PRIOR_WEIGHT + 1))
        self.assertAlmostEqual(sum(p), 1)
        self.assertAlmostEqual(p[2] / p[1], 2)
        self.assertAlmostEqual(FROZEN["p"][0], .4)

    def test_dead_unknown_and_new_match(self):
        m = AdaptiveTreeIntent()
        for frame in [scene(0), scene(1, None), scene(2), scene(3, 65), scene(4)]:
            m.observe(frame, CFG)
        self.assertEqual((m.learned, m.censored), (1, 1))
        self.assertLess(m.probabilities(scene(4), CFG, FROZEN)["p"][0], .4)
        m.observe(scene(0), CFG)
        self.assertEqual(m.learned, 0)
        self.assertEqual(m.censored, 0)

    def test_context_and_gaps(self):
        m = AdaptiveTreeIntent()
        m.observe(scene(0), CFG)
        m.observe(scene(1, guard=4), CFG)
        np.testing.assert_array_equal(m.probabilities(scene(1, guard=4), CFG, FROZEN)["p"], FROZEN["p"])
        m.observe(scene(1 + int(HALF_LIFE)), CFG)
        self.assertEqual(m.learned, 1)
        self.assertAlmostEqual(m.counts[1, 1], .5)

    def test_unobserved_step_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveTreeIntent().probabilities(scene(12), CFG, FROZEN)


if __name__ == "__main__":
    unittest.main()
