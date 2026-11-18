"""Targeted checks for recruitment-opportunity diagnostics and loss reweighting."""
import unittest

import numpy as np

from sapling_bootstrap import effective_unit_weights, opportunity_counts, role_action_weights


class BootstrapDiagnosticsTest(unittest.TestCase):
    def test_forced_waits_do_not_inflate_recruitment_opportunities(self):
        y = np.array([[0, 1, 0, -1]])
        kind = np.array([[2, 2, 2, 2]])
        legal = np.zeros((1, 4, 5), bool)
        legal[..., 0] = True
        legal[0, 1:3, 1:3] = True
        legal[0, 3, :] = True  # absent slots must remain excluded
        pred = np.array([[0, 2, 0, 1]])
        got = opportunity_counts(y, kind, legal, pred)
        self.assertEqual(got["base_forced_none_n"], 1)
        self.assertEqual(got["base_recruit_available"],
                         dict(n=2, teacher_recruits=1, predicted_recruits=1, correct=1))
        self.assertEqual(got["base"]["n"], 3)
        self.assertIsNone(got["collector"]["accuracy"])

    def test_balancing_equalizes_supported_role_action_mass(self):
        # Unequal populations: collector NONE x3, movement x1, base NONE x1,
        # recruit x1. No validation rows enter the weighting calculation.
        y = np.array([[0, 0, 0, 1, 0, 1, -1]])
        kind = np.array([[0, 0, 0, 0, 2, 2, 5]])
        w, report = role_action_weights({"y": y, "tok_kind": kind}, cap=100)
        self.assertAlmostEqual(float(w[y >= 0].mean()), 1.0)
        self.assertEqual(float(w[0, -1]), 0.0)
        masses = [float(w[0, :3].sum()), float(w[0, 3]), float(w[0, 4]), float(w[0, 5])]
        self.assertTrue(np.allclose(masses, 1.5))
        self.assertEqual(report["counts"][1], [0] * 5)

    def test_empty_supervision_fails(self):
        with self.assertRaisesRegex(ValueError, "supervision"):
            role_action_weights({"y": np.array([[-1]]), "tok_kind": np.array([[5]])})

    def test_sqrt_ablation_only_tempers_existing_weights(self):
        w = np.array([[.25, 4, 0]], np.float32)
        valid = np.array([[True, True, False]])
        actual = effective_unit_weights(w, valid, "sqrt-role-action")
        np.testing.assert_allclose(actual, [[.4, 1.6, 0]])
        self.assertEqual(float(actual[valid].mean()), 1.0)
        np.testing.assert_array_equal(w, [[.25, 4, 0]])

    def test_supported_action_macro_recall_is_not_overall_accuracy(self):
        y = np.array([[0, 0, 0, 0, 1]])
        kind = np.full_like(y, 2)
        legal = np.ones((1, 5, 5), bool)
        got = opportunity_counts(y, kind, legal, np.zeros_like(y))["base"]
        self.assertEqual(got["accuracy"], .8)
        self.assertEqual(got["macro_recall_supported_actions"], .5)


if __name__ == "__main__":
    unittest.main()
