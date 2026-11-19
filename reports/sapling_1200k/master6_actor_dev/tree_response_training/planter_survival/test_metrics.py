import unittest

from metrics import evaluate


def pair(seed, seat, t, worker, y, p, geo, paid=0, payable=True):
    actions = [
        {"name": n, "died": d, "paid": paid if n == "NONE" else 0, "p": q, "geometric": g}
        for n, d, q, g in zip(("NONE", "LEFT"), y, p, geo)
    ]
    return dict(seed=seed, seat=seat, t=t, uid=worker, physical_site=20, payable=payable, actions=actions)


class TestMetrics(unittest.TestCase):
    def test_counterfactual_signs_payments_alerts_and_inclusive_thresholds(self):
        data = [
            pair(11, 0, 51, "p1", (True, False), (0.25, 0.2), (False, False)),
            pair(11, 0, 52, "p1", (False, True), (0.5, 0.1), (False, True), paid=20),
            pair(12, 1, 57, "p1", (False, False), (0.2, 0.17), (True, False), paid=11),
            pair(12, 1, 58, "p2", (True, True), (0.1, 0.05), (False, True), payable=False),
        ]
        ret = evaluate(data)
        m = ret["equal_map"]["all"]["metrics"]
        for x in m["paired_deaths"].values():
            self.assertEqual(x["roots"], 1)
        observed = m["policy"]["fitted_p20_delta05"]
        self.assertEqual(
            [
                observed[k]
                for k in (
                    "selected_move",
                    "saved_by_moving",
                    "deaths_caused_by_moving",
                    "net_saved_vs_always_stay",
                    "chosen_purchases",
                    "foregone_actual_stay_purchases",
                    "foregone_purchase_debit",
                )
            ],
            [2, 1, 1, 0, 1, 1, 20],
        )
        self.assertEqual(m["policy"]["geometric"]["foregone_purchase_debit"], 11)
        self.assertEqual(m["alerts"]["fitted_p20"]["stay"]["counts"], {"tp": 1, "fn": 1, "fp": 2, "tn": 0})
        self.assertEqual(m["alerts"]["fitted_p50"]["stay"]["alerted"], 1)
        self.assertEqual(m["alerts"]["fitted_p20"]["paired_roots"]["both_fp_and_fn"], 1)
        self.assertEqual(ret["equal_map"]["can_pay_now"]["counts"]["pairs"], 3)
        self.assertEqual(ret["equal_map"]["can_pay_now"]["counts"]["can_pay_now_and_stay_died"], 1)
        self.assertEqual(ret["counts"]["worker_instances"], 3)
        self.assertEqual(ret["by_physical_seat"]["0"]["all"]["counts"]["worker_instances"], 1)
        self.assertEqual(ret["by_map_and_seat"]["12"]["0"]["pairs"], 0)

    def test_map_weighting_unequal_root_counts_and_empty_strata(self):
        a = pair(21, 0, 1, "p", (False, False), (0, 0), (False, False), payable=False)
        base = evaluate(
            [a] + [pair(22, 0, t, "p", (True, True), (0, 0), (True, True), payable=False) for t in range(3)]
        )
        m = base["equal_map"]["all"]["metrics"]
        self.assertEqual(m["brier"], {"fitted": 0.5, "binary_geometric": 0.0})
        self.assertEqual(m["alerts"]["fitted_p20"]["both_arms"]["counts"]["fn"], 6)
        self.assertEqual(m["alerts"]["fitted_p20"]["both_arms"]["false_negative_rate"], 1.0)
        self.assertIsNone(m["alerts"]["fitted_p20"]["both_arms"]["precision"])
        self.assertEqual(base["equal_map"]["can_pay_now"]["evaluated_maps"], [])
        self.assertIsNone(base["by_physical_seat"]["1"]["all"]["metrics"])

    def test_repeated_roots_and_invalid_actions_rejected(self):
        r = pair(21, 1, 8, "p", (False, False), (0.2, 0.1), (False, False))
        with self.assertRaisesRegex(ValueError, "repeated root"):
            evaluate([r, r])
        with self.assertRaisesRegex(ValueError, "physical movement"):
            evaluate([dict(r, actions=list(reversed(r["actions"])))])
        r["actions"][0]["p"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite range"):
            evaluate([r])


if __name__ == "__main__":
    unittest.main()
