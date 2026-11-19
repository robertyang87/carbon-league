"""Checks on physical one- and two-step official Board / FastCarbon transitions and sealed real roots."""

import gzip
import json
from pathlib import Path
import unittest

import lots

HERE = Path(__file__).resolve().parent


def uid(seat, num):
    return f"player-{seat}-worker-{num}"


def state(workers, *, trees=(), tiles=(), cfg=None):
    sim = lots.FastCarbon(cfg).reset(full_carbon=[0.0] * 225)
    obs = sim.full_observation()
    for p, i, pos, load in workers:
        obs["players"][p][2][uid(p, i)] = [pos, load, "COLLECTOR"]
    for p, i, pos, age in trees:
        tid = f"player-{p}-tree-{i}"
        obs["players"][p][3][tid] = [pos, age]
        obs["trees"][tid] = [None, 0.0]
    for cell, v in tiles:
        obs["carbon"][cell] = v
    return sim.load_observation(obs)


def obs(data):
    return json.loads(gzip.decompress(data.read_bytes()))


class LotsTest(unittest.TestCase):
    def test_relay_with_movement_fees_tree_tax_and_new_harvest(self):
        base = lots.FastCarbon().reset(full_carbon=[0.0] * 225).bases[0][0][1]
        a, b = uid(0, 0), uid(0, 1)
        sim = state(
            [(0, 0, base + 2, 100.0), (0, 1, base + 1, 10.0)],
            trees=[(1, 0, base + 16, 11)],
            tiles=[(base + 1, 30.0)],
            cfg={"moveCost": 0.1, "smeltCost": 0.0, "collectorAbsorptionRate": 0.1},
        )
        result = lots.from_sim(sim, [[{a: "LEFT"}, {}], [{b: "LEFT"}, {}]])
        for u, value, kind in ((a, 72.9, "own_relay"), (b, 8.1, "own_direct")):
            summary = result.summary(0, u)
            self.assertAlmostEqual(summary["amounts"][kind], value)
            self.assertAlmostEqual(summary["mean_steps_to_own_cash_conditional"], 2.0)
            events = result.events(0, u)
            self.assertAlmostEqual(sum(e["fraction"] for e in events if e["is_terminal_allocation"]), 1.0)
            self.assertEqual([e["step"] for e in events if e["kind"] == kind], [2])
        ea = result.events(0, a)
        self.assertEqual(
            [(e["step"], e["holder_uid"], e["recipient_uid"]) for e in ea if e["kind"] == "relay"],
            [(1, a, b)],
        )
        self.assertAlmostEqual(result.summary(0, a)["amounts"]["tax"], 9.0)
        self.assertAlmostEqual(result.summary(0, a)["amounts"]["ground_or_cost"], 18.1)
        self.assertGreater(
            sim.cash[0] - 80, 72.9 + 8.1
        )  # freshly harvested carbon sits outside both original lots

    def test_enemy_gets_it_even_if_original_side_recaptures_next_step(self):
        a, b, c = uid(0, 0), uid(1, 0), uid(0, 1)
        sim = state([(0, 0, 49, 100.0), (1, 0, 50, 0.0), (0, 1, 52, 0.0)])
        result = lots.from_sim(sim, [[{a: "RIGHT"}, {}], [{c: "LEFT"}, {b: "RIGHT"}]])
        self.assertEqual(sim.workers[c].carbon, 100.0)
        self.assertEqual(
            result.events(0, a),
            [
                {
                    "step": 1,
                    "kind": "lost_to_enemy",
                    "reason": "enemy_worker",
                    "holder_uid": a,
                    "recipient_uid": b,
                    "fraction": 1.0,
                    "amount": 100.0,
                    "is_terminal_allocation": True,
                }
            ],
        )
        self.assertNotIn((0, b), result.cargo)
        self.assertEqual(result.summary(1, b)["fractions"]["lost_to_enemy"], 1.0)

    def test_explicit_stock_then_ground_cap_allocation(self):
        a, b = uid(0, 0), uid(1, 0)
        sim = state([(0, 0, 49, 20.0), (1, 0, 51, 20.0)], tiles=[(50, 95.0)])
        data = lots.from_sim(sim, [[{a: "RIGHT"}, {b: "LEFT"}]])
        self.assertEqual(
            {e["reason"]: e["amount"] for e in data.events(0, a)},
            {"ground_admitted": 5.0, "ground_overflow": 15.0},
        )
        self.assertEqual(data.events(0, b)[0]["amount"], 20.0)
        self.assertEqual(data.events(0, b)[0]["reason"], "ground_overflow")
        self.assertEqual(sim.carbon[50], 100.0)

    def test_cash_to_tree_owner_with_mixed_seats(self):
        a, b = uid(0, 0), uid(1, 0)
        sim = state([(0, 0, 49, 20.0), (1, 0, 51, 20.0)], trees=[(0, 0, 50, 20)])
        data = lots.from_sim(sim, [[{a: "RIGHT"}, {b: "LEFT"}]])
        self.assertEqual(data.summary(0, a)["fractions"]["own_direct"], 1.0)
        self.assertEqual(data.events(0, b)[0]["reason"], "enemy_tree_collision")
        self.assertEqual(sim.cash[0], 120.0)

    def test_existing_inventory_survives_simultaneous_tax_from_two_trees(self):
        a = uid(0, 0)
        sim = state([(0, 0, 50, 80.0)], trees=[(1, 0, 51, 20), (1, 1, 65, 20)], tiles=[(50, 100.0)])
        data = lots.from_sim(sim, [[{}, {}], [{}, {}]])
        self.assertAlmostEqual(data.summary(0, a)["amounts"]["terminal_inventory"], 80 * 0.8**2)
        self.assertAlmostEqual(data.summary(0, a)["amounts"]["tax"], 80 * (1 - 0.8**2))
        self.assertEqual(sum(e["kind"] == "tax" for e in data.events(0, a)), 4)
        self.assertEqual(data.summary(2, a)["fractions"]["terminal_inventory"], 1.0)
        self.assertIsNone(data.summary(0, a)["mean_steps_to_own_cash_conditional"])

    def test_relay_loses_smelting_fee_proportionally(self):
        base = lots.FastCarbon().reset(full_carbon=[0.0] * 225).bases[0][0][1]
        a, b = uid(0, 0), uid(0, 1)
        sim = state([(0, 0, base + 1, 100.0), (0, 1, base, 20.0)], cfg={"smeltCost": 12.0})
        data = lots.from_sim(sim, [[{a: "LEFT"}, {}]])
        self.assertAlmostEqual(data.summary(0, a)["amounts"]["own_relay"], 90.0)
        self.assertAlmostEqual(data.summary(0, a)["amounts"]["ground_or_cost"], 10.0)
        self.assertAlmostEqual(data.summary(0, b)["amounts"]["own_direct"], 18.0)

    def test_previously_checked_real_deliveries(self):
        reference = json.loads((HERE.parent / "return_value_dev/factual_after.json").read_bytes())
        durations = []
        amounts = []
        for game in reference["games"]:
            if not game["edits"]:
                continue
            p = (
                HERE.parent
                / f'tree_response_training/batches/intake_04/source/raw_{game["seed"]}_{game["seat"]}.json.gz'
            )
            import hashlib

            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), game["tape_sha256"])
            tape = lots.build(obs(p), game["seat"])
            for change in game["edits"]:
                t, u = change["root_t"], change["uid"]
                s = tape.summary(t, u)
                self.assertEqual(s["fractions"]["own_direct"], 1.0)
                self.assertAlmostEqual(s["amounts"]["own_direct"], change["lot"])
                self.assertAlmostEqual(
                    s["mean_steps_to_own_cash_conditional"], change["root_lot_closed_at"] - t
                )
                events = tape.events(t, u)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["step"], change["root_lot_closed_at"])
                durations.append(events[0]["step"] - t)
                amounts.append(events[0]["amount"])
        self.assertEqual(sorted(durations), [13, 16, 17])
        self.assertAlmostEqual(sum(amounts), 539.66, places=2)


if __name__ == "__main__":
    unittest.main()
