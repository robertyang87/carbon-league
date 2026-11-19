"""Regression tests for the pre-open gate, physical pairs, frozen comparison and exclusivity."""

from contextlib import ExitStack
import fcntl
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import development as dev


class TestDevelopment(unittest.TestCase):
    def test_closed_uses_no_development_paths_or_native_workers(self):
        for reason in ("insufficient_realized_support", "heldout_gate_failed"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                area = Path(tmp) / "untouched"
                stack.enter_context(patch.object(dev, "RUN", area))
                stack.enter_context(
                    patch.object(
                        dev.train, "rebuild_release", side_effect=ValueError("no eligible release: " + reason)
                    )
                )
                forbidden = [
                    stack.enter_context(patch.object(dev, n))
                    for n in ("schedule_locks", "verify_maps", "freeze_packages", "official_arm")
                ]
                self.assertEqual(
                    dev.play(),
                    {
                        "stage": reason,
                        "started_this_call": 0,
                        "opened_development_maps_this_call": 0,
                        "formal": False,
                    },
                )
                self.assertFalse(area.exists())
                for fn in forbidden:
                    fn.assert_not_called()

    def test_release_identity_error_is_fatal(self):
        with patch.object(dev.train, "rebuild_release", side_effect=ValueError("altered real release")):
            with self.assertRaisesRegex(ValueError, "altered real release"):
                dev.play()

    def comparison(self, left, right):
        def row(m):
            return {
                "verified_edits": [],
                "margin": m,
                "U": dev.trial.terminal(m)[0],
                "result": "win" if m > 0 else "loss" if m < 0 else "draw",
            }

        keys = [(7, 0), (7, 1), (8, 0), (8, 1)]
        return {
            k: {key: row(m) for key, m in zip(keys, values)}
            for k, values in (("parent", left), ("candidate", right))
        }

    def test_four_conjunctive_map_and_seat_conditions(self):
        cases = [
            ([0] * 4, [100] * 4, "development_pass", []),
            ([-200] * 4, [-100] * 4, "development_pass", []),  # development-only; still zero wins
            (
                [0] * 4,
                [0, 0, 200, 200],
                "retain_v1",
                ["both_map_margin_gains_positive", "positive_U_in_at_least_three_pairs"],
            ),
            ([0] * 4, [99] * 4, "retain_v1", ["equal_map_mean_at_least_100"]),
            ([0] * 4, [300, -50, 300, -50], "retain_v1", ["positive_U_in_at_least_three_pairs"]),
            ([1, -1000, -1000, -1000], [-1, -600, -600, -600], "retain_v1", ["no_fewer_wins"]),
        ]
        for left, right, expected, failed in cases:
            with self.subTest(left=left, right=right):
                result = dev.compare([7, 8], self.comparison(left, right))
                self.assertEqual(result["verdict"], expected)
                self.assertEqual([k for k, v in result["criteria"].items() if not v], failed)
                self.assertFalse(result["formal"])
                self.assertEqual(len(result["pairs"]), 4)

    def test_missing_swapped_seat_and_duplicate_map_fail(self):
        arms = self.comparison([0] * 4, [100] * 4)
        with self.assertRaises(ValueError):
            dev.compare([7, 7], arms)
        del arms["candidate"][8, 1]
        with self.assertRaises(ValueError):
            dev.compare([7, 8], arms)

    def test_archive_replay_from_four_training_games_without_launching_an_agent(self):
        source = dev.INTAKE / "source"
        native = dev.trial.read(source / "results.json")
        header = native["header"]
        official = dev.trial.read(source / "official_replay.json")
        binding = {
            "maps": header["seeds"],
            "cpus": header["cpus"],
            "arena_sources": header["source_sha256"],
            "official_engine": official["engine_source_sha256"],
            "packages": {"parent": {k: header[k] for k in ("pack_sha256", "weight_sha256")}},
            "boards": {str(r["seed"]): r["map_sha256"] for r in official["runs"]["source"]["games"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            past = run / "parent"
            past.mkdir()
            for item in [source / "results.json", *source.glob("raw_*.json.gz")]:
                os.link(item, past / item.name)
            real_run = subprocess.run

            def replay_only(cmd, **kwargs):
                if cmd[2].endswith("tree_pack_arena.py"):
                    return
                self.assertTrue(cmd[1].endswith("official_game_replay.py"))
                return real_run(cmd, **kwargs)

            with patch.object(dev, "RUN", run), patch.object(
                dev, "other_native", return_value=[]
            ), patch.object(dev.subprocess, "run", side_effect=replay_only):
                rows = dev.official_arm("parent", (dev.deploy.PARENT, dev.INTAKE / "protocol.json"), binding)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["steps"] == row["search_invocations"] for row in rows.values()))
            self.assertEqual(
                {
                    (r["seed"], r["seat"]): r["official_reward_by_seat"]
                    for r in official["runs"]["source"]["games"]
                },
                {k: r["official_reward_by_seat"] for k, r in rows.items()},
            )

    def test_live_intake_refuses_shared_lane_and_releases_all_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            area = Path(tmp)
            (area / "source").mkdir()
            (area / "batches" / "intake_04").mkdir(parents=True)
            occupied = area / "batches/intake_04/.intake.schedule.lock"
            with patch.object(dev, "INTAKE", area):
                with occupied.open("a") as h, ExitStack() as stack:
                    fcntl.flock(h, fcntl.LOCK_EX)
                    with self.assertRaises(BlockingIOError):
                        dev.schedule_locks(stack)
                with ExitStack() as stack:
                    dev.schedule_locks(stack)
                    with self.assertRaises(BlockingIOError), ExitStack() as duplicate:
                        dev.schedule_locks(duplicate)


if __name__ == "__main__":
    unittest.main()
