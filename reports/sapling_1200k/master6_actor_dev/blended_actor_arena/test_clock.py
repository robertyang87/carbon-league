"""Exercise the real resume loop with mock policies, an old development map, and a four-step toy terminal."""
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
BASE = ACTOR.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

import clock


def load_driver():
    path = ACTOR / "tree_tenure_dev/pay_policy/native_run.py"
    spec = importlib.util.spec_from_file_location("synthetic_blend_clock_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


class ToyExaminer:
    """Same return counters as the published search, without evaluating any network or search node."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.__dict__.update(fixture.configuration)
        self._RX = self._SEARCH_OK = True
        self._torch = SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1)
        self._net = lambda: None
        self._S_COST = [0.] * 4
        self._S_STATS = dict(searched=0, single=0, aborted=0, changed=0, time=[], depth={}, rolls={}, cands={})
        self._S_REC = None
        self._Bel = object()
        self._ST = {"rng": np.random.RandomState(12345)}
        self.BeliefTracker = lambda *_: SimpleNamespace(step=lambda *_: None, set_actions=lambda *_: None)

    def _search(self, step):
        s = self._S_STATS
        if step == 1:
            s["single"] += 1  # the original deliberately leaves last unchanged
        elif step == 2:
            s["aborted"] += 1
            s["last"] = (4, 0, 1, 0., list(self._S_COST))
        else:
            s["searched"] += 1
            s["changed"] += int(step == 3)
            s["time"].append(0.)
            s["last"] = (4, 8, 1, 0., list(self._S_COST))
            for k, v in (("depth", 8), ("cands", 4), ("rolls", 1)):
                s[k][v] = s[k].get(v, 0) + 1
        return {}

    def agent(self, obs, cfg):
        t = obs["step"]
        if self.fixture.break_before == t:
            self.fixture.break_before = None
            raise KeyboardInterrupt("simulated worker interruption")
        self.fixture.new_calls.append(t)
        self._ST["rng"].uniform()
        return self._search(t)


class BlendResumeTests(unittest.TestCase):
    def setUp(self):
        self.manage = ExitStack()
        self.addCleanup(self.manage.close)
        self.tmp = Path(self.manage.enter_context(TemporaryDirectory()))
        self.d = load_driver()
        self.d.OUT = self.tmp / "out"
        self.d.PARENT = self.d.CANDIDATE = self.tmp / "mock_actor.py"
        self.d.PARENT.write_text("# never imported; both mock policies are provided by the fixture\n")
        self.seed, self.seat = 620106811, 0  # already present in the archived fixture
        archive = BASE / f"tree_exam_soft_it50_v5/raw_{self.seed}_{self.seat}.json.gz"
        old = self.d.read(archive)
        self.configuration = old["configuration"]
        sim = self.d.arena.exam.FastCarbon(self.d.arena.exam.CFG).reset(seed=self.seed)
        board = self.d.arena.terrain_sha256(sim.carbon)
        allowed = os.sched_getaffinity(0)
        self.addCleanup(os.sched_setaffinity, 0, allowed)
        cpu = min(allowed)
        identity = dict(parent=self.d.sha(self.d.PARENT), candidate=self.d.sha(self.d.CANDIDATE),
                        original_sha256=self.d.sha(self.d.arena.prior.SOURCE), protocol_sha256="mock",
                        boards={str(self.seed): board})
        self.job = ("candidate", self.seed, self.seat, cpu, board, identity)
        self.dest = self.d.OUT / "candidate"
        self.new_calls = []
        self.break_before = None
        self.loads = 0

        def load(path, name):
            if name == "master_original":
                self.loads += 1
                return ToyExaminer(self)
            if name == "candidate_screen":
                return SimpleNamespace(agent=lambda obs, cfg: {})
            raise AssertionError(f"unexpected loader {name}")
        self.manage.enter_context(patch.object(self.d.arena.prior.shared, "module", load))
        self.manage.enter_context(patch.object(self.d.arena.exam, "_extinct",
                                             lambda state, seat: seat == 1 and state["step"] >= 4))
        self.manage.enter_context(patch.object(self.d.official, "check", lambda path, row: dict(
            map_sha256=board, official_reward_by_seat=row["official_score_by_seat"])))

    def file(self, stem, suffix=".json.gz"):
        return self.dest / f"{stem}_{self.seed}_{self.seat}{suffix}"

    def test_interruption_resumes_exact_prefix_and_completed_run_never_calls_examiner(self):
        self.break_before = 2
        with self.assertRaises(KeyboardInterrupt):
            clock.run_one(self.d, self.job)
        self.assertEqual(self.new_calls, [0, 1])
        checkpoint = self.d.read(self.file("progress"))
        self.assertEqual(checkpoint["step"], 2)
        self.assertIn("_clock", checkpoint)
        self.assertTrue(self.file("wall_progress").is_file())
        original_prefix = checkpoint["actions"][:]
        result = clock.run_one(self.d, self.job)
        self.assertEqual(self.new_calls, [0, 1, 2, 3])
        wall = self.d.read(self.file("wall"))
        self.assertEqual([r["step"] for r in wall["calls"]], [0, 1, 2, 3])
        self.assertEqual([r["kind"] for r in wall["calls"]],
                         ["searched", "single", "depth_zero_return_base", "searched"])
        self.assertEqual(self.d.read(self.file("raw"))["actions"][:2], original_prefix)
        self.assertEqual(wall["raw_sha256"], result["game"]["raw_sha256"])
        self.assertEqual(wall["raw_sha256"], self.d.sha(self.file("raw")))
        loads = self.loads
        self.assertEqual(clock.run_one(self.d, self.job), result)
        self.assertEqual(self.loads, loads)
        self.assertEqual(self.new_calls, [0, 1, 2, 3])

    def test_legacy_checkpoint_cannot_be_backfilled(self):
        self.break_before = 2
        with self.assertRaises(KeyboardInterrupt):
            clock.run_one(self.d, self.job)
        old = self.d.read(self.file("progress"))
        del old["_clock"]
        self.d.save(self.file("progress"), old)
        self.file("wall_progress").unlink(missing_ok=True)
        with self.assertRaises(ValueError):
            clock.run_one(self.d, self.job)
        self.assertEqual(self.new_calls, [0, 1])
        self.assertEqual(self.d.read(self.file("progress")), old)

    def test_authoritative_checkpoint_survives_interruption_before_mirror_rename(self):
        replace = os.replace
        def cut_between_checkpoint_and_mirror(source, target):
            if target == self.file("wall_progress") and self.d.read(self.file("progress"))["step"] == 2:
                raise KeyboardInterrupt("checkpoint committed, mirror still one step behind")
            replace(source, target)
        with patch.object(clock.os, "replace", cut_between_checkpoint_and_mirror):
            with self.assertRaises(KeyboardInterrupt):
                clock.run_one(self.d, self.job)
        self.assertEqual(self.d.read(self.file("progress"))["step"], 2)
        self.assertEqual(self.d.read(self.file("wall_progress"))["step"], 1)
        clock.run_one(self.d, self.job)
        self.assertEqual(self.new_calls, [0, 1, 2, 3])
        self.assertEqual(clock.verify_completed(self.d, self.job)["step"], 4)

    def test_full_receipt_and_last_mirror_recover_without_a_new_examiner(self):
        agreed = self.d.agreed
        def cut_final_publish(path, data):
            if path == self.file("wall"):
                raise KeyboardInterrupt("original raw and receipt published; before final clock")
            agreed(path, data)
        with patch.object(self.d, "agreed", cut_final_publish):
            with self.assertRaises(KeyboardInterrupt):
                clock.run_one(self.d, self.job)
        self.assertFalse(self.file("progress").exists())
        self.assertTrue(self.file("receipt", ".json").is_file())
        self.assertTrue(self.file("wall_progress").is_file())
        self.assertFalse(self.file("wall").exists())
        old_loads = self.loads
        clock.run_one(self.d, self.job)
        self.assertEqual(self.loads, old_loads)
        self.assertEqual(self.new_calls, [0, 1, 2, 3])
        self.assertEqual(clock.verify_completed(self.d, self.job)["step"], 4)

    def test_end_state_raw_without_row_recovers_the_committed_clock(self):
        save = self.d.save
        def cut_before_row(path, data):
            if path == self.file("row", ".json"):
                raise KeyboardInterrupt("terminal raw exists; row has not been published")
            save(path, data)
        with patch.object(self.d, "save", cut_before_row):
            with self.assertRaises(KeyboardInterrupt):
                clock.run_one(self.d, self.job)
        self.assertTrue(self.file("raw").exists())
        self.assertFalse(self.file("row", ".json").exists())
        old_loads = self.loads
        signed = clock.run_one(self.d, self.job)
        self.assertTrue(signed["game"]["recovered_from_full_raw"])
        wall = clock.verify_completed(self.d, self.job)
        self.assertEqual(wall["native_stats"]["time"], [0., 0.])
        self.assertEqual(self.new_calls, [0, 1, 2, 3])
        self.assertEqual(self.loads, old_loads)

    def test_mirror_without_a_committed_game_never_restarts_the_examiner(self):
        self.break_before = 2
        with self.assertRaises(KeyboardInterrupt):
            clock.run_one(self.d, self.job)
        self.file("progress").unlink()
        old_loads = self.loads
        with self.assertRaises(ValueError):
            clock.run_one(self.d, self.job)
        self.assertEqual(self.loads, old_loads)
        self.assertEqual(self.new_calls, [0, 1])
        self.assertTrue(self.file("wall_progress").exists())

    def test_valid_but_wrong_raw_return_type_is_rejected(self):
        save = self.d.save
        def corrupt_when_saving_row(path, data):
            if path == self.file("row", ".json"):
                raw = self.d.read(self.file("raw"))
                self.assertEqual(raw["search_calls"][0], [0, "searched"])
                raw["search_calls"][0][1] = "single"
                save(self.file("raw"), raw)
                data["raw_sha256"] = self.d.sha(self.file("raw"))
            save(path, data)

        with patch.object(self.d, "save", corrupt_when_saving_row):
            with self.assertRaises(ValueError):
                clock.run_one(self.d, self.job)
        self.assertEqual(self.new_calls, [0, 1, 2, 3])
        self.assertTrue(self.file("raw").exists())
        self.assertFalse(self.file("wall").exists())


if __name__ == "__main__":
    unittest.main()
