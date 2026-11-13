"""Teacher identity / pilot isolation regression, no neural inference.

Runs the actual gate entry point with deterministic continuation workers.
This checks certification plumbing, not teacher strength.
"""
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import stage1_teacher_gate as gate


class TeacherIdentityTests(unittest.TestCase):
    def test_consumer_checks_version_and_belief(self):
        current = f"v{gate.T.TEACHER_VERSION}"
        self.assertNotEqual(current, "v2")
        good = {"teacher_version": current, "belief": "sample"}
        self.assertIs(gate._expect_teacher(good, "sample"), good)
        for bad in ({"teacher_version": "v2", "belief": "sample"},
                    {"teacher_version": current, "belief": "oracle"}):
            with self.assertRaises(RuntimeError):
                gate._expect_teacher(bad, "sample")

    @staticmethod
    def workers(fn, tasks, _workers):
        assert fn is gate._du_scene
        return [dict(scene, du=0.03 + 0.001 * int(scene["game"][1:]),
                     u_guided=0.55, u_greedy=0.50, teacher_calls=1,
                     teacher_ms=1.0) for scene, _cfg in tasks]

    def run_pilot(self, root, belief, extra=(), n_games=12):
        replays = root / "replays"
        replays.mkdir(exist_ok=True)
        games = [f"p{i}" for i in range(n_games)]
        for game in games:
            with gzip.open(replays / f"{game}.json.gz", "wt") as fh:
                json.dump({"steps": [[]] * 300}, fh)
        (root / "weights.npz").write_bytes(b"hash-only fixture; workers do not load it")
        (root / "split.json").write_text(json.dumps({"du": games}))
        (root / "train.txt").write_text("training_only_game\n")
        out = root / f"{belief}.json"
        argv = ["gate", "--weights", str(root / "weights.npz"),
                "--replays", str(replays), "--split-manifest", str(root / "split.json"),
                "--ds-games-file", str(root / "train.txt"), "--train-data-version", "fixture-v1",
                "--mode", "du", "--pilot", "--belief", belief, "--out", str(out), *extra]
        with patch.object(sys, "argv", argv), patch.object(gate, "run_tasks", self.workers), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(), 0)
        return json.loads(out.read_text()), Path(str(out) + ".pilot.json")

    def test_new_pilot_roundtrip_and_old_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, path = self.run_pilot(root, "sample")
            header = result["prereg"]
            artifact = json.loads(path.read_text())
            self.assertEqual(artifact["grade"], "formal")
            self.assertEqual(artifact["teacher_version"], f"v{gate.T.TEACHER_VERSION}")
            ok, locked, _ = gate._verify_pilot_artifact(path, header, {"stage_b_only"},
                                                       {"training_only_game"})
            self.assertTrue(ok)
            self.assertGreater(locked, 0)
            old = dict(artifact, teacher_version="v2")
            body = {k: v for k, v in old.items() if k != "artifact_id"}
            old["artifact_id"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            path.write_text(json.dumps(old))
            ok, _, reason = gate._verify_pilot_artifact(path, header, {"stage_b_only"})
            self.assertFalse(ok)
            self.assertIn("teacher_version", reason)

    def test_oracle_and_point_estimate_never_formal(self):
        with tempfile.TemporaryDirectory() as tmp:
            for belief in ("oracle", "mean"):
                result, path = self.run_pilot(Path(tmp), belief)
                self.assertFalse(result["prereg"]["formal"])
                self.assertEqual(json.loads(path.read_text())["grade"], "diagnostic")
                self.assertNotEqual(result["du_gate"]["verdict"], "PASS")

    def test_endgame_window_is_enforced_and_bound_to_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, path = self.run_pilot(Path(tmp), "sample", ["--step-min", "200", "--step-max", "299"], n_games=32)
            self.assertTrue(result['rows'])
            self.assertTrue(all(200 <= r['step'] < 299 for r in result['rows']))
            self.assertEqual(result['prereg']['scene_window'], [200, 299])
            artifact = json.loads(path.read_text())
            self.assertEqual(artifact['scene_window'], [200, 299])
            header = dict(result['prereg'], scene_window=[210, 299])
            ok, _, why = gate._verify_pilot_artifact(path, header, {'other_game'})
            self.assertFalse(ok)
            self.assertIn('scene_window', why)


if __name__ == "__main__":
    unittest.main(verbosity=2)
