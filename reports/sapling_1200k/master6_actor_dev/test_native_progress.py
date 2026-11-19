"""Exercise the native owner supervisor with temporary scripts and Linux locks."""
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import native_progress as progress
from native_progress import Job, ensure_owner, lock_held, matching_pids


DUMMY = """\
import fcntl
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
assert sys.argv[1:] == ["run"]
with (root / "entries").open("a") as entries:
    entries.write(f"{os.getpid()} {os.getsid(0)}\\n")
# Leave time for another supervisor to attempt a launch before the native lock.
time.sleep(0.2)
deadline = time.monotonic() + 10
while (root / "pause").exists() and time.monotonic() < deadline:
    time.sleep(0.02)
with (root / "native.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "ready").touch()
    deadline = time.monotonic() + 10
    while not (root / "stop").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
"""


def resume_at_barrier(job, cache, barrier, replies, release):
    """Keep the launching parent alive to reap its own detached test child."""
    barrier.wait(timeout=10)
    try:
        outcome = ensure_owner(job, cache, resume=True, startup_timeout=3)
        replies.put(outcome["state"])
    finally:
        release.wait(timeout=10)
        entries = job.script.parent / "entries"
        if entries.exists():
            for line in entries.read_text().splitlines():
                try:
                    os.waitpid(int(line.split()[0]), 0)
                except ChildProcessError:
                    pass


class NativeOwnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.script = self.root / "dummy.py"
        self.script.write_text(DUMMY)
        self.lock = self.root / "native.lock"
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.job = Job("fixture", self.script, "run", self.lock)

    def wait_for_file(self, name):
        deadline = time.monotonic() + 4
        while not (self.root / name).exists():
            self.assertLess(time.monotonic(), deadline, f"dummy did not create {name}")
            time.sleep(0.02)

    def test_idle_status_has_no_filesystem_side_effects(self):
        for cache in (self.cache, self.root / "missing-cache"):
            with self.subTest(cache=cache):
                before = set(self.root.rglob("*"))
                self.assertFalse(lock_held(self.lock))
                status = ensure_owner(self.job, cache)
                self.assertEqual(status["state"], "idle")
                self.assertEqual(set(self.root.rglob("*")), before)

    def test_resume_respects_an_external_native_lock_without_a_matching_pid(self):
        with self.lock.open("w") as owner:
            owner.write("external owner\n")
            owner.flush()
            fcntl.flock(owner, fcntl.LOCK_EX)
            self.assertTrue(lock_held(self.lock))
            self.assertEqual(list(matching_pids(self.job)), [])
            self.assertEqual(ensure_owner(self.job, self.cache, resume=True)["state"], "active")
            self.assertFalse((self.root / "entries").exists())
        self.assertFalse(lock_held(self.lock))
        self.assertEqual(self.lock.read_text(), "external owner\n")

    def test_match_real_relative_script_in_its_working_directory(self):
        pause = self.root / "pause"
        pause.touch()
        with subprocess.Popen([sys.executable, self.script.name, "run"], cwd=self.root) as child:
            try:
                self.wait_for_file("entries")
                self.assertFalse(lock_held(self.lock))
                self.assertEqual(set(matching_pids(self.job)), {child.pid})
                state = ensure_owner(self.job, self.cache, resume=True, startup_timeout=1)["state"]
                self.assertIn(state, {"active", "starting"})
            finally:
                (self.root / "stop").touch()
                pause.unlink()
                child.wait(timeout=4)
            self.assertEqual(child.returncode, 0)
            self.assertEqual(len((self.root / "entries").read_text().splitlines()), 1)

    def fake_pid(self, proc, number, cwd, args, state="S", start_seconds=0):
        entry = proc / str(number)
        entry.mkdir()
        (entry / "cwd").symlink_to(cwd)
        (entry / "exe").symlink_to(sys.executable)
        argv = [sys.executable, *map(str, args)]
        (entry / "cmdline").write_bytes(os.fsencode("\0".join(argv) + "\0"))
        fields = [state, "1"] + ["0"] * 48
        fields[19] = str(round(start_seconds * os.sysconf("SC_CLK_TCK")))
        (entry / "stat").write_text(f"{number} (python) " + " ".join(fields) + "\n")
        (entry / "status").write_text(f"Name:\tpython\nState:\t{state}\nPPid:\t1\n")

    def test_match_exact_script_and_action_and_exclude_zombies(self):
        proc = self.root / "proc"
        proc.mkdir()
        elsewhere = self.root / "other"
        elsewhere.mkdir()
        other_script = elsewhere / self.script.name
        other_script.touch()

        self.fake_pid(proc, 11, elsewhere, [self.script, "run"])
        self.fake_pid(proc, 12, self.root, [self.script.name, "run"])
        self.fake_pid(proc, 13, elsewhere, [self.script.name, "run"])
        self.fake_pid(proc, 14, self.root, [self.script, "status"])
        self.fake_pid(proc, 15, self.root, [self.script, "run"], state="Z")
        self.fake_pid(proc, 16, elsewhere, [other_script, self.script, "run"])
        self.assertEqual(set(matching_pids(self.job, proc=proc)), {11, 12})

    def test_pid_without_native_lock_only_reserves_the_startup_interval(self):
        proc = self.root / "proc"
        proc.mkdir()
        self.fake_pid(proc, 21, self.root, [self.script, "run"], start_seconds=100)
        for uptime, expected in ((109.98, "starting"), (110.02, "idle")):
            with self.subTest(uptime=uptime):
                (proc / "uptime").write_text(f"{uptime} 0\n")
                status = ensure_owner(self.job, self.cache, proc=proc, startup_timeout=10)
                self.assertEqual(status["matching_live_pids"], [21])
                self.assertFalse(status["native_lock_held"])
                self.assertEqual(status["state"], expected)

    def test_surviving_child_drains_before_rechecking_completion_and_native_owner(self):
        job = Job("fixture", self.script, "run", self.lock, ("case_*/child.lock",))
        child_lock = self.root / "case_71/child.lock"
        child_lock.parent.mkdir()
        for phase, reclaimed in (("ready", False), ("ready", True), ("complete", False)):
            with (self.subTest(phase=phase, reclaimed=reclaimed),
                  child_lock.open("a") as child, self.lock.open("a") as supervisor,
                  patch.object(progress, "BASE", self.root), patch.object(os, "execv") as execute):
                fcntl.flock(child, fcntl.LOCK_EX)
                with patch.object(subprocess, "Popen") as launch:
                    launch.return_value.poll.return_value = 0
                    launch.return_value.returncode = 0
                    outcome = ensure_owner(job, self.cache, resume=True)
                    self.assertEqual(outcome["surviving_worker_locks"], ["case_71/child.lock"])
                    self.assertEqual(launch.call_args.args[0][-2:], ["_drain", "fixture"])

                def exit_child(_seconds):
                    fcntl.flock(child, fcntl.LOCK_UN)
                    if reclaimed:
                        fcntl.flock(supervisor, fcntl.LOCK_EX)

                def check_after_drain(_name):
                    self.assertFalse(lock_held(child_lock))
                    return phase, {}

                with (patch.object(time, "sleep", side_effect=exit_child) as slept,
                      patch.object(progress, "summary", side_effect=check_after_drain) as checked):
                    progress.drain_orphans(job)
                    slept.assert_called_once()
                    checked.assert_called_once_with("fixture")
                if phase == "ready" and not reclaimed:
                    execute.assert_called_once_with(sys.executable, [sys.executable, "-u", str(self.script), "run"])
                else:
                    execute.assert_not_called()

    def test_concurrent_resume_launches_one_detached_native_owner(self):
        ctx = multiprocessing.get_context("fork")
        barrier = ctx.Barrier(2)
        replies = ctx.Queue()
        release = ctx.Event()
        owners = [
            ctx.Process(target=resume_at_barrier, args=(self.job, self.cache, barrier, replies, release))
            for _ in range(2)
        ]
        for owner in owners:
            owner.start()
        try:
            states = [replies.get(timeout=8) for _ in owners]
            self.assertTrue(all(state in {"active", "started", "starting"} for state in states), states)
            self.wait_for_file("ready")
            self.assertTrue(lock_held(self.lock))
            entries = (self.root / "entries").read_text().splitlines()
            self.assertEqual(len(entries), 1, entries)
            pid, session = map(int, entries[0].split())
            self.assertEqual(pid, session)
            self.assertEqual(set(matching_pids(self.job)), {pid})
        finally:
            (self.root / "stop").touch()
            release.set()
            for owner in owners:
                owner.join(timeout=12)
            replies.close()
            replies.join_thread()
        self.assertEqual([owner.exitcode for owner in owners], [0, 0])
        self.assertFalse(lock_held(self.lock))
        self.assertEqual(list(matching_pids(self.job)), [])
        self.assertEqual(len((self.root / "entries").read_text().splitlines()), 1)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.train = self.base / "training"
        self.sup = self.train / "supplement"
        self.head = self.train / "head"
        self.early = self.base / "early/native"
        self.econ = self.base / "economic"
        self.queue = self.base / "queue"
        self.hire = self.base / "hire"
        changes = patch.multiple(progress, BASE=self.base, TRAIN=self.train, SUP=self.sup,
                                 HEAD=self.head, EARLY=self.early, ECON=self.econ, QUEUE=self.queue, HIRE=self.hire)
        changes.start()
        self.addCleanup(changes.stop)

    def write(self, path, obj):
        data = json.dumps(obj, sort_keys=True).encode()
        if path.suffix == ".gz":
            data = gzip.compress(data, mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def test_final_responses_require_all_four_paired_receipts(self):
        cases = [{"seed": x, "seat": x%2} for x in range(71,75)]
        plan = self.write(self.queue / "native_protocol.json",
                          {"cases": cases, "candidate_sha256": "candidate", "source_examiner":"original"})
        self.assertEqual(progress.summary("endgame_queue")[0], "ready")
        complete = self.write(self.sup / "completion.json", {"phase":"closed"})
        grant = self.write(self.sup / "endgame_queue_grant.json", {"cpu":0, "permitted_native_calls":8,
                           "completion_sha256":complete,"last_move_protocol_sha256":plan})
        self.assertEqual(progress.summary("endgame_queue")[1]["phase"], "last_turns_granted")
        bindings = {}
        for case in cases:
            path = self.queue / "native_{seed}_{seat}.json.gz".format(**case)
            both = {arm:{"end":{"status":["DONE","DONE"]},"search_call_count":1} for arm in ("v1","candidate")}
            bindings[path.name] = self.write(path, {**case, "protocol_sha256":plan, "arms":both})
        proof = {"protocol_sha256":plan,"grant_sha256":grant,"completion_sha256":complete,
                 "runner_sha256":self.write(self.queue / "native_last.py", {}),"candidate_sha256":"candidate",
                 "examiner_sha256":"original","pairs":cases,"native_calls":8,"pair_sha256":bindings}
        self.write(self.queue / "native_result.json", proof)
        self.assertEqual(progress.summary("endgame_queue")[0], "complete")
        self.write(path, {"missing":"one real decision"})
        with self.assertRaises(ValueError):
            progress.summary("endgame_queue")

    def test_late_hires_require_all_paired_action_tapes_and_official_scores(self):
        out = self.hire / "native"
        cases = [{"seed":71+i,"seat":i,"t":211} for i in range(2)]
        plan = self.write(out / "protocol.json", {"cases":cases,"parent_sha256":"parent","candidate_sha256":"candidate"})
        self.assertEqual(progress.summary("late_hiring")[0], "ready")
        self.write(out / "granted.json", {"protocol_sha256":plan,
                   "shared_final_probe_sha256":self.write(self.queue/"native_result.json",{}),
                   "tree_head_acceptance_sha256":self.write(self.head/"independent_acceptance.json",{})})
        rows = []
        for c in cases:
            wd = out / "case_{seed}_{seat}".format(**c)
            arms = {}
            for a in ("parent","alternative"):
                path = wd / f"live_{a}_actions.json.gz"
                sha = self.write(path, [[{},{}],[{},{}]])
                self.write(wd/f"live_{a}.json", {"action_sha256":sha,"action_tape":path.name,
                          "statuses":["DONE","DONE"],"search_calls":1,"score":[20,30]})
                arms[a]={"score_by_physical_seat":[20,30],"verified_native_search_calls":1,
                         "raw_sha256":sha,"statuses":["DONE","DONE"]}
            rows.append({"seed":c["seed"],"seat":c["seat"],"turn":c["t"],
                         "proof_sha256":self.write(wd/"verified.json.gz", {"official":arms})})
        self.write(out/"results.json", {"protocol_sha256":plan,"candidate_sha256":"candidate","source_parent_sha256":"parent",
                   "all_four_responses":True,"formal_games":0,"native_cases":rows,"decision":"do_not_expand"})
        self.assertEqual(progress.summary("late_hiring")[0], "complete")
        self.write(path, [[{},{}]])
        with self.assertRaises(ValueError):
            progress.summary("late_hiring")

    def acceptance(self, passed):
        doc = {
            "passed": passed,
            "reason": "eligible" if passed else "insufficient_realized_support",
            "completion_sha256": self.write(self.sup / "completion.json", {"phase": "all_registered_blocks_exhausted"}),
            "status_sha256": self.write(self.head / "status.json", {"eligible_for_development": passed}),
            "sample_tape_sha256": self.write(self.head / "samples.json.gz", []),
        }
        self.write(self.head / "independent_acceptance.json", doc)

    def test_independent_acceptance_controls_development_but_does_not_seal_eight_games(self):
        self.acceptance(False)
        state, info = progress.summary("response_head")
        self.assertEqual((state, info["development_games"]), ("complete", 0))
        self.acceptance(True)
        self.assertEqual(progress.summary("response_head")[0], "ready")
        self.write(self.head / "samples.json.gz", [{"modified": True}])
        with self.assertRaises(ValueError):
            progress.summary("response_head")

    def comparison(self):
        """Small, on-disk schema fixtures for the registered two-map comparison."""
        self.acceptance(True)
        run = self.head / "development_runs"
        maps = [71, 72]
        seats = [(seed, seat) for seed in maps for seat in (0, 1)]
        release = {
            "maps": maps, "boards": {str(s): f"board-{s}" for s in maps},
            "development_protocol": self.write(self.head / "protocol.json", {"maps": maps}),
            "source_protocol": self.write(self.train / "protocol.json", {"name": "training fixture"}),
            "master3s2_sha256": "examiner", "arena_sources": {"data/master3s2_source.py": "examiner"},
            "official_engine": {"engine.py": "engine"}, "cpus": [0, 1],
            "packages": {a: {"pack_sha256": f"{a}-pack", "weight_sha256": f"{a}-weights"}
                         for a in ("parent", "candidate")},
        }
        release_hash = self.write(run / "release.json", release)
        pairs = {key: {"seed": key[0], "seat": key[1]} for key in seats}
        for arm, package in release["packages"].items():
            native, official = [], []
            for seed, seat in seats:
                raw = f"raw_{seed}_{seat}.json.gz"
                raw_hash = self.write(run / arm / raw, [arm, seed, seat])
                native.append({"seed": seed, "candidate_seat": seat, "raw_file": raw, "raw_sha256": raw_hash,
                               "official_score_by_seat": [10, 10], "initial_terrain_u8_sha256": f"board-{seed}",
                               "actual_transitions": 299, "result": "draw", "margin": 0, "cpu_affinity": [seat]})
                replay = {"seed": seed, "seat": seat, "transcript_sha256": raw_hash, "result": "draw",
                          "map_sha256": f"board-{seed}", "steps": 299, "statuses": ["DONE", "DONE"],
                          "official_reward_by_seat": [10, 10]}
                official.append(replay)
                pairs[seed, seat][arm] = dict(replay, margin=0, U=0.5, verified_edits=[])
            header = dict(package, seeds=maps, cpus=release["cpus"], source_sha256=release["arena_sources"])
            self.write(run / arm / "results.json", {"header": header, "status": "complete", "games": native})
            self.write(run / arm / "official_replay.json", {"engine_source_sha256": release["official_engine"],
                       "runs": {arm: {"pack_sha256": package["pack_sha256"], "games": official}}})
        self.write(run / "comparison.json", {"pairs": list(pairs.values()), "release_sha256": release_hash,
                                            "games": 8, "formal": False, "verdict": "retain_v1"})
        return run

    def test_eight_official_receipts_and_each_candidate_binding(self):
        for change in ("missing_seat", "status", "raw", "pack", "weights", "map", "engine"):
            with self.subTest(change=change):
                run = self.comparison()
                state, info = progress.summary("response_head")
                self.assertEqual((state, info["development_games"]), ("complete", 8))
                native_path = run / "candidate/results.json"
                replay_path = run / "candidate/official_replay.json"
                pairs_path = run / "comparison.json"
                native, official, comparison = map(progress.read, (native_path, replay_path, pairs_path))
                replay = official["runs"]["candidate"]
                if change == "missing_seat":
                    native["games"].pop()
                elif change == "status":
                    replay["games"][0]["statuses"] = comparison["pairs"][0]["candidate"]["statuses"] = ["DONE", "ERROR"]
                elif change == "raw":
                    self.write(run / "candidate" / native["games"][0]["raw_file"], ["modified tape"])
                elif change == "pack":
                    native["header"]["pack_sha256"] = replay["pack_sha256"] = "parent-pack"
                elif change == "weights":
                    native["header"]["weight_sha256"] = "parent-weights"
                elif change == "map":
                    replay["games"][0]["map_sha256"] = comparison["pairs"][0]["candidate"]["map_sha256"] = "other-board"
                    native["games"][0]["initial_terrain_u8_sha256"] = "other-board"
                elif change == "engine":
                    official["engine_source_sha256"] = {"engine.py": "other-engine"}
                for path, doc in ((native_path, native), (replay_path, official), (pairs_path, comparison)):
                    self.write(path, doc)
                with self.assertRaises(ValueError):
                    progress.summary("response_head")

    def test_training_completion_requires_signed_labels_and_registration(self):
        labels = self.train / "intake_01/training_labels.json"
        registered = self.sup / "registration.json"
        receipt = {"phase": "all_registered_blocks_exhausted", "next": None,
                   "registration_sha256": self.write(registered, {"batches": ["intake_01"]}),
                   "counts": {"actual_paid_pairs": 1, "source_intakes": [
                       {"dir": "intake_01", "labels_sha256": self.write(labels, [{"paid": True}])}]}}
        self.write(self.sup / "completion.json", receipt)
        self.assertEqual(progress.summary("training")[0], "complete")
        self.write(labels, [])
        with self.assertRaises(ValueError):
            progress.summary("training")
        self.write(labels, [{"paid": True}])
        self.write(registered, {})
        with self.assertRaises(ValueError):
            progress.summary("training")

    def test_economic_attribution_requires_twelve_game_release_and_four_comparisons(self):
        slot = self.sup / "economy_slot"
        self.assertEqual(progress.summary("economy_report")[0], "waiting")
        games = [{"seed": seed, "seat": seat} for seed in (71, 72) for seat in (0, 1)]
        comparison_hash = self.write(self.econ / "comparison.json", {"paired_map_seats": games})
        grant_hash = self.write(slot / "granted.json", {"name": "grant fixture"})
        arms = {}
        for arm in ("parent", "candidate", "control"):
            arms[arm] = {
                "results_sha256": self.write(self.econ / arm / "results.json", {"status": "complete", "games": games}),
                "official_sha256": self.write(self.econ / arm / "official_replay.json", {"runs": {arm: {"games": games}}}),
            }
        receipt = {"grant_sha256": grant_hash, "comparison_sha256": comparison_hash, "signed_games": 11, "arms": arms}
        self.write(slot / "released.json", receipt)
        with self.assertRaises(ValueError):
            progress.summary("economy_report")
        receipt["signed_games"] = 12
        release_hash = self.write(slot / "released.json", receipt)
        self.assertEqual(progress.summary("economy")[0], "complete")
        self.assertEqual(progress.summary("economy_report")[0], "ready")
        evidence = {"native_release_sha256": release_hash, "comparison_sha256": comparison_hash, "games": games}
        self.write(self.econ / "actual_attribution.json.gz", evidence)
        (self.econ / "RESULTS.md").write_text("Fixture attribution\n")
        self.assertEqual(progress.summary("economy_report")[0], "complete")
        evidence["native_release_sha256"] = "other-release"
        self.write(self.econ / "actual_attribution.json.gz", evidence)
        with self.assertRaises(ValueError):
            progress.summary("economy_report")
        self.write(self.econ / "candidate/official_replay.json", {})
        with self.assertRaises(ValueError):
            progress.summary("economy")

    def test_early_lane_skip_requires_authoritative_zero_game_evidence(self):
        slot = self.sup / "early_lane_slot"
        skip = self.early.parent / "early_lane_skip.json"
        seal = {"protocol_sha256": self.write(self.early / "protocol.json", {}),
                "skip_sha256": self.write(skip, {"reason": "fixture"}), "native_arms": 0}
        self.write(slot / "skipped.json", seal)
        self.assertEqual(progress.summary("early_lane")[0], "complete")
        self.write(skip, {"reason": "changed"})
        with self.assertRaises(ValueError):
            progress.summary("early_lane")
        self.write(skip, {"reason": "fixture"})
        self.write(slot / "released.json", {})
        with self.assertRaises(ValueError):
            progress.summary("early_lane")

    def test_early_lane_release_requires_four_distinct_verified_cases(self):
        slot = self.sup / "early_lane_slot"
        cases = []
        for seed in (71, 72):
            for seat in (0, 1):
                verification = self.early / f"case_{seed}_{seat}" / "verified.json.gz"
                proof = self.write(verification, {"seed": seed, "seat": seat})
                cases.append({"seed": seed, "seat": seat, "proof_sha256": proof})
        seal = {"grant_sha256": self.write(slot / "granted.json", {"name": "grant fixture"}),
                "result_sha256": self.write(self.early / "result.json", {"status": "complete", "completed_pairs": 4,
                                                                         "completed_arms": 8}),
                "cases": cases, "original_responses": 8}
        self.write(slot / "released.json", seal)
        self.assertEqual(progress.summary("early_lane")[0], "complete")
        seal["cases"] = [cases[0], *cases[:-1]]
        self.write(slot / "released.json", seal)
        with self.assertRaises(ValueError):
            progress.summary("early_lane")
        seal["cases"] = cases
        self.write(slot / "released.json", seal)
        self.write(self.early / "case_71_0/verified.json.gz", {"changed": True})
        with self.assertRaises(ValueError):
            progress.summary("early_lane")


if __name__ == "__main__":
    unittest.main()
