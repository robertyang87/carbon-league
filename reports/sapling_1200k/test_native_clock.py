"""Formal clock coverage without calling a real examiner or drawing any maps."""
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import native_clock as clock
import tree_audited_game as arena
import d1_sapling_arena as d1
from compound_formal import D1


def host(t):
    return dict(hostname="test", cpu_model="mock", kernel="test", pid=100,
                affinity=[3], thread_affinity=[dict(cpus=[3], threads=2)],
                torch_threads=2, torch_interop_threads=1,
                thread_env=dict.fromkeys(clock.THREAD_ENV), pools=[], pool_error=None,
                cgroup="/", cpu_max="200000 100000",
                cpu_stat=dict(usage_usec=t, nr_throttled=0, throttled_usec=0),
                perf_counter=t, process_cpu_seconds=t / 2)


def fixture(run, pack_sha):
    stats = dict(searched=0, single=0, aborted=0, changed=0, time=[], depth={}, rolls={}, cands={})
    master = SimpleNamespace(_S_STATS=stats, _S_REC=None, _S_ABORT=.58, _S_D2=8)
    kinds = ["searched", "single", "depth_zero_return_base", "searched"]

    def original(obs, cfg):
        t = obs["step"]
        if t == 1:
            stats["single"] += 1  # deliberately retain the last searched call's stats
        else:
            d = (2, None, 0, 8)[t]
            stats["last"] = (4, d, 1, 0., (0.,) * 4)
            stats["aborted"] += int(t in (0, 2))
            if d:
                stats["searched"] += 1
                stats["time"].append(0.)
                stats["changed"] += int(t == 3)
                for k, v in (("cands", 4), ("depth", d), ("rolls", 1)):
                    stats[k][v] = stats[k].get(v, 0) + 1
        return {"worker-1": "LEFT"}
    master.agent = original
    candidate = SimpleNamespace(_AG=SimpleNamespace(stats=dict(depth_sum=0, changed=0)), _LAST={})

    def candidate_agent(obs, cfg):
        candidate._LAST = dict(changed=False, base={}, legal_branches=[],
                               selected_unforced_differs_from_expert=[])
        return {}
    candidate.agent = candidate_agent
    raw = dict(seed=123, candidate_seat=1, pack_sha256=pack_sha, weight_identity="weights",
               examiner_md5=arena.prior.EXPECTED_MD5, search_calls=list(map(list, enumerate(kinds))),
               configuration=dict(_S_K=32, _S_D2=8, _S_M=1, _S_MMAX=1, _S_ABORT=.58))

    def loader(path, name):
        return master if name == "master_original" else candidate

    def base(job):
        student = arena.prior.shared.module(job[0], "candidate_screen")
        examiner = arena.prior.shared.module(arena.prior.SOURCE, "master_original")
        for t in range(4):
            assert student.agent({"step": t}, {}) == {}
            assert examiner.agent({"step": t}, {}) == {"worker-1": "LEFT"}
        return dict(seed=123, candidate_seat=1, cpu_affinity=[3], actual_transitions=4,
                    raw_sha256="raw", call_seconds_by_seat=[{"sum": 1.}, {"sum": 1.}],
                    search=stats)

    job = ("mock-agent", "weights", 123, 1, 3, str(run), pack_sha)
    with patch.object(arena.prior.shared, "module", loader), patch.object(arena, "play", base), \
            patch.object(d1, "_BASE_PLAY", base), patch.object(clock, "host_snapshot", side_effect=[host(10), host(11)]):
        row = clock.clocked_play(job)
        assert arena.prior.shared.module is loader
    return row, raw


class ClockTests(unittest.TestCase):
    def test_assignment_remains_balanced_and_stable_when_only_one_seat_remains(self):
        plan = dict(protocol_sha256={clock.SOURCE_KEY: "bound"}, maps=[{"seed": k + 101} for k in range(100)])
        cpus = clock.assignments(plan, [4, 7])
        for seat in (0, 1):
            used = [cpus[(m["seed"], seat)] for m in plan["maps"]]
            self.assertEqual(used.count(4), 50)
            self.assertEqual(used.count(7), 50)
        self.assertEqual(clock.assignments(plan, [4, 7])[(102, 0)], 7)
        self.assertEqual(clock.assignments(dict(protocol_sha256={}, maps=plan["maps"]), [4, 7]), {})
        self.assertTrue(clock.required(plan))  # removing the descriptive schema label cannot downgrade
        with self.assertRaises(ValueError):
            clock.assignments(plan, [7, 7])

    def test_normal_and_compound_entries_keep_actions_and_sidecars(self):
        for pack in ("single-or-hybrid", D1):
            with self.subTest(pack=pack), TemporaryDirectory() as run:
                row, raw = fixture(run, pack)
                clock.verify(run, row, raw)
                self.assertEqual("candidate_search_audit" in row, pack == D1)
                receipt = json.loads(gzip.decompress((Path(run) / row["native_examiner_clock"]["file"]).read_bytes()))
                self.assertEqual([(c["kind"], c["candidates"], c["max_depth"]) for c in receipt["calls"]],
                                 [("searched", 4, 2), ("single", 1, 0), ("depth_zero_return_base", 4, 0), ("searched", 4, 8)])
                self.assertIsNone(receipt["calls"][1]["native_seconds"])
                self.assertTrue(receipt["calls"][3]["changed"])
                self.assertEqual(receipt["summary"]["non_single_below_reachable_depth"], 2)
                self.assertEqual(receipt["summary"]["max_completed_depth"], {"0": 2, "2": 1, "8": 1})

    def test_signed_sidecar_cross_checks_native_aggregates_or_fails(self):
        with TemporaryDirectory() as run:
            row, raw = fixture(run, "single")
            proof = row["native_examiner_clock"]
            path = Path(run) / proof["file"]
            receipt = json.loads(gzip.decompress(path.read_bytes()))
            for mutate in (lambda d: d["calls"][0].update(max_depth=1),
                           lambda d: d["calls"][1].update(native_seconds=.1),
                           lambda d: d["calls"][2].update(native_aborted=False),
                           lambda d: d["calls"][3].update(changed=False),
                           lambda d: d["calls"][3].update(process_cpu_seconds=float("nan")),
                           lambda d: d["calls"].pop(),
                           lambda d: d["errors"].append([2, "unknown return"]),
                           lambda d: d["host_start"].update(affinity=[99]),
                           lambda d: d.update(raw_sha256="another tape")):
                broken = copy.deepcopy(receipt)
                mutate(broken)
                data = gzip.compress(json.dumps(broken).encode(), mtime=0)
                path.write_bytes(data)
                proof["sha256"] = hashlib.sha256(data).hexdigest()  # even a consistent artifact digest is insufficient
                with self.assertRaises(ValueError):
                    clock.verify(run, row, raw)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                clock.verify(run, row, raw)
            del row["native_examiner_clock"]
            with self.assertRaisesRegex(ValueError, "requires"):
                clock.verify(run, row, raw)

    def test_optional_native_pool_inventory_is_explicit_when_unavailable(self):
        master = SimpleNamespace(_torch=SimpleNamespace(get_num_threads=lambda: 2, get_num_interop_threads=lambda: 1))
        with patch.dict(sys.modules, {"threadpoolctl": None}):
            snap = clock.host_snapshot(master)
        self.assertIn("ModuleNotFoundError", snap["pool_error"])
        self.assertEqual(snap["pools"], [])
        self.assertEqual(snap["torch_threads"], 2)
        clock.validate_host(snap, snap["affinity"])

    def test_observer_failure_preserves_actual_action_and_surfaces_collection_error(self):
        master = SimpleNamespace(_S_STATS=dict.fromkeys(clock.COUNTERS, 0), _S_REC=None)
        with patch.object(clock, "host_snapshot", return_value=host(10)):
            rec = clock.Recorder(master)
        action = {"real": "UP"}
        self.assertIs(rec.wrap(lambda obs, cfg: action)({"step": 0}, {}), action)
        self.assertEqual(len(rec.errors), 1)


if __name__ == "__main__":
    unittest.main()
