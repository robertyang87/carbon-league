"""Producer provenance through the real scanner and build gate.

Runs without weights or replay fixtures. Positive cases stop at the worker
boundary (after scan/gate/fingerprint); negative cases must stop before it
and preserve an existing generation pointer. This tests metadata admission,
not simulator or feature correctness.
"""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "producer"))
import cnet2_build as B
import selfplay_gen as P


class ReachedWorkers(Exception):
    pass


class MirrorIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rd = Path(self.tmp.name) / "corpus"
        (self.rd / "replays").mkdir(parents=True)
        self.names = B.F.TEACHERS[:2]
        conf = {"OPEN_T": 1.0, "OPEN_STEPS": 50, "OPEN_SEED": 7}
        ident = {"weights_md5": "a" * 32,
                 "config": conf,
                 "config_hash": hashlib.sha256(json.dumps(conf, sort_keys=True).encode()).hexdigest(),
                 "names": list(self.names),
                 "code_md5": {k: "c" * 32 for k in
                              ("eval_v53.py", "fastsim.py", "cnet2_search.py", "selfplay_gen.py")},
                 "diversity": P.diversity_of(conf)}
        self.entry = {"id": "selfplay", "game": 5, "order": ["b", "a"],
                      "scores": {"a": 11.0, "b": 10.0}, "steps": 299,
                      "file": "replays/selfplay-5.json.gz", "identity": ident,
                      "open_seed": P.game_open_seed(7, 5), "mirror_signature": False,
                      "strategies": {k: {"id": "", "name": name}
                                     for k, name in zip(("a", "b"), self.names)}}

    def write(self, entry=None, replay_edit=None, run_identity=False):
        entry = copy.deepcopy(entry or self.entry)
        scores = [entry["scores"][k] for k in entry["order"]]
        result = {"scores": scores, "a_side": entry["order"].index("a")}
        for k in ("identity", "mirror_signature", "open_seed"):
            if k in entry:
                result[k] = copy.deepcopy(entry[k])
        # Full enough for orphan reconstruction. Worker decoding is out of scope.
        replay = {"id": entry["id"], "steps": [[{"status": "DONE"}, {"status": "DONE"}]],
                  "result": result}
        if "identity" in entry:
            replay["identity"] = copy.deepcopy(entry["identity"])
        if replay_edit:
            replay_edit(replay)
        with gzip.open(self.rd / entry["file"], "wt") as f:
            json.dump(replay, f)
        (self.rd / "index.json").write_text(json.dumps([entry]))
        if run_identity:
            (self.rd / "identity.json").write_text(json.dumps({"identity": entry.get("identity")}))
        return entry

    def scan(self):
        games, missing, _ = B._scan([str(self.rd)])
        self.assertEqual(len(games), 1)
        self.assertEqual(missing, [])
        return games

    def gate(self, accepted, allow=False, match=None):
        out = Path(self.tmp.name) / "out"
        out.mkdir(exist_ok=True)
        pointer = out / "current.json"
        pointer.write_text('{"existing":true}')
        env = {"CNET2_ALLOW_MIRROR": "1" if allow else "0"}
        want = ReachedWorkers if accepted else (SystemExit, ValueError)
        with patch.dict(os.environ, env), patch.object(B, "ProcessPoolExecutor", side_effect=ReachedWorkers):
            with self.assertRaises(want) as caught:
                B.main(str(out), [str(self.rd)])
        if match:
            self.assertIn(match, str(caught.exception))
        self.assertEqual(pointer.read_text(), '{"existing":true}')
        self.assertFalse((out / "meta.json").exists())

    def test_sampled_aliases_swapped_seats_are_admitted(self):
        self.write(run_identity=True)
        games = self.scan()
        self.assertEqual(games[0]["names"], self.names[::-1])
        self.assertEqual(games[0]["producer_provenance"]["identity"]["names"], self.names)
        self.assertEqual(B.mirror_signature(games)["selfplay_games"], 1)
        self.gate(True)

    def test_same_weights_aliases_equal_scores_are_refused(self):
        self.entry["scores"]["a"] = 10.0
        self.entry["mirror_signature"] = True
        self.write()
        self.assertEqual(B.mirror_signature(self.scan())["share"], 1.0)
        self.gate(False, match="same-policy")

    def test_one_retained_teacher_side_cannot_bypass(self):
        self.entry["strategies"]["b"]["name"] = "not-a-listed-teacher"
        self.entry["identity"]["names"][1] = "not-a-listed-teacher"
        self.entry["scores"]["a"] = 10.0
        self.entry["mirror_signature"] = True
        self.write()
        self.assertEqual(len(self.scan()[0]["sides"]), 1)
        self.gate(False)

    def test_same_display_name_different_ladder_strategies_are_not_selfplay(self):
        e = {k: v for k, v in self.entry.items() if k not in ("identity", "open_seed", "mirror_signature")}
        e.update(id="ladder-match", file="replays/ladder-match-5.json.gz")
        e["strategies"] = {k: {"name": self.names[0], "id": k,
                                "weights_md5": h * 32} for k, h in (("a", "a"), ("b", "b"))}
        e["scores"] = {"a": 10.0, "b": 10.0}
        self.write(e)
        self.assertEqual(B.mirror_signature(self.scan())["selfplay_games"], 0)
        self.gate(True)

    def test_legacy_mirror_is_explicitly_unverified_and_refused(self):
        e = {k: v for k, v in self.entry.items() if k not in ("identity", "open_seed", "mirror_signature")}
        e["scores"] = {"a": 10.0, "b": 10.0}
        self.write(e)
        sig = B.mirror_signature(self.scan())
        self.assertEqual((sig["legacy_games"], sig["share"]), (1, 1.0))
        self.gate(False, match="legacy/unverified")
        self.gate(True, allow=True)

    def test_legacy_non_draw_does_not_become_formal(self):
        e = {k: v for k, v in self.entry.items() if k not in ("identity", "open_seed", "mirror_signature")}
        self.write(e)
        self.assertEqual(B.mirror_signature(self.scan())["share"], 0.0)
        self.gate(False, match="legacy/unverified")

    def test_declared_rehearsal_requires_build_override_even_without_draws(self):
        self.entry["identity"]["diversity"]["allow_mirror"] = True
        self.write()
        self.gate(False, match="declared rehearsal")
        self.gate(True, allow=True)

    def test_replay_identity_mismatch_cannot_be_overridden(self):
        self.write(replay_edit=lambda r: r["identity"].update(weights_md5="b" * 32))
        self.gate(False, allow=True, match="identity mismatch")

    def test_corpus_identity_mismatch_cannot_be_overridden(self):
        self.write(run_identity=True)
        d = json.loads((self.rd / "identity.json").read_text())
        d["identity"]["weights_md5"] = "b" * 32
        (self.rd / "identity.json").write_text(json.dumps(d))
        self.gate(False, allow=True, match="corpus/index")

    def test_bad_generation_config_is_rejected(self):
        self.entry["identity"]["config_hash"] = "d" * 64
        self.write()
        self.gate(False, allow=True, match="config_hash mismatch")

    def test_bad_diversity_is_rejected(self):
        self.entry["identity"]["diversity"]["open_t"] = 0
        self.write()
        self.gate(False, allow=True, match="diversity")

    def test_incomplete_code_identity_is_rejected(self):
        self.entry["identity"]["code_md5"] = {"unknown.py": "a" * 32}
        self.write()
        self.gate(False, match="code hashes")

    def test_unsupported_per_seat_weight_schema_is_rejected(self):
        self.entry["identity"]["weights_md5"] = {"a": "a" * 32, "b": "b" * 32}
        self.write()
        self.gate(False, match="single-weights")

    def test_open_seed_and_signature_must_be_bound_to_game(self):
        self.entry["open_seed"] += 1
        self.write()
        self.gate(False, match="opening seed")
        self.entry["open_seed"] -= 1
        self.entry["mirror_signature"] = True
        self.write()
        self.gate(False, allow=True, match="mirror_signature")

    def test_orphan_recovery_preserves_provenance_fields(self):
        self.write()
        adopted = P._entry_from_replay(str(self.rd), "selfplay-5.json.gz", *self.names)
        self.assertEqual(adopted["open_seed"], self.entry["open_seed"])
        self.assertEqual(adopted["mirror_signature"], self.entry["mirror_signature"])
        (self.rd / "index.json").write_text(json.dumps([adopted]))
        self.gate(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
