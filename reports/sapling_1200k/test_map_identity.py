"""Regression tests on synthetic and previously used development maps only."""
import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import map_identity as identity
import tree_win_gate as gate


class TerrainTests(unittest.TestCase):
    def test_hash_representation_and_bad_cells(self):
        self.assertEqual(identity.terrain_sha256([1]*225), identity.terrain_sha256([1.0]*225))
        for cells in ([0]*224, [0]*224+[.4], [0]*224+[256], [0]*224+[float("nan")], [0]*224+[True]):
            with self.subTest(last=cells[-1]), self.assertRaises(ValueError):
                identity.terrain_sha256(cells)

    def test_same_terrain_at_different_seed_retires_draw(self):
        known = identity.terrain_sha256([0]*225)
        other = identity.terrain_sha256([0]*224+[1])
        with patch.object(identity,"local_board",side_effect=lambda seed: {11:known,12:other,13:other}[seed]):
            with self.assertRaisesRegex(identity.OverlappingDraw,"terrain"):
                identity.checked_maps([11],{10},{known})
            with self.assertRaisesRegex(identity.OverlappingDraw,"seed"):
                identity.checked_maps([12],{12},{known})
            with self.assertRaisesRegex(identity.OverlappingDraw,"terrain"):
                identity.checked_maps([12,13],set(),{known})
            self.assertEqual(identity.checked_maps([12],{10},{known}),[{"seed":12,"initial_terrain_u8_sha256":other}])

    def test_both_seats_and_origin_payload(self):
        cells = [0]*225
        fingerprint = identity.terrain_sha256(cells)
        game = {"initial_terrain_u8_sha256":fingerprint}
        raw = dict(game, initial_full_carbon=cells)
        identity.verify_observed_map(fingerprint,game,raw)
        for row, tape in (({},raw), (game,{}), (game,dict(raw,initial_full_carbon=[1]*225))):
            with self.assertRaises(ValueError):
                identity.verify_observed_map(fingerprint,row,tape)

    def test_historical_record_and_terrain_locked_replay(self):
        from tree_audited_game import exam
        folder = Path(__file__).with_name("tree_exam_basecal")
        record = json.loads((folder/"results.json").read_text())["games"][0]
        raw = json.loads(gzip.decompress((folder / record["raw_file"]).read_bytes()))
        game = exam.FastCarbon(exam.CFG).reset(seed=record["seed"])
        fingerprint = identity.terrain_sha256(game.carbon)
        gate.verify_terminal_replay(record,raw)
        with self.assertRaisesRegex(ValueError,"initial terrain"):
            gate.verify_terminal_replay(record,raw,fingerprint)
        # Synthetic instrumentation of a known dev map verifies the new comparison, leaving archival bytes intact.
        witnessed = dict(raw,initial_terrain_u8_sha256=fingerprint,initial_full_carbon=list(game.carbon))
        gate.verify_terminal_replay(dict(record,initial_terrain_u8_sha256=fingerprint),witnessed,fingerprint)

    def test_historical_schema_cannot_start_new_exam(self):
        maps=[{"seed":101,"initial_terrain_u8_sha256":identity.local_board(101)}]
        ledger, _, _ = identity.ladder_boards()
        plan=dict(schema=identity.SCHEMA,weight_sha256="w",pack_sha256="p",examiner_md5=gate.EXAMINER,
                  income_metric="official_margin",income_minimum=500,protocol_sha256=gate.protocol_hashes(),
                  nonce="3"*64,maps=maps,excluded_reserved_seeds=[],run_directory="sandbox",ladder_ledger_sha256=ledger)
        with patch.object(gate,"seed_list",return_value=[101]):
            self.assertEqual(gate.validate(plan,"w","p",gate.EXAMINER,"sandbox"),[101])
            with self.assertRaisesRegex(ValueError,"initial-terrain exposure lock"):
                gate.validate(dict(plan,schema="sapling-90-and-income/3"),"w","p",gate.EXAMINER,"sandbox")
            with self.assertRaisesRegex(ValueError,"allocated terrain"):
                gate.validate(dict(plan,maps=[dict(maps[0],initial_terrain_u8_sha256="9"*64)]),"w","p",gate.EXAMINER,"sandbox")

    def test_frozen_evidence(self):
        _, seeds, boards = identity.ladder_boards()
        self.assertEqual(len(seeds),1440)
        self.assertEqual(len(boards),1440)
        for filename, *_ in identity.SOURCES:
            ledger = Path(__file__).parent / filename
            for line in gzip.decompress(ledger.read_bytes()).splitlines():
                witness=json.loads(line)
                self.assertIn(witness["seed"],seeds)
                self.assertIn(witness["initial_terrain_u8_sha256"],boards)


if __name__ == "__main__":
    unittest.main()
