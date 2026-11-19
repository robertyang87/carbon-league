"""Exercise the byte-bound map-reservation exception without scanning or playing any new map."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import reference_receipts as refs


class ReferenceReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve().parent
        cls.plan = json.loads((here / 'batches/intake_07/protocol.json').read_bytes())
        batch = next(row for row in json.loads((here / 'registration.json').read_bytes())['batches']
                     if row['name'] == cls.plan['intake_name'])
        cls.seed = cls.plan['train_maps'][0]
        cls.board = batch['terrain_at_queue_time'][str(cls.seed)]
        cls.old = str(Path(refs.ROOT.name) / refs.OLD)
        cls.new = str(Path(refs.ROOT.name) / refs.NEW)

    def check_scan(self, conflicts, **extras):
        scan = {'conflicts': conflicts, **extras}
        original = copy.deepcopy(scan)
        result = refs.checked_scan(scan, self.seed, self.board, self.plan)
        self.assertEqual(scan, original)
        return result

    def test_proven_local_reservations_leave_other_workspace_and_git_evidence(self):
        foreign = 'another_checkout/' + refs.OLD
        unexpected = 'reports/real_game_with_the_same_seed.json'
        allowed = [[self.old, 'exact_number_or_terrain'], [self.new, 'exact_number_or_terrain']]
        unknown = [foreign, 'exact_number_or_terrain']
        git = ['frozen_git', 'unexpected_exact_references', [refs.OLD, unexpected]]
        row, proofs = self.check_scan([*allowed, unknown, git])
        self.assertEqual(proofs, refs.PINNED)
        self.assertEqual(row['conflicts'], [unknown, ['frozen_git', 'unexpected_exact_references', [unexpected]]])

    def test_recorded_maps_ranges_and_terrain_rows_remain_blocking(self):
        blockers = [[self.old, 'recorded_seed'], [self.new, 'numeric_interval', 'registered range'],
                    [self.old, 'exact_number_or_terrain', 'not a literal-only hit']]
        observed = [[self.board, self.seed, 'observed_trajectory']]
        row, proofs = self.check_scan(blockers, same_terrain_matches=observed, reserved=True,
                                      ladder_seed_or_terrain_match=True)
        self.assertEqual(proofs, {})
        self.assertEqual(row['conflicts'], blockers)
        self.assertEqual(row['same_terrain_matches'], observed)
        self.assertTrue(row['reserved'])
        self.assertTrue(row['ladder_seed_or_terrain_match'])

    def test_wrong_registration_cannot_turn_an_exact_hit_into_a_reservation(self):
        scan = {'conflicts': [[self.old, 'exact_number_or_terrain']]}
        for modified in ({**self.plan, 'intake_name': 'intake_06'},
                         {**self.plan, 'train_maps': list(reversed(self.plan['train_maps']))}):
            with self.subTest(key=modified['intake_name']), self.assertRaises(ValueError):
                refs.checked_scan(scan, self.seed, self.board, modified)
        with self.assertRaises(ValueError):
            refs.checked_scan(scan, self.seed - 1, self.board, self.plan)

    def test_same_physical_board_from_another_ladder_seed_is_rejected(self):
        old = json.loads((refs.ROOT / refs.OLD).read_bytes())
        name = old['identity_sources'][0][0]
        published = json.loads(gzip.decompress((refs.BASE / name).read_bytes()).splitlines()[0])
        self.assertNotEqual(self.seed, published['seed'])
        for key in (self.old, self.new):
            with self.subTest(which=key), self.assertRaisesRegex(ValueError, 'opening|seed'):
                refs.checked_scan({'conflicts': [[key, 'exact_number_or_terrain']]},
                                  self.seed, published['initial_terrain_u8_sha256'], self.plan)

    def test_changed_pinned_bytes_before_or_after_receipt_chain_are_rejected(self):
        original = Path.read_bytes
        for altered in (refs.OLD, refs.NEW):
            edited = refs.ROOT / altered
            def replaced(path):
                data = original(path)
                return data + b' ' if path == edited else data
            with self.subTest(altered=altered), patch.object(Path, 'read_bytes', replaced), self.assertRaises(ValueError):
                self.check_scan([[self.new, 'exact_number_or_terrain']])

    def test_archive_digest_also_covers_actual_terrain_rows(self):
        source = json.loads((refs.ROOT / refs.OLD).read_bytes())['identity_sources'][0][0]
        archive = refs.BASE / source
        original = Path.read_bytes
        def altered(path):
            data = original(path)
            return gzip.compress(gzip.decompress(data) + b'\n', mtime=0) if path == archive else data
        with patch.object(Path, 'read_bytes', altered), self.assertRaisesRegex(ValueError, 'map evidence'):
            self.check_scan([[self.old, 'exact_number_or_terrain']])

    def test_git_hit_requires_the_pinned_bytes_at_the_frozen_revision(self):
        version = self.plan['source_revision']
        path = f'{version}:{refs.OLD}'
        probe = subprocess.check_output(['git', 'show', path], cwd=refs.ROOT)
        self.assertEqual(hashlib.sha256(probe).hexdigest(), refs.PINNED[refs.OLD])
        hit = [['frozen_git', 'unexpected_exact_references', [refs.OLD]]]
        result, proof = self.check_scan(hit)
        self.assertFalse(result['conflicts'])
        self.assertEqual(proof, {refs.OLD: refs.PINNED[refs.OLD]})
        with patch.object(refs.subprocess, 'check_output', return_value=probe + b' '), self.assertRaises(ValueError):
            self.check_scan(hit)


if __name__ == '__main__':
    unittest.main()
