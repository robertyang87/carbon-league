"""No native games: a seventh-intake release is ordered against the actual tree-head evidence."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import economy_lease as lease


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    return lease.sha(path)


def token(value):
    return hashlib.sha256(value.encode()).hexdigest()


class EconomyOrderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.train = Path(self.tmp.name)
        self.sup = self.train / 'supplement_01'
        self.head = self.train / 'response_head'
        self.seven = self.sup / 'batches/intake_07'
        self.run = self.head / 'development_runs'
        self.reg = lease.load(lease.TRAIN / 'protocol.json')
        shutil.copyfile(lease.TRAIN / 'protocol.json', self.train / 'protocol.json')
        self.examiner = lease.sha(lease.ROOT / 'data/master3s2_source.py')
        self.head.mkdir()
        shutil.copyfile(lease.HEAD / 'protocol.json', self.head / 'protocol.json')
        self.head_registration = lease.sha(self.head / 'protocol.json')
        self.maps = self.reg['development_maps']
        self.boards = {str(s): token(str(s)) for s in self.maps}
        self.locks = patch.multiple(lease, TRAIN=self.train, HERE=self.sup, HEAD=self.head)
        self.locks.start()
        self.addCleanup(self.locks.stop)

    def seal(self, maps=12, paid=32):
        source = [{'dir': 'supplement_01/batches/intake_07',
                   'labels_sha256': write(self.seven / 'training_labels.json', {'used': True})}]
        write(self.seven / 'aggregate.json', {'source_intakes': source, 'distinct_maps': maps, 'filled_pairs': paid})
        self.completion = write(self.sup / 'completion.json',
                                {'phase': 'registered_count_target_reached', 'counts': {'source_intakes': source,
                                 'actual_paid_pairs': paid, 'distinct_source_maps': maps}})
        self.paid = paid

    def qualify(self, passed):
        status = write(self.head / 'status.json', {'eligible_for_development': passed})
        sample = write(self.head / 'samples.json.gz', [])
        write(self.head / 'independent_acceptance.json',
              {'completion_sha256': self.completion, 'counts': {'paid_pairs': self.paid}, 'passed': passed,
               'status_sha256': status, 'sample_tape_sha256': sample})

    def finish_both_arms(self):
        packages = {a: {'pack_sha256': token(a), 'weight_sha256': token(a + '_weights')}
                    for a in ('parent', 'candidate')}
        packages['parent'] = {'pack_sha256': self.reg['parent_pack_sha256'],
                              'weight_sha256': self.reg['parent_weight_manifest_sha256']}
        arena = {'data/master3s2_source.py': self.examiner}
        engine = {'official.py': token('official.py')}
        release = {'development_protocol': self.head_registration, 'source_protocol': lease.sha(self.train / 'protocol.json'),
                   'maps': self.maps, 'boards': self.boards, 'cpus': [0, 1], 'packages': packages,
                   'master3s2_sha256': self.examiner,
                   'official_engine': engine, 'arena_sources': arena}
        release_hash = write(self.run / 'release.json', release)
        pair_rows = {(s, seat): {'seed': s, 'seat': seat} for s in self.maps for seat in (0, 1)}
        for arm, package in packages.items():
            folder = self.run / arm
            native_rows = []
            official_rows = []
            for (s, seat), pair in pair_rows.items():
                filename = f'raw_{s}_{seat}.json.gz'
                raw = {'seed': s, 'candidate_seat': seat, 'fixture_arm': arm,
                       'initial_terrain_u8_sha256': self.boards[str(s)]}
                sig = write(folder / filename, raw)
                scores = [3.0, 2.0]
                outcome = 'win' if seat == 0 else 'loss'
                margin = scores[seat] - scores[1 - seat]
                native = {'seed': s, 'candidate_seat': seat, 'raw_file': filename, 'raw_sha256': sig,
                          'official_score_by_seat': scores, 'actual_transitions': 299, 'result': outcome,
                          'initial_terrain_u8_sha256': self.boards[str(s)], 'margin': margin}
                official = {'seed': s, 'seat': seat, 'transcript_sha256': sig, 'statuses': ['DONE', 'DONE'],
                            'map_sha256': self.boards[str(s)], 'official_reward_by_seat': scores,
                            'steps': 299, 'result': outcome}
                pair[arm] = {**official, 'margin': margin}
                native_rows.append(native)
                official_rows.append(official)
            header = {**package, 'examiner_md5': self.reg['examiner_md5'], 'source_sha256': arena,
                      'seeds': self.maps, 'cpus': [0, 1], 'plan_sha256': None}
            write(folder / 'results.json', {'header': header, 'status': 'complete', 'games': native_rows})
            write(folder / 'official_replay.json',
                  {'runs': {arm: {'pack_sha256': package['pack_sha256'], 'games': official_rows}},
                   'engine_source_sha256': engine})
        write(self.run / 'comparison.json',
              {'games': 8, 'formal': False, 'release_sha256': release_hash, 'pairs': list(pair_rows.values())})

    def test_no_seal_or_support_shortfall_needs_no_tree_court(self):
        self.assertIsNone(lease.predecessor())
        for maps, paid in ((12, 31), (7, 32)):
            with self.subTest(maps=maps, paid=paid):
                self.seal(maps, paid)
                rec = lease.predecessor()
                self.assertEqual(rec['tree_head'], 'insufficient_support')
                self.assertEqual(rec['training_counts'], [maps, paid])

    def test_changed_seventh_labels_rejected_even_with_insufficient_support(self):
        self.seal(paid=31)
        write(self.seven / 'training_labels.json', {'used': False})
        with self.assertRaises(ValueError):
            lease.predecessor()

    def test_sufficient_support_waits_for_held_terrain_acceptance(self):
        self.seal()
        self.assertIsNone(lease.predecessor())
        self.qualify(False)
        rec = lease.predecessor()
        self.assertEqual(rec['tree_head'], 'held_terrain_failed')
        self.assertEqual(rec['head_acceptance_sha256'], lease.sha(self.head / 'independent_acceptance.json'))
        self.qualify(True)
        self.assertIsNone(lease.predecessor())

    def test_changed_training_cutoff_or_eligibility_bit_rejected(self):
        self.seal()
        self.qualify(False)
        write(self.head / 'status.json', {'eligible_for_development': True})
        with self.assertRaises(ValueError):
            lease.predecessor()
        self.qualify(True)
        write(self.sup / 'completion.json', {})
        with self.assertRaises(ValueError):
            lease.predecessor()

    def test_replaced_training_samples_cannot_release_failed_held_gate(self):
        self.seal()
        self.qualify(False)
        write(self.head / 'samples.json.gz', [{'substitute': True}])
        with self.assertRaises(ValueError):
            lease.predecessor()

    def test_both_arms_both_physical_seats_on_each_map_release_the_court(self):
        self.seal()
        self.qualify(True)
        self.finish_both_arms()
        rec = lease.predecessor()
        self.assertEqual(rec['tree_head'], 'eight_games_complete')
        self.assertEqual(rec['head_development_sha256'], lease.sha(self.run / 'comparison.json'))

    def test_threshold_matches_specific_training_signature(self):
        self.seal(paid=34)
        self.qualify(True)
        path = self.head / 'independent_acceptance.json'
        certificate = lease.load(path)
        certificate['counts']['paid_pairs'] = 32
        write(path, certificate)
        with self.assertRaises(ValueError):
            lease.predecessor()

    def test_changed_examiner_or_model_header_rejected(self):
        self.seal()
        self.qualify(True)
        self.finish_both_arms()
        path = self.run / 'candidate/results.json'
        for key in ('weight_sha256', 'examiner_md5', 'pack_sha256'):
            with self.subTest(key=key):
                native = lease.load(path)
                original = native['header'][key]
                native['header'][key] = token('other')
                write(path, native)
                with self.assertRaises(ValueError):
                    lease.predecessor()
                native['header'][key] = original
                write(path, native)

    def test_short_arm_or_substitute_raw_is_rejected(self):
        self.seal()
        self.qualify(True)
        self.finish_both_arms()
        path = self.run / 'candidate/results.json'
        native = lease.load(path)
        write(path, {**native, 'games': native['games'][:3]})
        with self.assertRaises(ValueError):
            lease.predecessor()
        write(path, native)
        write(self.run / 'candidate' / native['games'][0]['raw_file'], {'replaced': True})
        with self.assertRaises(ValueError):
            lease.predecessor()

    def test_four_rows_cannot_be_four_copies_of_the_same_official_game(self):
        self.seal()
        self.qualify(True)
        self.finish_both_arms()
        path = self.run / 'parent/official_replay.json'
        report = lease.load(path)
        recorded = report['runs']['parent']['games'][0]
        report['runs']['parent']['games'] = [recorded] * 4
        write(path, report)
        with self.assertRaises(ValueError):
            lease.predecessor()

    def test_comparison_scores_must_match_corresponding_official_seat(self):
        self.seal()
        self.qualify(True)
        self.finish_both_arms()
        path = self.run / 'comparison.json'
        result = lease.load(path)
        wrong = result['pairs'][0]['candidate']['official_reward_by_seat']
        wrong.reverse()
        write(path, result)
        with self.assertRaises(ValueError):
            lease.predecessor()


if __name__ == '__main__':
    unittest.main()
