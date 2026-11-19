"""Exercises only declaration bytes and synthetic reports, without generating or playing any maps."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import batch_clearance as audit


class SupplementReferenceTest(unittest.TestCase):
    def setUp(self):
        self.tree = audit.HERE
        queue = self.tree / 'next_intake.json'
        self.plan = {'intake_name':'intake_05', 'train_maps':audit.load(queue)['batches'][-1]['maps'],
                     'queue_sha256':audit.digest(queue), 'source_revision':'synthetic'}
        self.fifth = self.tree / 'batches' / self.plan['intake_name'] / 'protocol.json'
        self.note = self.tree / 'supplement_01/registration.json'

    def test_registered_preview(self):
        self.assertEqual(audit.supplementary_nominal_reference(self.fifth, self.plan), self.note)
        self.assertIsNone(audit.supplementary_nominal_reference(self.fifth, {**self.plan,'intake_name':'intake_04'}))

    def test_changed_board_or_game_disguised_as_registration(self):
        for attack in ('board', 'tape'):
            reg = copy.deepcopy(audit.load(self.note))
            if attack == 'board':
                reg['previous_predeclared_terrain'][str(self.plan['train_maps'][0])] = '0' * 64
            else:
                reg['played'] = [{'seed':self.plan['train_maps'][0], 'actions':[{}, {}]}]
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / 'supplement_01'
                folder.mkdir()
                (folder / 'registration.json').write_text(json.dumps(reg))
                (folder / 'register.py').write_bytes((self.note.parent / 'register.py').read_bytes())
                with patch.object(audit, 'HERE', folder.parent), self.assertRaises(ValueError):
                    audit.supplementary_nominal_reference(self.fifth, self.plan)

    def test_only_literal_preview_is_admissible(self):
        declaration = self.tree / 'protocol.json'
        seed = self.plan['train_maps'][0]
        board = 'a' * 64
        filename = str(self.note.relative_to(audit.ROOT.parent))
        allowed = {audit.relative(self.note):audit.digest(self.note)}
        row = {'candidate_seed':seed, 'frozen_git_revision':self.plan['source_revision'],
               'current_declaration':{'sha256':audit.digest(declaration)}, 'candidate_initial_terrain_u8_sha256':board,
               'official_opening_sha256':board, 'reserved':False, 'ladder_seed_or_terrain_match':False,
               'conflicts':[[filename,'exact_number_or_terrain']], 'same_terrain_matches':[],
               'local_native_or_training_maps':[], 'verdict':'blocked'}
        def validate(data):
            return audit.validate_scan(json.dumps(data).encode(),seed,board,declaration,self.plan,allowed)
        validate(row)
        for which in ('same_terrain_other_seed', 'observed', 'nominal_from_note', 'registered_seed', 'external_tape'):
            bad = copy.deepcopy(row)
            if which in ('registered_seed','external_tape'):
                source = filename if which == 'registered_seed' else 'other/actual_tape.json'
                bad['conflicts'].append([source,'recorded_seed'])
            else:
                alt = seed + 5 if which == 'same_terrain_other_seed' else seed
                method = 'observed_225_cells' if which == 'observed' else 'inferred_default_native_generator'
                bad['same_terrain_matches'].append([board,alt,method])
                bad['local_native_or_training_maps'].append({'terrain':board,'seed':alt,'method':method,'source':[filename]})
            with self.subTest(which=which), self.assertRaises(ValueError):
                validate(bad)


if __name__ == '__main__':
    unittest.main()
