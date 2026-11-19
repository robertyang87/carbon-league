"""No-game tests for original-seal, fixed order and outcome-independent stopping."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('training_supplement',Path(__file__).with_name('run.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.tree=Path(self.tmp.name);self.sup=self.tree/'supplement_01';self.sup.mkdir()
        self.reg={'batches':[{'name':f'intake_0{i}'} for i in (6,7,8)],'expected_original_05_prefix':['.','batches/intake_05'],
                  'goals':{'independent_training_maps':8,'completed_actually_paid_pairs':32,'minimum_distinct_maps_per_sign':2}}
        for patcher in (patch.object(s,'TRAIN',self.tree),patch.object(s,'HERE',self.sup),patch.object(s,'registered',return_value=self.reg)):
            patcher.start();self.addCleanup(patcher.stop)

    def seal(self,root,name):
        wd=root/'batches'/name;wd.mkdir(exist_ok=True,parents=True);(wd/'aggregate.json').write_text('{}')

    @staticmethod
    def fake_counts(n,neg=2):
        return {'distinct_source_maps':10,'actual_paid_pairs':n,'positive_maps':[1,2],'negative_maps':list(range(neg))}

    def test_cannot_issue_or_create_case_before_fifth_intake(self):
        self.assertEqual(s.status()['phase'],'waiting_for_original_05_seal')
        with self.assertRaisesRegex(ValueError,'previous intakes incomplete'):s.issue('intake_06',self.reg)
        self.assertFalse((self.sup/'batches').exists())

    def test_signed_prefix_issues_only_next_and_stops_on_counts_even_when_one_sign_absent(self):
        self.seal(self.tree,'intake_05')
        with patch.object(s,'signature',return_value=self.fake_counts(28)):
            self.assertEqual(s.status()['next'],'intake_06')
            with self.assertRaisesRegex(ValueError,'wrong block'):s.issue('intake_07',self.reg)
        self.seal(self.sup,'intake_06')
        with patch.object(s,'signature',side_effect=lambda areas:self.fake_counts(28 if len(areas)==2 else 32,neg=0)):
            result=s.status();self.assertIsNone(result['next']);self.assertFalse(result['support_for_fit'])
            self.assertEqual(result['phase'],'registered_count_target_reached')

    def test_exhaustion_reports_true_deficit_without_creating_more(self):
        self.seal(self.tree,'intake_05')
        for item in self.reg['batches']:self.seal(self.sup,item['name'])
        with patch.object(s,'signature',side_effect=lambda areas:self.fake_counts(20+len(areas))):
            result=s.status();self.assertEqual(result['phase'],'all_registered_blocks_exhausted')
            self.assertEqual(result['missing_paid_pairs'],7);self.assertIsNone(result['next'])


if __name__=='__main__':unittest.main()
