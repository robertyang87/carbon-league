"""Admission and evidence-path checks without simulating a game or loading any held-out map."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import evidence as ev


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ev.canonical(value))


class SupplementEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("protocol.json", "next_intake.json", "training_labels.json", "batches/intake_03/protocol.json",
                     "batches/intake_03/aggregate.json", "supplement_01/registration.json", "supplement_01/register.py"):
            path = self.root / name
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ev.CODE / name, path)
        self.reg = ev.supplement_registration(self.root)
        self.queue = ev.read(self.root / "next_intake.json")
        self.original = [self.root / n for n in self.reg["expected_original_05_prefix"]]
        self.rows = self.reg["batches"]
        self.areas = [ev.intake_folder(self.root, r["name"]) for r in self.rows]

    def visible(self):
        return [n for n,_,_ in ev.batches(self.root)]

    def sign_original(self, paid=24):
        for d in self.original:
            if not (d / 'training_labels.json').is_file():
                write(d / 'training_labels.json', {'fixture':d.name})
            if d != self.root and not (d/'aggregate.json').is_file():
                write(d/'aggregate.json', {})
        self.summary(self.original, paid)

    def summary(self, dirs, paid):
        write(dirs[-1] / 'aggregate.json', {'filled_pairs':paid,'distinct_maps':2*len(dirs),
              'source_intakes':[{'dir':str(d.relative_to(self.root)), 'labels_sha256':ev.sha(d/'training_labels.json')} for d in dirs]})

    def issue(self, idx):
        row, d = self.rows[idx], self.areas[idx]
        past = self.original + self.areas[:idx]
        plan = {**ev.read(self.root / 'batches/intake_03/protocol.json'),
                'intake_name':row['name'],'schema':'sapling-tree-response-training-supplement-batch-v1',
                'train_maps':row['maps'],'training_job_registry':row['jobs'],'queue_sha256':ev.SUPPLEMENT_HASH,
                'prior_receipts':{str(p.relative_to(self.root)):ev.sha(p/'training_labels.json') for p in past},
                'registered_utc':self.reg['registered_utc'],'source_revision':self.reg['revision_at_registration']}
        write(d/'protocol.json', plan)
        return plan

    def test_original_listing_stable_and_only_double_sealed_supplement_is_visible(self):
        original_names = ['intake_01'] + [r['name'] for r in self.queue['batches']]
        self.assertEqual(self.visible(), original_names)
        self.sign_original()
        self.issue(0)
        self.assertEqual(self.visible(), original_names)
        write(self.areas[0]/'training_labels.json', {'fixture':'first signed labels'})
        self.assertEqual(self.visible(), original_names)
        self.summary(self.original+self.areas[:1],26)
        self.assertEqual(self.visible(), original_names + [self.rows[0]['name']])
        second = self.issue(1)
        second['training_job_registry'][0]['seat'] ^= 1
        write(self.areas[1]/'protocol.json', second)
        with self.assertRaisesRegex(ValueError,'diverges'):
            self.visible()

    def test_rejects_unsealed_fifth_prefix_and_data_already_sufficient(self):
        self.sign_original(paid=32)
        self.issue(0)
        with self.assertRaisesRegex(ValueError,'data counts'):
            self.visible()
        (self.original[-1]/'aggregate.json').unlink()
        with self.assertRaisesRegex(ValueError,'previous double seals'):
            self.visible()

    def test_wrong_order_replaced_registration_and_gate_before_both_counts(self):
        self.sign_original()
        write(self.areas[1]/'protocol.json',{})
        with self.assertRaisesRegex(ValueError,'previous double seals'):
            self.visible()
        path = self.root / 'supplement_01/registration.json'
        reg = ev.read(path)
        reg['batches'][0]['terrain_at_queue_time'].clear()
        write(path,reg)
        with self.assertRaisesRegex(ValueError,'unrecognized supplemental registration'):
            self.visible()
        self.assertTrue(ev.count_shortfall(7,32,self.reg['goals']))
        self.assertTrue(ev.count_shortfall(8,31,self.reg['goals']))
        self.assertFalse(ev.count_shortfall(8,32,self.reg['goals']))

    def test_cumulative_feature_paths_are_relative_to_tree_root(self):
        area = self.areas[0]
        lab = {'features':'case_s/root.json.gz','target_delta_official_margin':12,'pay_intent_filled':True,'observed_stratum':[1,0,1]}
        write(area/'case_s/training_label.json',lab)
        record = {'batch':self.rows[0]['name'],'label':lab,'board':'terrain','seat':1,'t':5,'uid':'worker','seed':17}
        folder = str(area.relative_to(self.root))
        feature = folder+'/case_s/root.json.gz'
        target = folder+'/case_s/training_label.json'
        sample = {'source_intake':folder,'features':feature,'label':target,'label_sha256':ev.sha(self.root/target),
                  'key':[ev.PARENT,'terrain',1,5,'worker'],'observed_stratum':[1,0,1],'seed':17,'seat':1,'filled':True,'target':12}
        expected = {'policy_sha256':ev.PARENT,'source_intakes':[],'distinct_maps':1,'complete_pairs':1,'filled_pairs':1,
                    'positive_maps':[17],'negative_maps':[],'fit_support_gate':False,'held_out_games':0,
                    'fit_performed':False,'examples':[sample]}
        write(area/'aggregate.json', expected)
        ev.check_aggregate(self.root,area,[record],{},1)
        sample['features'] = 'batches/'+area.name+'/case_s/root.json.gz'
        write(area/'aggregate.json', expected)
        with self.assertRaisesRegex(ValueError,'cumulative'):
            ev.check_aggregate(self.root,area,[record],{},1)


if __name__ == '__main__':
    unittest.main()
