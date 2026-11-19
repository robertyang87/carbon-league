"""Independent regression on 32 wholly artificial paid outcomes. No original-opponent games."""
import copy
import unittest

import finish_registered as finish
from test_train import fixture
import train


class IndependentFitTests(unittest.TestCase):
    def test_held_map_check_including_unpaid_decision(self):
        rows=[{**r,'defer_margin':100.,'pay_margin':100.+r['y']} for r in fixture()]
        rows.append({**rows[0],'uid':'unpaid','filled':False,'pay_debit':0.})
        evaluation=train.leave_one_map_out(rows);weights,_=train.ridge([r for r in rows if r['filled']])
        status={'support':train.support(rows),'eligible_for_development':evaluation['permits_mechanism_check'],'stage':'synthetic'}
        audit=finish.verify(status,rows,evaluation,weights,train.PROTOCOL)
        self.assertEqual(audit['counts']['paid_pairs'],32)
        self.assertEqual(sum(f['intent'] for f in audit['maps']),33)
        self.assertTrue(audit['passed'])
        broken=copy.deepcopy(evaluation);fold=broken['folds'][0]
        fold['fold_weights']['training_terrains'].append(fold['board'])
        with self.assertRaisesRegex(ValueError,'leakage'):finish.verify(status,rows,broken,weights,train.PROTOCOL)
        broken=copy.deepcopy(evaluation);broken['folds'][0]['roots'][0]['gain']+=1
        with self.assertRaisesRegex(ValueError,'numeric check'):finish.verify(status,rows,broken,weights,train.PROTOCOL)

    def test_insufficient_count_issues_no_fit(self):
        rows=[{**r,'defer_margin':100.,'pay_margin':100.+r['y']} for r in fixture()[:-1]]
        status={'support':train.support(rows),'eligible_for_development':False}
        self.assertFalse(finish.verify(status,rows,None,None,train.PROTOCOL)['passed'])
        with self.assertRaisesRegex(ValueError,'premature'):finish.verify(status,rows,None,{'x':0},train.PROTOCOL)

if __name__=='__main__':unittest.main()
