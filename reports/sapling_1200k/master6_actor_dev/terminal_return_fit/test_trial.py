from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trial


class TerminalFitTests(unittest.TestCase):
    def test_labels_are_strictly_increasing_and_draw_is_center(self):
        prior = None
        for m in (-10000,-500,-1,0,1,500,10000):
            result = trial.terminal(m)
            if prior: self.assertTrue(all(a < b for a,b in zip(prior,result)))
            if m==0: self.assertEqual(result,(.5,0.))
            prior = result
        with self.assertRaises(ValueError):trial.terminal(float('nan'))

    def test_nonzero_pairing_reversed_and_blind(self):
        per={"plus":[230.,-20.],"minus":[-200.,-700.]}
        def row(arm, *unused):
            self.assertIn(arm,('plus','minus'))
            return [{'J':trial.terminal(m)[1]} for m in per[arm]]
        with patch.object(trial,'read_arm',side_effect=row), patch.object(trial,'prepare',return_value=(b'',{'pack_sha256':'fake'})):
            baseline=trial.compute_fit()
            self.assertTrue(baseline['nonzero_update'])
            expected=sum(trial.terminal(p)[1]-trial.terminal(m)[1] for p,m in zip(per['plus'],per['minus']))/2
            self.assertAlmostEqual(baseline['mean_J_plus_minus'],expected)
            self.assertAlmostEqual(baseline['theta'][0],min(.75,3*expected/1.3))
            per['minus'],per['plus']=per['plus'],per['minus']
            reverse=trial.compute_fit()
            self.assertEqual(reverse['theta'],[-v for v in baseline['theta']])

    def test_no_difference_yields_zero(self):
        with patch.object(trial,'read_arm',return_value=[{'J':0.2},{'J':-0.1}]),patch.object(trial,'prepare',return_value=(b'',{'pack_sha256':'fake'})):
            fit=trial.compute_fit()
        self.assertFalse(fit['nonzero_update'])
        self.assertEqual(fit['theta'],[0.]*4)

    def test_immutable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            name=Path(temp)/'checkpoint.json'
            trial.frozen_file(name,{'theta':.3})
            trial.frozen_file(name,{'theta':.3})
            with self.assertRaises(ValueError):trial.frozen_file(name,{'theta':-.3})


if __name__=='__main__':
    unittest.main()
