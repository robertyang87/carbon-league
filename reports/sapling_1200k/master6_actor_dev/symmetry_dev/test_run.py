import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run

class FourViewDecisionTests(unittest.TestCase):
    def roll(self,changes):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for name in ('release','mechanism','arena_binding'): (path/f'{name}.json').write_text('{}')
            (path/'protocol.json').write_text(json.dumps({'maps':[101,102]}))
            for arm in run.ARMS:
                (path/arm).mkdir()
                for name in ('results','official_replay'):(path/arm/f'{name}.json').write_text('{}')
            pair={a:{} for a in run.ARMS}
            for i,inc in enumerate(changes):
                s,p=[101,102][i//2],i%2
                for arm,margin in (('parent',-1000.),('candidate',-1000.+inc)):
                    pair[arm][s,p]={'margin':margin,'U':.5/(1+__import__('math').exp(-margin/1000)),'outcome':'loss'}
            with patch.object(run,'HERE',path):return run.compare(pair)

    def test_sign_and_three_positive_seats_are_distinct(self):
        two=self.roll([100.,-10.,90.,-10.])
        self.assertEqual(two['verdict'],'retain_parent');self.assertTrue(two['criteria']['positive_each_map'])
        three=self.roll([100.,-10.,3.,4.])
        self.assertEqual(three['verdict'],'exploratory_followup_signal')
        self.assertFalse(three['formal'])
        rolled=self.roll([100.,10.,3.,-4.])
        self.assertFalse(rolled['criteria']['positive_each_map'])

if __name__=='__main__':unittest.main()
