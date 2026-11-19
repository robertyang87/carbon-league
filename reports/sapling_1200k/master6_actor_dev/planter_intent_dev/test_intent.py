import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import intent


class IntentTests(unittest.TestCase):
    def test_unpaid_order_changes_the_fit(self):
        row = dict(x=[0.] * len(intent.NAMES), board='map-1')
        rows = [{**row,'y':120.,'pay_realized':True},{**row,'y':-360.,'pay_realized':False}]
        self.assertAlmostEqual(intent.fit(rows)['constant'], -120 / 1.1)
        self.assertEqual(intent.fit(rows)['paid'], 1)

    def test_unseen_board_and_both_directions(self):
        sample = dict(x=[0.] * len(intent.NAMES), board='map-1', y=250., pay_realized=True)
        model = intent.fit([sample])
        with self.assertRaisesRegex(ValueError,'seen terrain'):
            intent.decide([{**sample, 'parent_pay':False}], model)
        row = {**sample,'board':'map-2','parent_pay':False}
        intent.decide([row],model)
        self.assertTrue(row['model_pay'])
        row['x'] = [2.] * len(intent.NAMES)
        self.assertFalse(intent.choose(row,model)[0])
        negative = intent.fit([{**sample,'y':-250.}])
        self.assertFalse(intent.choose({**sample, 'parent_pay':True},negative)[0])

    def test_missed_order_is_scored_against_factual_parent(self):
        row = dict(board='a',seed=0,seat=0,parent_pay=True,model_pay=False,constant_pay=True,y=-50.,pay_realized=False)
        self.assertEqual(intent.summarize([row])['equal_map_mean'],{'model':50.,'constant':0.})

    def test_sealing_prevents_replacement(self):
        with tempfile.TemporaryDirectory() as td, patch.object(intent,'HERE',Path(td)):
            path = intent.seal('output.json',{'n':1})
            intent.seal('output.json',{'n':1})
            with self.assertRaisesRegex(ValueError,'previously sealed'):
                intent.seal('output.json',{'n':2})
            self.assertEqual(json.loads(path.read_bytes()),{'n':1})


if __name__ == '__main__':
    unittest.main()
