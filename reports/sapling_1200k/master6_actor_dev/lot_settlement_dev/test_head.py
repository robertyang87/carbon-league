"""Public-state invariances, delayed pay and the single-file read-only head."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from protocol import HERE
import head
from data import encode_events

class Estimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w=json.loads(gzip.decompress((HERE/"weights.json.gz").read_bytes()))
        ns={};exec(compile((HERE/"lot_forecast.py").read_bytes(),"copied_forecast.py","exec"),ns)
        cls.solo=staticmethod(ns["forecast"])
    def obs(self,t):
        return dict(step=t,player=1,players=[[12,{"b0":165},{"a":[55,7,"COLLECTOR"]},{}],
                    [100,{"b1":65},{"r":[78,130,"COLLECTOR"]},{}]],carbon=[0.]*225)
    def test_bounded_outputs_and_equivalence(self):
        o=self.obs(296);hs=[self.obs(291),self.obs(295)]
        r=head.predict(self.w,o,hs,1,"r","UP",{"r":"UP"})
        self.assertEqual(r,self.solo(o,hs,"r","UP",{"r":"UP"}))
        self.assertAlmostEqual(sum(r["outcome_fraction"].values()),1)
        self.assertLessEqual(r["expected_cash"],130+1e-8)
        self.assertLessEqual(r["expected_time_if_cash"],3)
        self.assertEqual(r["cash_by_deadline"]["3"],r["expected_cash"])
        curve=list(r["cash_by_deadline"].values())
        self.assertTrue(all(v<=w+1e-7 for v,w in zip(curve,curve[1:])))
    def test_cannot_use_future_or_wrong_seat_or_zero_principal(self):
        o=self.obs(120)
        with self.assertRaises(ValueError):self.solo(o,[self.obs(120)],"r")
        with self.assertRaises(ValueError):head.features(o,[],0,"a")
        o["players"][1][2]["r"][1]=0
        with self.assertRaises(ValueError):self.solo(o,[],"r")
    def test_mask_seed_hidden_map_and_names(self):
        o=self.obs(79);r=self.solo(o,[],"r","UP",{"r":"UP"})
        o["full_carbon"]=[100]*225;o["randomSeed"]=1199;o["opponentName"]="Master3s2"
        o["players"][1][2]["new_id"]=o["players"][1][2].pop("r")
        self.assertEqual(r,self.solo(o,[],"new_id","UP",{"new_id":"UP"}))
    def test_ten_step_unsettled_amount_and_transfer(self):
        events=[dict(step=3,kind="relay",reason="transfer",fraction=1.,is_terminal_allocation=False),
                dict(step=6,kind="tax",reason="enemy_tree_tax",fraction=.1,is_terminal_allocation=True),
                dict(step=19,kind="own_relay",reason="own_bank",fraction=.9,is_terminal_allocation=True)]
        y,extras=encode_events(events,1)
        self.assertEqual(y[:2].tolist(),[0.,.9])
        self.assertAlmostEqual(extras[0],.9);self.assertAlmostEqual(extras[1],.9)
        self.assertEqual(extras[2],0.)
        self.assertAlmostEqual(y[head.CATEGORIES.index("tax")],.1)
if __name__=="__main__":unittest.main()
