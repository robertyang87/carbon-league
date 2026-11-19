import copy
import math
import unittest
from model import NAMES, death_risk, destination, features, score


class PublicInputTests(unittest.TestCase):
    def setUp(self):
        self.obs = {"step":5,"player":1,"carbon":[0]*225,"players":[
            [70,{"enemy-base":36},{"op":[19,0,"COLLECTOR"]},{}],
            [100,{"our-base":0},{"focal":[20,0,"PLANTER"],"ally":[80,0,"COLLECTOR"]},{}]]}
        self.history = []
        for t,pos in ((2,16),(3,17),(4,18)):
            turn = copy.deepcopy(self.obs)
            turn["step"] = t
            turn["players"][0][2]["op"][0] = pos
            self.history.append(turn)

    def test_targets_heading_and_wrap(self):
        x,geo = features(self.obs,self.history,{},"focal","NONE")
        self.assertTrue(geo)
        by_name = dict(zip(NAMES,x))
        self.assertEqual(by_name["collector_adjacent"],1)
        self.assertEqual(by_name["collector_heading_to_cell"],1)
        self.assertEqual(by_name["collector_closing_last"],1)
        self.assertEqual(by_name["collector_closing_three"],1)
        x,geo = features(self.obs,self.history,{},"focal","UP")
        self.assertFalse(geo)
        self.assertEqual(destination(0,"UP"),210)
        self.assertEqual(destination(14,"RIGHT"),0)

    def test_no_carbon_or_future(self):
        before = features(self.obs,self.history,{},"focal","NONE")
        obscured = dict(self.obs,carbon=[object()]*225)
        self.assertEqual(features(obscured,self.history,{},"focal","NONE"),before)
        with self.assertRaises(ValueError):
            features(dict(self.obs,full_carbon=[]),self.history,{},"focal","NONE")
        with self.assertRaises(ValueError):
            features(self.obs,self.history+[self.obs],{},"focal","NONE")

    def test_callable(self):
        model = {"features":NAMES,"weights":[0.]*len(NAMES),"intercept":math.log(1/3)}
        self.assertEqual(death_risk(model,self.obs,self.history,{},"focal","UP"),0)
        self.assertAlmostEqual(death_risk(model,self.obs,self.history,{},"focal","NONE"),.25)
        with self.assertRaises(ValueError):
            score(dict(model,features=[]),[0.]*len(NAMES),True)


if __name__ == "__main__":
    unittest.main()
