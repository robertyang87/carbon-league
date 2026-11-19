import json
from pathlib import Path
import unittest
import route

class ReturnRoute(unittest.TestCase):
    def setUp(self):
        self.w=json.loads((Path(__file__).parent/"weights.json").read_bytes())
        self.cfg=dict(size=15,episodeSteps=300,treeLifespan=50,moveCost=0.,collectorAbsorptionRate=.1)
        self.obs=dict(step=10,player=0,players=[[100,{"base":82},{"u":[50,200,"COLLECTOR"]},{}],
                                              [100,{"b1":166},{"v":[127,0,"COLLECTOR"]},{"tree":[35,15]}]])
    def choices(self,old="RIGHT",**orders):return route.ranked(self.obs,{"u":old,**orders},self.cfg,self.w)
    def test_equal_homeward_moves(self):
        r=self.choices()[0]
        self.assertEqual((r["old"],r["alternative"],r["bank"]),("RIGHT","DOWN",82))
        self.assertEqual(route.D[r["new_cell"]][82],route.D[r["old_cell"]][82])
        self.assertGreater(r["expected_first_step_retained_gain"],19.)
    def test_protect_harvest_immediate_delivery_and_expiry(self):
        self.assertFalse(self.choices("NONE"))
        self.obs["players"][0][1]["base"]=51
        self.assertFalse(self.choices())
        self.obs["players"][0][1]["base"]=82
        self.obs["players"][1][3]["tree"]=[35,50]
        self.assertFalse(self.choices())
    def test_committed_friends_visible_hostile_and_insufficient_time(self):
        self.obs["players"][0][2]["team"]=[64,0,"PLANTER"]
        self.assertFalse(self.choices(team="RIGHT"))
        del self.obs["players"][0][2]["team"]
        self.obs["players"][1][2]["v"]=[64,1,"COLLECTOR"]
        self.assertFalse(self.choices())
        self.obs["players"][1][2]["v"]=[127,0,"COLLECTOR"];self.obs["step"]=297
        self.assertFalse(self.choices())
    def test_mirror_both_physical_seats(self):
        a=self.choices()
        self.obs["players"].reverse();self.obs["player"]=1
        self.assertEqual(self.choices(),a)
if __name__=="__main__":unittest.main()
