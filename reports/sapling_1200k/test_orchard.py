"""Regressions for the planter's exact view, price, phase and receipt labels."""
import unittest
from orchard_train import events
from orchard_value import orchard_features


def view(t, own_trees=None, enemy_trees=None, tree_income=None):
    own = [250., {"a":1}, {"p":[0,0,"PLANTER"]}, own_trees or {}]
    other = [250., {"b":112}, {}, enemy_trees or {}]
    carbon = [0.]*225
    for cell in (210,15,14,1):
        carbon[cell]=20.
    return {"step":t,"players":[own,other],"trees":tree_income or {},"carbon":carbon}


class OrchardTests(unittest.TestCase):
    def test_price_uses_live_count_and_first_tree_discount(self):
        board=view(40,enemy_trees={"other":[42,49]})
        first=orchard_features(board,board["carbon"],0,0,40,{})
        self.assertEqual(first[1],20.)
        board["players"][0][3]["ours"]=[40,1]
        paid=orchard_features(board,board["carbon"],0,0,40,{})
        self.assertAlmostEqual(paid[1],20+5*1.235**2)
        board["players"][1][3]["dead"]=[66,50]
        self.assertEqual(orchard_features(board,board["carbon"],0,0,40,{}),paid)

    def test_hidden_and_own_cell_have_no_effect_and_wrap_is_visible(self):
        board=view(278)
        public=board["carbon"]
        before=orchard_features(board,public,0,0,278,{})
        self.assertAlmostEqual(before[2][1],.8)
        board["full_carbon"]=[91.]*225
        public[0]=97.
        public[88]=93.
        self.assertEqual(before,orchard_features(board,public,0,0,278,{}))
        public[210]=0.
        self.assertAlmostEqual(orchard_features(board,public,0,0,278,{})[2][1],.6)

    def test_last_step_seizure_and_same_id_recapture(self):
        own="player-0-tree-0"
        views=[view(0), view(1,{own:[0,1]},tree_income={own:[None,0]}),
               view(2,{own:[0,2]},tree_income={own:[None,8]}),
               view(3,enemy_trees={own:[0,3]},tree_income={own:[None,6]}),
               view(4,{own:[0,4]},tree_income={own:[None,7]})]
        for obs in views: obs["players"][1][2]={"p2":[0,0,"PLANTER"]}
        steps=[[{"observation":o},{"observation":{"carbon":o["carbon"]}}] for o in views]
        rows=events({"configuration":{"episodeSteps":5},"steps":steps},{"id":"g","seed":3})
        self.assertEqual([(r["kind"],r["seat"],r["gross"]) for r in rows],
                         [("plant",0,8.),("seize",1,6.),("seize",0,7.)])
        self.assertEqual(rows[-1]["x"][0],1/50)
        blocked=view(4,enemy_trees={own:[0,50]})
        self.assertIsNone(orchard_features(blocked,blocked["carbon"],0,0,4,{}))

if __name__=="__main__":
    unittest.main()
