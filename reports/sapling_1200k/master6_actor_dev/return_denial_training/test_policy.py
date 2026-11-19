"""Public-only geometry tests. Synthetic tapes have no source-game identifiers or actions by the rival."""
import unittest

import policy


def config(**updates):
    cfg = dict(size=15,episodeSteps=300,workerLimit=10,recCollectorCost=30,recCollectorIncreaseCost=0,
               recPlanterCost=30,recPlanterIncreaseCost=0,treeLifespan=50,moveCost=0)
    return {**cfg,**updates}


def fixture(seat=0):
    def make(t, enemy_pos):
        friendly = [80,{"bank":13},{"light":[48,20,"COLLECTOR"]},{}]
        rival = [80,{"home":1},{"carrier":[enemy_pos,150,"COLLECTOR"]},{}]
        players = [friendly,rival] if seat==0 else [rival,friendly]
        return dict(step=t,player=seat,players=players,carbon=[0]*225)
    return make(10,34),[make(7,37),make(8,36),make(9,35)]


def run(obs,hist,cmd=None,cfg=None):
    return policy.public_turn(obs,hist,cmd or {},cfg or config())


class PublicRoutes(unittest.TestCase):
    def test_visible_root_and_all_five_actions(self):
        obs,hist=fixture()
        r,=run(obs,hist)
        self.assertEqual((r["origin"],r["cargo"],r["return_distance"]),(34,150,5))
        self.assertEqual(r["action_cells"],[34,19,35,49,33])
        self.assertEqual(r["shortest"],[False,True,False,False,True])
        self.assertTrue(all(len(row)==len(policy.NAMES) for row in r["features"]))
        self.assertEqual([x[policy.NAMES.index("repeats_last_direction")] for x in r["features"]],[0,0,0,0,1])
        self.assertEqual(len(r["options"]),1)
        opt=r["options"][0]
        self.assertEqual((opt["old"],opt["order"],opt["own_origin"],opt["cell"]),("NONE","UP",48,33))
        self.assertEqual(opt["open_shortest_old"],[19,33])
        self.assertEqual(opt["open_shortest_new"],[19])
        self.assertEqual(opt["open_old"],r["action_cells"])
        self.assertEqual(opt["open_new"],r["action_cells"][:-1])

    def test_standing_and_releasing_old_exit(self):
        obs,hist=fixture()
        obs["players"][0][2]["light"][0]=33
        r,=run(obs,hist,{"light":"RIGHT"})
        opt=next(x for x in r["options"] if x["order"]=="NONE")
        self.assertEqual(opt["old_cell"],34)
        self.assertNotIn(34,opt["open_old"])
        self.assertIn(34,opt["open_new"])
        self.assertNotIn(33,opt["open_new"])
        self.assertFalse(any(x["order"]=="RIGHT" for x in r["options"]))

    def test_population_excludes_near_or_unapproaching_but_keeps_no_options(self):
        obs,hist=fixture()
        obs["players"][0][2].clear()
        self.assertEqual(run(obs,hist)[0]["options"],[])
        obs["players"][1][2]["carrier"][1]=99.99
        self.assertEqual(run(obs,hist),[])
        obs["players"][1][2]["carrier"]=[3,100,"COLLECTOR"]
        self.assertEqual(run(obs,hist),[])
        obs["players"][1][2]["carrier"]=[4,100,"COLLECTOR"]
        self.assertEqual(len(run(obs,hist)),1) # exactly three away
        hist[0]["players"][1][2]["carrier"][0]=4
        self.assertEqual(run(obs,hist),[])

    def test_contenders_must_be_strictly_lighter_after_their_physical_move(self):
        obs,hist=fixture()
        ours=obs["players"][0][2]["light"]
        ours[1]=150
        self.assertEqual(run(obs,hist)[0]["options"],[])
        # Holding 140 loses the equality/weight test against a moving enemy now carrying 135.
        ours[:]=[33,140,"COLLECTOR"]
        self.assertTrue(any(o["order"]=="NONE" for o in run(obs,hist,{"light":"UP"})[0]["options"]))
        self.assertFalse(any(o["order"]=="NONE" for o in run(obs,hist,{"light":"UP"},config(moveCost=.1))[0]["options"]))

    def test_tied_collector_denies_cell_but_planter_does_not(self):
        obs,hist=fixture()
        own=obs["players"][0]
        own[2]["other"]=[19,150,"COLLECTOR"]
        opt,=run(obs,hist)[0]["options"]
        self.assertNotIn(19,opt["open_old"])
        own[2]["other"][2]="PLANTER"
        opt,=run(obs,hist)[0]["options"]
        self.assertIn(19,opt["open_old"])
        own[2]["other"][0]=33
        self.assertEqual(run(obs,hist)[0]["options"],[])

    def test_recruit_only_reserves_funded_capped_base_in_real_order(self):
        obs,hist=fixture()
        own=obs["players"][0]
        own[1]={"first":22,"second":33}; own[0]=30
        orders={"first":"RECCOLLECTOR","second":"RECCOLLECTOR"}
        self.assertEqual(len(run(obs,hist,orders)[0]["options"]),1)
        own[0]=60
        self.assertEqual(run(obs,hist,orders)[0]["options"],[])
        self.assertEqual(len(run(obs,hist,orders,config(workerLimit=2))[0]["options"]),1)
        # Both present collectors contribute to the price, keeping second base free.
        self.assertEqual(len(run(obs,hist,orders,config(recCollectorIncreaseCost=5))[0]["options"]),1)

    def test_wraparound_and_towards_either_home(self):
        obs,hist=fixture()
        own,enemy=obs["players"]
        enemy[1]={"east":2,"west":11}
        enemy[2]["carrier"][0]=14
        hist[0]["players"][1][1]={"east":2,"west":11}
        hist[0]["players"][1][2]["carrier"][0]=59
        r,=run(obs,hist)
        self.assertEqual([c for c,yes in zip(r["action_cells"],r["shortest"]) if yes],[0,13])

    def test_relative_seat_public_only_and_no_future(self):
        a,h=fixture()
        b,j=fixture(1)
        self.assertEqual(run(a,h),run(b,j))
        expected=run(a,h)
        for v in [a,*h]:
            v["carbon"]=object()
            v["full_carbon"]=object()
            v["hidden"]=object()
            v["opponent_commands"]=object()
        self.assertEqual(run(a,h),expected)
        with self.assertRaises(ValueError):
            run(a,h+[dict(a,step=11)])
        with self.assertRaises(ValueError):
            run(a,h+[dict(a,player=1)])
        self.assertEqual(run(dict(a,step=299),h),[])
        self.assertEqual(run(a,h[1:]),[])


if __name__=="__main__":
    unittest.main()
