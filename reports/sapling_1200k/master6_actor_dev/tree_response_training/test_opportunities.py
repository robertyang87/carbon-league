"""Synthetic predecision fixtures plus the actual embedded actor-prefix identity."""
import hashlib
import unittest
from opportunities import F, FEATURE_SHA, MOVES, commitments, neighbors, options, quote, state
from pathlib import Path

CFG = {"size":15,"episodeSteps":300,"treeLifespan":50,"plantCost":20,"plantCostInflationRatio":5,
       "plantCostInflationBase":1.235,"playerTreeProtectiveNumber":0,"seizeCost":20,"workerLimit":10,
       "recCollectorCost":30,"recPlanterCost":30,"recCollectorIncreaseCost":0,"recPlanterIncreaseCost":0}


def fixture(t=60,our_seat=0):
    own=[90.,{"b":224},{"p":[0,0,"PLANTER"],"c":[225-15,10,"COLLECTOR"]},{"a":[33,12]}]
    enemy=[80.,{"e":4},{"q":[2,0,"PLANTER"],"r":[3,160.,"COLLECTOR"]},{"x":[1,49],"y":[15,50]}]
    players=[own,enemy] if not our_seat else [enemy,own]
    return {"player":our_seat,"step":t,"players":players,"trees":{},"carbon":[11.]*225}


def detail():
    return {"slot":1,"prior":[[0,"c","NONE"]],"legal":[True]*5,"reserved":[0,0,0,0,0]}


class Opportunities(unittest.TestCase):
    def test_archival_actor_feature_identity(self):
        self.assertEqual(hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest(),FEATURE_SHA)

    def test_exact_stay_price_and_moving_never_charges(self):
        st=state(fixture(),CFG);row=options(st,{"p":"RIGHT"},"p",detail(),[])
        a=row["actions"];self.assertEqual(set(a),set(MOVES))
        self.assertTrue(a["NONE"]["can_pay_now"])
        self.assertAlmostEqual(a["NONE"]["price"],20+5*1.235**2)  # only two nonwithered trees
        self.assertEqual(a["RIGHT"]["cell"],1)
        self.assertEqual(a["RIGHT"]["current_site_type"],"seize")
        self.assertEqual((a["RIGHT"]["price"],a["RIGHT"]["current_turn_intended_cost"],a["RIGHT"]["can_pay_now"]),(20.,0.,False))
        self.assertEqual(a["DOWN"]["current_site_type"],"plant")  # age 50 cleared before pay
        self.assertEqual(quote(st,33),("own_tree",None));self.assertEqual(quote(st,4),("base",None))
        self.assertEqual(a["UP"]["cell"],210);self.assertTrue(a["UP"]["friendly_conflict"])
        self.assertTrue(a["RIGHT"]["possible_enemy_contact_next"])

    def test_protective_price_and_joint_cash_budget(self):
        obs=fixture();owner=obs["players"][0];owner[0]=53.;owner[3]={}
        owner[2]["p2"]=[16,0,"PLANTER"]
        st=state(obs,CFG);row=options(st,{"b":"RECPLANTER"},"p",detail(),[])
        a=row["actions"]["NONE"]
        self.assertEqual((a["price"],a["available_cash"],a["approved_own_hire_reserve"],a["requested_other_planter_reserve"]),(20.,3.,30.,20.))
        self.assertFalse(a["can_pay_now"])
        owner[0]=7.
        self.assertFalse(commitments(state(obs,CFG),{"b":"RECPLANTER"})[0])

    def test_masked_unknown_is_distinct_from_visible_zero(self):
        obs=fixture();st=state(obs,CFG);obs["carbon"]=[0. if i in st["visible"] else 99999. for i in range(225)]
        contaminated=state(obs,CFG);r=options(contaminated,{},"p",detail(),[])
        self.assertEqual(r["actions"]["RIGHT"]["observable_resource_sum"],0)
        self.assertTrue(any(c["carbon"] is None for c in r["actions"]["RIGHT"]["adjacent_carbon"]))
        self.assertTrue(all(c["carbon"] in (0,None) for a in r["actions"].values() for c in a["adjacent_carbon"]))

    def test_historical_root_and_strata_do_not_read_future(self):
        st=state(fixture(60,1),CFG)
        prior=fixture(57,1);prior["players"][0][2]["r"][1]=10
        before_birth=fixture(55,1);before_birth["players"][1][2].pop("p")
        row=options(st,{},'p',detail(),[state(prior,CFG),state(before_birth,CFG)])
        self.assertEqual(row["stratum_bits"],[1,0,1]);self.assertEqual(row["heavy_return"],160)
        with self.assertRaisesRegex(ValueError,"earlier same-seat"):
            options(st,{},"p",detail(),[state(fixture(61,1),CFG)])
        self.assertEqual([a["lag"] for a in row["history"]],[3,5,10])
        self.assertEqual(row["history"][0]["predecision"]["heavy_return"],10)
        self.assertFalse(row["history"][1]["predecision"]["worker_present"]);self.assertFalse(row["history"][2]["observed"])
        self.assertEqual(row["actions"]["RIGHT"]["cell"],1)  # seat 1 is physical right as well

    def test_productive_area_excludes_all_trees_and_bases(self):
        obs=fixture()
        obs["players"][0][3]={"t0":[0,40],"t1":[15,42],"t2":[16,1]}
        st=state(obs,CFG)
        cells=(set(neighbors(0))|set(neighbors(16))) - set(st["bases"])-set(st["trees"])
        self.assertEqual(set(st["footprint"]),cells)


if __name__=="__main__":
    unittest.main()
