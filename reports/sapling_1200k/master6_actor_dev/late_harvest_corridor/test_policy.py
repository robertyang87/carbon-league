"""Artificial public states for route planning, threat response and fog discipline."""
import copy
import unittest

from policy import Context, choose, route


CFG=dict(size=15,episodeSteps=300,smeltCost=0,moveCost=0,treeLifespan=50,
         initialCollectRate=.3,collectDecreaseRate=0,collectorAbsorptionRate=.1)
ME="player-0-worker-2"
THEM="player-1-worker-2"

def state(t=290):
    obs={"step":t,"player":0,"carbon":[0.]*225,"trees":{},
         "players":[[100,{"b":64,"c":160},{ME:[66,40.,"COLLECTOR"]},{}],
                    [100,{"d":70,"e":154},{THEM:[89,15.,"COLLECTOR"]},{}]]}
    obs["carbon"][66]=80.
    return obs

class Routes(unittest.TestCase):
    def test_gain_cancellation_and_both_bases(self):
        obs=state();base={ME:"LEFT"}
        chosen,plan,a=choose(obs,{},base,CFG)
        self.assertEqual((a["status"],a["stage"],chosen),("start","harvest",{}))
        self.assertEqual((plan["worker"],plan["home"]), (ME,64))
        new=copy.deepcopy(obs);new["step"]+=1
        new["players"][0][2][ME][1]+=24;new["carbon"][66]=56.
        new["players"][1][2][THEM][0]=67
        # The enemy is now adjacent to the mine and cannot contest cell 65 in one move.
        move,p,signals=choose(new,{290:obs},base,CFG,plan,allow_start=False)
        self.assertEqual((move[ME],signals["stage"]),("LEFT","return"))
        self.assertEqual(p["holds"],0)
        # Repeated unknown forecasts cannot retain an old plan after a turn has been skipped.
        self.assertEqual(choose(dict(new,step=295),{},base,CFG,p,False)[2]["status"],"abort")
        new=state();new["players"][0][1]={"other":160,"closest":68}
        self.assertEqual(choose(new,{}, {ME:"RIGHT"},CFG)[1]["home"],68)

    def test_mask_stale_or_future_ground_and_occupancy(self):
        x=state(); x["carbon"][66]=0.
        x["carbon"][65]=100.; x["full_carbon"]=[100]*225
        self.assertEqual(choose(x,{291:dict(x,step=291)}, {ME:"LEFT"},CFG)[2]["status"],"fallback")
        x=state()
        x["players"][0][2]["friend"]=[81,1.,"COLLECTOR"]
        self.assertEqual(choose(x,{}, {ME:"LEFT","friend":"UP"},CFG)[2]["status"],"fallback")

    def test_no_harvest_on_live_tree_and_explicit_enemy_tax(self):
        x=state();x["players"][1][3]["tree"]=[66,4]
        ctx=Context(x,{}, {ME:"LEFT"},CFG)
        self.assertIsNone(route(ctx,ME,66,1,1))
        x["players"][1][3]["tree"]=[67,4];ctx=Context(x,{}, {ME:"LEFT"},CFG)
        out=route(ctx,ME,66,1,1)
        self.assertAlmostEqual(out["harvest"],24.)
        self.assertAlmostEqual(out["enemy_tax"],6.4)
        self.assertEqual(choose(dict(x,step=295),{}, {ME:"LEFT"},CFG)[2]["status"],"fallback")

    def test_no_new_mission_after_previous_one(self):
        x=state()
        self.assertEqual(choose(x,{}, {ME:"LEFT"},CFG,allow_start=False)[2]["status"],"fallback")

if __name__=="__main__":unittest.main()
