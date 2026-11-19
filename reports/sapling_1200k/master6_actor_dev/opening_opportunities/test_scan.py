import unittest
from scan import FastCarbon, classify, direct_replays, open_routes, routes


class OpeningAuditTest(unittest.TestCase):
    def setUp(self):
        self.sim=FastCarbon().reset(seed=123)
        cfg=self.sim.config
        self.cfg=dict(cfg)
        commands=[{next(iter(self.sim.obs(p)['players'][p][1])):'RECCOLLECTOR'} for p in (0,1)]
        self.sim.step(*commands)
        self.h,self.v=[next(iter(self.sim.player_workers[p])) for p in (0,1)]
        self.sim.workers[self.h].pos=100
        self.sim.workers[self.v].pos=101
        self.sim.workers[self.v].carbon=20
        self.sim.cash[1]=29

    def test_recorded_step_counterfactual_positive_and_correct_direction(self):
        pairs=[(self.h,self.v)]
        x=direct_replays(self.sim,self.cfg,[{},{}],1,pairs)
        self.assertEqual([(h['alternative_hunter'],h['cell']) for h in x],[('RIGHT',101)])
        self.assertEqual(direct_replays(self.sim,self.cfg,[{}, {self.v:'RIGHT'}],1,pairs),[])
        self.assertTrue(self.v in self.sim.workers)

    def test_public_only_price_and_liquidity(self):
        obs=self.sim.obs(0)
        self.cfg.update(recCollectorIncreaseCost=2,recPlanterCost=44)
        obs={'players':obs['players'],'step':obs['step']}
        counts, x=classify(obs,self.cfg,1)
        self.assertEqual(x['fees'],[34,44])
        self.assertEqual(counts['joint_direct'],1)
        obs['players'][1][0]=34
        self.assertEqual(classify(obs,self.cfg,1)[0]['joint'],0)
        obs['players'][1][0]=0
        obs['players'][0][2][self.h][1]=20
        self.assertEqual(classify(obs,self.cfg,1)[0]['broad_geometry'],0)

    def test_return_geometry_differs_from_open_path_and_wraps(self):
        self.assertIsNotNone(open_routes(100,104,(105,),15))
        self.assertIsNone(routes(100,104,(105,),15))
        self.assertEqual(open_routes(104,90,(97,),15)[0],1)


if __name__=='__main__':
    unittest.main()
