import unittest
import sys
from pathlib import Path

from policy import LateRecruit
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'code'))
from fastsim import DEFAULT_CONFIG


class LateWorkforceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(DEFAULT_CONFIG)
        self.base=112;self.target=113
        self.obs=dict(step=250,player=0,carbon=[0]*225,players=[
            [30,{'b':self.base},{'p':[20,0,'PLANTER']},{}],
            [90,{'e':197},{},{}]])
        self.cmd={'b':'RECPLANTER'}
        self.head=LateRecruit()
        self.head.seat=0;self.head.last=249;self.head.known[self.target]=(249,68.)

    def run_policy(self):
        return self.head.propose(self.obs,self.cfg,self.cmd)

    def test_equal_wage_with_exact_last_thirty_is_allowed(self):
        alt,e=self.run_policy()
        self.assertEqual(alt,{'b':'RECCOLLECTOR'});self.assertGreaterEqual(e['bank_estimate'],35)
        self.assertEqual(self.cmd['b'],'RECPLANTER')
        self.assertEqual(e['path'],[self.target])

    def test_actual_prices_cash_and_free_worker_cap(self):
        self.cfg['recCollectorCost']=31
        self.assertEqual(self.run_policy()[1]['reason'],'different_price')
        self.setUp();self.obs['players'][0][0]=29
        self.assertEqual(self.run_policy()[0],self.cmd)
        self.setUp();self.cfg['workerLimit']=1
        self.assertEqual(self.run_policy()[0],self.cmd)
        self.setUp();self.obs['players'][1][2]['E']=[180,0,'COLLECTOR'];self.cfg['recCollectorIncreaseCost']=2
        self.assertEqual(self.run_policy()[1]['reason'],'different_price')

    def test_earlier_base_can_spend_the_wage(self):
        self.obs['players'][0][1]={'earlier':100,'b':self.base}
        self.cmd['earlier']='RECCOLLECTOR'
        self.assertEqual(self.run_policy()[1]['reason'],'not_affordable_or_no_spare_role')

    def test_explicit_incoming_worker_or_enemy_at_harvest(self):
        self.obs['players'][0][2]['friendly']=[127,5,'COLLECTOR'];self.cmd['friendly']='UP'
        self.assertEqual(self.run_policy()[1]['reason'],'base_traffic')
        self.setUp();self.obs['players'][1][2]['enemy']=[115,0,'COLLECTOR']
        self.assertEqual(self.run_policy()[1]['reason'],'no_public_safe_payback')

    def test_appealing_alternative_planter_job_retained(self):
        self.obs['players'][1][3]['tree']=[111,21]
        self.assertEqual(self.run_policy()[1]['reason'],'nearby_enemy_tree')
        self.setUp()
        for p in [97,99,83,128,114,143,129]:
            self.head.known[p]=(249,100.)
        self.assertEqual(self.run_policy()[1]['reason'],'nearby_funded_tree_site')

    def test_expiry_endgame_and_hidden_cell_values_not_used(self):
        self.head.known[self.target]=(243,100.)
        self.obs['carbon'][self.target]=100.
        self.assertEqual(self.run_policy()[1]['reason'],'no_public_safe_payback')
        self.setUp();self.obs['step']=293
        self.assertEqual(self.run_policy()[0],self.cmd)
        self.setUp();self.head.used=True
        self.assertEqual(self.run_policy()[0],self.cmd)

    def test_replaced_or_seen_zero_overwrites_prior_belief(self):
        self.obs['players'][0][2]['me']=[114,0,'PLANTER']
        self.assertEqual(self.run_policy()[1]['reason'],'no_public_safe_payback')
        self.assertEqual(self.head.known[113],(250,0.0))

if __name__=='__main__':unittest.main()
