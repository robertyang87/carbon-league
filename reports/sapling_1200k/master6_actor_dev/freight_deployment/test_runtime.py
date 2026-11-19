"""Constructed public-only decisions; run with python -m unittest discover -s <this folder>."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'freight_response'))
import policy
from runtime import FreightOverlay


class FixedForecast:
    def __init__(self):
        self.calls = []
    def predict(self, root, history):
        assert set(root) == {'player','step','players'}
        assert all(h['player']==root['player'] and h['step'] < root['step'] for h in history)
        self.calls.append((root, history))
        return {'op-c': {'p_deposit_10': .8, 'expected_root_cash_10': 80, 'expected_root_fraction_10': .8}}


def observation(seat=0, step=225):
    own = [50,{'home':66},{'our-c':[2,5,'COLLECTOR']},{}]
    other = [50,{'en-home':0},{'op-c':[32,100,'COLLECTOR']},{}]
    return {'step':step,'player':seat,'players':[own,other] if seat==0 else [other,own]}


class TestOverlay(unittest.TestCase):
    def setUp(self):
        self.forecast = FixedForecast()
        self.overlay = FreightOverlay(self.forecast, policy)

    def test_selector_physical_seats_and_episode_cap(self):
        for s in (0,1):
            obs = observation(s)
            expected = {'our-c':'DOWN'}
            self.assertEqual(self.overlay.consider(obs,{}, {}, lambda u,a:{u:a}),expected)
            self.assertTrue(self.overlay.last['changed'])
            obs['step']+=1
            self.assertEqual(self.overlay.consider(obs,{}, {}, lambda *_:self.fail()), {})
            self.assertEqual(self.overlay.last['reason'],'already_changed')

    def test_immediate_cash_and_uncovered_tree(self):
        obs = observation()
        obs['players'][0][1]['home'] = 2
        self.assertEqual(self.overlay.consider(obs,{}, {}, lambda *_:self.fail()), {})
        self.assertEqual(self.overlay.last['reason'],'forgone_immediate_bank')
        obs = observation(step=226)
        obs['players'][0][3]['our-tree']=[2,10]
        obs['players'][1][2]['op-p']=[3,0,'PLANTER']
        self.assertEqual(self.overlay.consider(obs,{}, {}, lambda *_:self.fail()), {})
        self.assertEqual(self.overlay.last['reason'],'uncovered_contested_tree')

    def test_changed_suffix_including_hire_rejected(self):
        obs = observation()
        self.assertEqual(self.overlay.consider(obs, {}, {}, lambda u,a:{u:a,'home':'RECCOLLECTOR'}),{})
        self.assertEqual(self.overlay.last['reason'],'changed_suffix_or_illegal_prefix')
        obs['step']+=1
        self.assertEqual(self.overlay.consider(obs, {}, {}, lambda *_:None),{})
        self.assertFalse(self.overlay.used)

    def test_team_and_recruit_occupancy(self):
        obs = observation()
        offer = {'worker':'our-c','alternative':'DOWN','landing':17}
        obs['players'][0][2]['our-p']=[17,0,'PLANTER']
        self.assertEqual(self.overlay.veto(obs,{},offer),'reserved_landing')
        del obs['players'][0][2]['our-p']
        obs['players'][0][1]['home']=17
        self.assertEqual(self.overlay.veto(obs,{'home':'RECPLANTER'},offer),'reserved_landing')

    def test_past_snapshot_owned_input_and_hidden_data_isolation(self):
        obs = observation(step=214)
        obs['hidden_carbon'] = [999]*225
        obs['future_commands'] = {'our-c':'NONE'}
        for t in range(214,220):
            obs['step']=t
            obs['players'][0][0]=t
            self.overlay.consider(obs,{'randomSeed':7}, {}, lambda *_:None)
        root, history=self.forecast.calls[-1]
        self.assertEqual([f['step'] for f in history],list(range(214,219)))
        self.assertEqual([f['players'][0][0] for f in history],list(range(214,219)))
        self.assertEqual(set(root),{'player','step','players'})
        obs = observation(1,step=225)
        self.overlay.consider(obs,{}, {}, lambda *_:None)
        self.assertEqual(len(self.forecast.calls[-1][1]),0)

    def test_unsupported_rules_and_no_forecast(self):
        for config in ({'size':11},{'moveCost':.1},{'smeltCost':.1}):
            self.assertEqual(self.overlay.consider(observation(),config,{},lambda *_:self.fail()),{})
            self.assertEqual(self.overlay.last['reason'],'unvalidated_rules')
        self.forecast.predict=lambda obs,past:{}
        self.assertEqual(self.overlay.consider(observation(),{}, {},lambda *_:self.fail()),{})
        self.assertEqual(self.overlay.last['reason'],'no_shared_offer')


if __name__=='__main__':
    unittest.main()
