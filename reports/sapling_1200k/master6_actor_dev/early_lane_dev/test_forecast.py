"""Synthetic observation invariants; no archived matches are accessed by these tests."""
import copy
import math
import unittest

from forecast import NAMES, predict, target_cells


CFG = {'size': 15, 'episodeSteps': 300, 'treeLifespan': 50}


def frame(t, pos, seat=0):
    us = [90, {'base-observer': 125}, {'a': [20, 0, 'COLLECTOR'], 'b': [2, 199, 'COLLECTOR'],
          'p': [45, 0, 'PLANTER']}, {'tree-a': [19, 4]}]
    rival = [222, {'base-owner': 2}, {'target': [pos, 151, 'COLLECTOR'], 'm': [31, 0, 'PLANTER']},
             {'tree-b': [17, 49]}]
    return {'player': seat, 'step': t, 'players': [us, rival] if seat == 0 else [rival, us]}


def shift(obs, dx, dy, replace):
    obs = copy.deepcopy(obs)
    f = lambda p: (p % 15 + dx) % 15 + 15*((p // 15 + dy) % 15)
    for pl in obs['players']:
        for k in pl[1]:
            pl[1][k] = f(pl[1][k])
        for d in pl[2:]:
            for k,v in list(d.items()):
                v[0] = f(v[0])
                if replace:
                    d[k+'x'] = d.pop(k)
    return obs


class ForecastTest(unittest.TestCase):
    def setUp(self):
        self.o = frame(17, 0)
        self.hist = [frame(t, c) for t,c in [(14,12), (15,13),(16,14)]]
        self.w = {'features': list(NAMES), 'horizons': {str(h): [(i-5)/7 for i in range(len(NAMES))] for h in (3,5)}}

    def test_normalized_support_and_features(self):
        for h,n in [(3,26), (5,62)]:
            c,x = target_cells(self.o, self.hist, 'target', h, CFG)
            self.assertEqual((len(c),len(x),len(set(c)),c[-1]), (n,n,n,-1))
            self.assertTrue(all(len(row)==len(NAMES) and all(-1<=v<=1 for v in row) for row in x))
            d = predict(self.w,self.o,self.hist,'target',h,CFG)
            self.assertEqual(list(d),c)
            self.assertAlmostEqual(sum(d.values()),1.0)
            self.assertGreater(min(d.values()),0)
            self.assertTrue(all(math.isfinite(v) for v in d.values()))
            self.assertEqual(x[c.index(1)][NAMES.index('project_last_direction')],1/h)
            self.assertEqual(x[c.index(14)][NAMES.index('project_last_direction')],-1/h)
            self.assertEqual(x[c.index(2)][NAMES.index('position_is_target_home')],1.)
            self.assertEqual(x[-1][NAMES.index('other')],1.)

    def test_observer_symmetry_and_ignored_information(self):
        o = copy.deepcopy(self.o)
        o.update(seed=77, full_carbon=[999]*225, actions={'target':'DOWN'}, arbitrary='junk')
        intended = predict(self.w,self.o,self.hist,'target',5,CFG)
        self.assertEqual(intended,predict(self.w,o,self.hist,'target',5,CFG))
        o['player'] = 1
        o['players'].reverse()
        h = [frame(f['step'], f['players'][1][2]['target'][0],1) for f in self.hist]
        self.assertEqual(intended,predict(self.w,o,h,'target',5,CFG))
        def translate(k):
            return -1 if k==-1 else (k%15+5)%15+15*((k//15+9)%15)
        result = predict(self.w,shift(self.o,5,9,True),[shift(f,5,9,True) for f in self.hist],'targetx',5,CFG)
        for k,p in intended.items():
            self.assertAlmostEqual(p,result[translate(k)])

    def test_seat_time_role_and_weight_contract(self):
        for hist in [[frame(18,0)],[frame(16,0,1)],list(reversed(self.hist))]:
            with self.assertRaises(ValueError):
                target_cells(self.o,hist,'target',3,CFG)
        for uid in ['m','a','missing']:
            with self.assertRaises(ValueError):
                target_cells(self.o,[],uid,3,CFG)
        with self.assertRaises(ValueError):
            target_cells(self.o,self.hist,'target',2,CFG)
        for key, val in [('features',list(reversed(NAMES))), ('horizons',{'3':[float('nan')]*len(NAMES)})]:
            with self.assertRaises(ValueError):
                predict({**self.w,key:val},self.o,self.hist,'target',3,CFG)
        empty = target_cells(self.o,[],'target',3,CFG)
        self.assertEqual(empty, target_cells(self.o,self.hist[:1],'target',3,CFG))
        self.assertEqual(target_cells(self.o,self.hist,'target',3,CFG),target_cells(self.o,self.hist+[self.o],'target',3,CFG))


if __name__=='__main__':
    unittest.main()
