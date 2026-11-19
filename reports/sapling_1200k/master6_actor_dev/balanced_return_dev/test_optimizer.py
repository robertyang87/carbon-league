import math
import unittest
import numpy as np
from optimizer import optimize

def root(seed,t=1,margin_sign=1,util_sign=1):
    a=['a','a','b','a','b','b']
    return dict(seed=seed,t=t,seat=0,terrain=f'm{seed}',samples=[
       dict(key=k,u=.5+util_sign*(.15 if k=='a' else -.15),margin=margin_sign*(1500 if k=='a' else -1500),old_p=.5) for k in a])

def dist(root,delta):
    p=1/(1+math.exp(-float(delta[0])))
    g=np.eye(1,15,0).ravel()*p*(1-p)
    return {'a':(p,g),'b':(1-p,-g)}

class TestOptimizer(unittest.TestCase):
    def test_map_macro_and_updates(self):
        r=optimize([root(s) for s in range(4)],dist)
        self.assertEqual(r['decision'],'UPDATE');self.assertEqual(r['trials'][-1]['positive_maps'],4)
        self.assertTrue(all(np.linalg.norm(x['capped_gradient'])<=.15000001 for x in r['map_gradients'].values()))
        self.assertGreater(r['trials'][-1]['macro']['u_gain'],0.)
    def test_shaped_gain_cannot_override_flat_official_u(self):
        r=optimize([root(s,util_sign=0) for s in range(5)],dist)
        self.assertEqual(r['decision'],'KEEP_ZERO');self.assertEqual(len(r['trials']),11)
        self.assertTrue(all('raw_official_u_ppo_gain' in x['rejections'] for x in r['trials']))
    def test_map_with_more_roots_gives_equal_weight(self):
        a=[root(s) for s in range(4)]
        v=optimize(a,dist);w=optimize(a+[root(0,2),root(0,3)],dist)
        np.testing.assert_array_equal(v['delta'],w['delta'])
        self.assertAlmostEqual(v['trials'][-1]['macro']['u_gain'],w['trials'][-1]['macro']['u_gain'])
    def test_undrawn_vanished_mode_rejects(self):
        def disappearing(r,d):
            z=dist(r,d)
            if d[0]!=0:return z
            return {k:(p*.9,g*.9) for k,(p,g) in z.items()}|{'unseen':(.1,np.zeros(15))}
        a=[root(s) for s in range(4)]
        for r in a:
            for s in r['samples']:s['old_p']=.45
        v=optimize(a,disappearing)
        self.assertEqual(v['decision'],'KEEP_ZERO')
        self.assertTrue(all('old_support_lost' in x['rejections'] for x in v['trials']))
    def test_positive_map_count_and_bad_data(self):
        z=optimize([root(i) for i in [1,2]],dist)
        self.assertEqual(z['reason'],'fewer_than_required_positive_maps')
        x=root(1);x['samples'][0]['old_p']=.6
        with self.assertRaises(ValueError):optimize([x],dist)
        with self.assertRaises(ValueError):optimize([root(4),root(4)],dist)

if __name__=='__main__': unittest.main()
