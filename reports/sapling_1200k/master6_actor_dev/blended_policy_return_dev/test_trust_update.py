import math
import unittest
import numpy as np
from trust_update import train

def root(seed,seat=0,t=20,reverse=False):
    return dict(seed=seed,seat=seat,t=t,
                samples=[dict(key=a,u=float((a=='a')!=reverse),old_p=.5) for a in ['a','a','b','a','b','b','b','a']])

def regular(r,d):
    p=1/(1+math.exp(-float(d[0])))
    g=np.array([p*(1-p),0.,0.,0.,0.])
    return {'a':(p,g),'b':(1-p,-g)}

class UpdateTests(unittest.TestCase):
    def test_balanced_update_halves_and_retains_duplicates(self):
        q=train([root(50),root(51)],regular)
        self.assertEqual(q['decision'],'UPDATE')
        self.assertEqual(len(q['trials']),2)
        self.assertEqual(q['delta'],[0.25,0.,0.,0.,0.])
        self.assertGreater(q['trials'][0]['max_root_kl'],.02)
        self.assertEqual(q['samples'],16)
        self.assertGreater(q['trials'][-1]['map_weighted_ppo_gain'],0)
    def test_no_update_for_equally_weighted_opposing_maps(self):
        a=[root(1,t=t) for t in [1,2,3]]
        a.append(root(2,reverse=True))
        result=train(a,regular)
        self.assertEqual(result['decision'],'KEEP_ZERO')
        self.assertEqual(result['delta'],[0.]*5)
        self.assertEqual(result['roots'],4)
    def test_reject_unsampled_support_discontinuity(self):
        def dist(r,d):
            if d[0]!=0:
                return {'a':(.5,np.zeros(5)),'b':(.5,np.zeros(5))}
            g=np.array([.25,0.,0.,0.,0.])
            return {'a':(.45,g),'b':(.45,-g),'never-drawn':(.10,np.zeros(5))}
        sample=root(1)
        for s in sample['samples']:s['old_p']=.45
        result=train([sample],dist)
        self.assertEqual(result['decision'],'KEEP_ZERO')
        self.assertEqual(len(result['trials']),7)
        self.assertTrue(all(it['any_old_support_lost'] for it in result['trials']))
    def test_constant_rewards_and_bad_propensity(self):
        r=root(56)
        for s in r['samples']:s['u']=.3
        self.assertEqual(train([r],regular)['reason'],'zero_map_balanced_gradient')
        r['samples'][0]['old_p']=.4
        with self.assertRaises(ValueError):train([r],regular)
    def test_inverted_new_objective_refuses_all(self):
        def wrong(r,d):
            if d[0]==0:return regular(r,d)
            return regular(r,-d)
        result=train([root(1)],wrong)
        self.assertEqual(result['decision'],'KEEP_ZERO')
        self.assertTrue(all('insufficient_sample_ppo_gain' in tr['rejections'] for tr in result['trials']))
    def test_duplicate_roots_and_wrong_draw_count(self):
        with self.assertRaises(ValueError):train([root(92),root(92)],regular)
        r=root(92)
        r['samples'].pop()
        with self.assertRaises(ValueError):train([r],regular)

if __name__=='__main__':unittest.main()
