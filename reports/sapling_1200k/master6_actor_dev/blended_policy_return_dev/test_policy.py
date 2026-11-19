"""Construct a protected downstream recruit that forces two separate proposal intervals onto one physical order."""
from types import SimpleNamespace
import unittest
import numpy as np
from policy import Root,ZERO,key


class ExecutedMassTests(unittest.TestCase):
    def test_recruit_fallback_merges_subintervals_and_gradients(self):
        class ArtificialRoot(Root):
            def decode(self,residual,slot=None,forced=None):
                raw=np.array([.3,.7,0,0,0]) if residual else np.array([.8,.2,0,0,0])
                p=raw*np.exp(.25*self.m._NETS[1].w['whead.bias'])
                p/=p.sum()
                a=int(np.argmax(p)) if forced is None else forced
                y=np.zeros(11,dtype=int)
                y[0]=a
                if residual and a==1:
                    y[10]=1
                return y,{0:p}
        root=object.__new__(ArtificialRoot)
        root.m=SimpleNamespace(_initial_bias=ZERO.copy(),_NETS=[None,SimpleNamespace(w={'whead.bias':ZERO.copy()})],
                               _return_conflicts=lambda x,labels:set())
        root.x={'tok_kind':np.zeros(11,int)}
        root.f=SimpleNamespace(to_actions=lambda y:{'worker':'RIGHT'} if y[0] else {})
        law=root.distribution(0,ZERO)
        self.assertEqual(set(law),{key({}),key({'worker':'RIGHT'})})
        p,g=law[key({})]
        # U from 0 to .3 returns the residual proposal. U from .3 to .8 rolls back to the same base joint.
        self.assertAlmostEqual(p,.8)
        self.assertTrue(np.allclose(g,[.04,-.04,0,0,0]))
        h=ZERO.copy();h[0]=.01
        a=root.distribution(0,h)[key({})][0];b=root.distribution(0,-h)[key({})][0]
        self.assertAlmostEqual((a-b)/.02,float(g[0]),places=5)
