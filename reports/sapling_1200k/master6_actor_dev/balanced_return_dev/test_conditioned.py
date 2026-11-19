from pathlib import Path
import tempfile
import unittest
import numpy as np
from conditioned import Conditioning

class TestConditioned(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(15)
        self.h=self.rng.normal(size=(30,96)).astype(np.float32)
        self.targets=np.column_stack([np.arange(30)*3.,np.arange(30)%15,np.arange(30)%2])
        self.maps=['a']*20+['b']*10
        self.model=Conditioning.fit(self.h,self.targets,self.maps)
    def test_map_balance_and_terrain_duplication(self):
        np.testing.assert_allclose(self.model.hidden_mean,(self.h[:20].astype(float).mean(0)+self.h[20:].astype(float).mean(0))/2)
        other=Conditioning.fit(np.concatenate([self.h,self.h[:20]]),np.concatenate([self.targets,self.targets[:20]]),self.maps+['a']*20)
        np.testing.assert_allclose(other.projection,self.model.projection,atol=1e-12,rtol=0)
    def test_actual_f32_weights_match_dense_factors_and_zero_is_identity(self):
        w=self.rng.normal(size=(5,96)).astype(np.float32);b=self.rng.normal(size=5).astype(np.float32)
        s={'whead.weight':w,'whead.bias':b,'other':np.array([3,9])};delta=self.rng.normal(scale=.2,size=15)
        out=self.model.inject(s,delta)
        actual=self.h@out['whead.weight'].T+out['whead.bias']
        expected=self.h@w.T+b+self.model.head_increment(self.h,delta)
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-6)
        self.assertTrue(np.array_equal(out['other'],s['other']))
        self.assertTrue(np.array_equal(self.model.inject(s,np.zeros(15))['whead.weight'],w))
    def test_constant_columns_and_save(self):
        z=self.targets.copy();z[:,2]=0
        m=Conditioning.fit(self.h,z,self.maps)
        self.assertTrue(np.all(m.projection[:,2]==0))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'conditioning.json';m.write(p)
            np.testing.assert_array_equal(Conditioning.load(p).features(self.h),m.features(self.h))
    def test_invalid_public_inputs(self):
        z=self.targets.copy();z[3,2]=.5
        with self.assertRaises(ValueError):Conditioning.fit(self.h,z,self.maps)
        with self.assertRaises(ValueError):self.model.head_increment(self.h,np.ones(5))
if __name__=='__main__':unittest.main()
