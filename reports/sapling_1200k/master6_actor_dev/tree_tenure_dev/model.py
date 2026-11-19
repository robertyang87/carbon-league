"""Small portable tree-ensemble predictor; inference needs only NumPy."""
from pathlib import Path
import numpy as np
from dataset import FEATURES

HERE = Path(__file__).resolve().parent


class Predictor:
    def __init__(self, weights):
        self.w = weights

    def predict_array(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError('finite feature matrix in registered order required')
        out = np.zeros((len(x), 2))
        for q in (0, 1):
            out[:,q] = self.w['base'][q]
            for st, en in self.w['offset'][q]:
                ix = np.full(len(x), st)
                while True:
                    mask = self.w['left'][ix] >= 0
                    if not mask.any():
                        break
                    z = ix[mask]
                    goleft = x[mask, self.w['feature'][z]] <= self.w['threshold'][z]
                    ix[mask] = np.where(goleft, self.w['left'][z], self.w['right'][z])
                out[:,q] += self.w['value'][ix]
        kind = x[:,0].astype(int)
        if not (np.isin(x[:,0], [0.,1.])).all():
            raise ValueError('unknown purchase kind')
        for q in (0,1):
            a = self.w['calib'][q,kind]
            out[:,q] = np.maximum(0., a[:,0] + a[:,1]*out[:,q])
        out[:,0] = np.minimum(out[:,0], x[:, FEATURES.index('remaining_turns')])
        return out

    def predict(self, rows):
        x=[]
        for r in rows:
            if r['kind'] not in ('plant','seize'):
                raise ValueError('expected plant or seize')
            v=[r[k] for k in FEATURES]
            v[0]=int(r['kind']=='seize'); x.append(v)
        return self.predict_array(x)


def load():
    with np.load(HERE/'weights.npz',allow_pickle=False) as z:
        return Predictor({k:z[k] for k in z.files})
