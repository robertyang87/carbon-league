"""Verify the antithetic draws mirror the native categorical draws on the unit interval."""
import numpy as np
import runtime

runtime.np = np
seed = 53281
reference = np.random.RandomState(seed)
u = np.random.RandomState(seed)
anti = runtime._AntitheticResponse(seed)
p = np.array([0.10,0.25,0.25,0.40], dtype=np.float64)
c = np.cumsum(p)
matrix = np.zeros((len(p),len(p)), dtype=np.int32)
for _ in range(20000):
    drawn = u.random_sample()
    a = reference.choice(len(p), p=p)
    b = anti.choice(len(p), p=p)
    assert a == min(len(p)-1,int(np.searchsorted(c,drawn*c[-1],side='right')))
    assert b == min(len(p)-1,int(np.searchsorted(c,(1.0-drawn)*c[-1],side='right')))
    matrix[a,b] += 1
for totals in (matrix.sum(axis=0),matrix.sum(axis=1)):
    assert np.max(np.abs(totals/totals.sum()-p)) < 0.02
print(matrix.tolist())
