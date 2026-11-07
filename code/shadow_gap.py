"""shadow_gap -- where does a BC shadow disagree with its teacher? Val games only.
  python3 shadow_gap.py DATA.npz WEIGHTS.npz [WEIGHTS2.npz ...]"""
import json, sys, numpy as np
import nn_agent as A
from nn_search2 import FastPolicy
data = sys.argv[1]; ws = sys.argv[2:]
D = np.load(data, mmap_mode='r'); games = json.load(open(data.replace('.npz', '_games.json')))['games']
val = np.array([g['val'] for g in games])[D['gid'][:]]
idx = np.nonzero(val)[0]
X = D['X']; Sc = D['Sc']; y = D['y'][:]; kind = D['kind'][:]; step = D['step'][:]
# contexts (collectors only), egocentric centre (7,7)
yy, xx = np.mgrid[0:15, 0:15]; dist = np.abs(yy - 7) + np.abs(xx - 7)
win2 = (dist <= 2) & (dist > 0)
Xv = np.asarray(X[idx]); Sv = np.asarray(Sc[idx]).astype(np.float32)
own = Xv[:, 3, 7, 7] / 127.0 * 200
encol = (Xv[:, 5] > 0) & win2
encarry = np.where(encol, Xv[:, 6] / 127.0 * 200, -1).max(axis=(1, 2))
near_en = encol.any(axis=(1, 2))
atbase_dist = Xv[:, 12, 7, 7] / 127.0 * 14
k = kind[idx]; st = step[idx]; yv = y[idx]
ctx = np.full(len(idx), 'other', dtype=object)
c = k == 0
ctx[c & ~near_en & (own < 1)] = 'C empty, no enemy'
ctx[c & ~near_en & (own >= 1)] = 'C loaded, no enemy'
ctx[c & near_en & (own >= 1) & (encarry <= own)] = 'C loaded, threat'
ctx[c & near_en & (encarry > own) & (encarry > 0)] = 'C hunt chance'
ctx[c & near_en & (ctx == 'other')] = 'C near enemy, other'
ctx[k == 1] = 'planter'; ctx[k == 2] = 'base'
MASK = {0: (0, 5), 1: (0, 5), 2: (5, 8)}
for w in ws:
    pol = FastPolicy(A.load_weights(w)); pred = np.empty(len(idx), np.int64)
    for b in range(0, len(idx), 4096):
        lg = pol._forward(Xv[b:b+4096].astype(np.float32) / 127.0, Sv[b:b+4096])
        kk = k[b:b+4096]; m = np.full_like(lg, -1e9)
        for kd, (lo, hi) in MASK.items():
            r = kk == kd; m[r, lo:hi] = lg[r, lo:hi]
        pred[b:b+4096] = m.argmax(1)
    ok = pred == yv
    print(f"\n== {w.split('/')[-1]}  all {ok.mean():.3f}  (n={len(idx)})")
    for name in sorted(set(ctx)):
        r = ctx == name
        print(f"  {name:22s} n={r.sum():6d}  acc {ok[r].mean():.3f}")
    for lo, hi in ((0, 30), (30, 100), (100, 200), (200, 270), (270, 300)):
        r = (st >= lo) & (st < hi)
        print(f"  step {lo:3d}-{hi:3d}          n={r.sum():6d}  acc {ok[r].mean():.3f}")
    for kd, nm in ((2, 'base'),):
        r = k == kd
        cm = np.zeros((3, 3), int)
        for a, b_ in zip(yv[r] - 5, pred[r] - 5): cm[a, b_] += 1
        print("  base confusion (rows teacher NONE/RECC/RECP, cols pred):", cm.tolist())
