"""cnet2_search_test -- parity of batched encode/decode vs NP.Policy.act; batched timing."""
import os, sys, time
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cnet2_feat as F, cnet2_np as NP, cnet2_search as S
from fastsim import FastCarbon
import eval_v53
W = os.path.join(HERE, "..", "data", "weights")
pols = {n: NP.Policy(S.load_weights(os.path.join(W, n + ".npz"))) for n in ("cnet2_b128_q8", "cnet2_s64")}
sim = FastCarbon(eval_v53.CFG).reset(seed=990003)
fe = [F.Feat(0), F.Feat(1)]
xs, auxs = [], []
p0 = pols["cnet2_b128_q8"]
for step in range(1, 301):
    a = []
    for p in range(2):
        x = fe[p].observe(sim.obs(p)); y, _, _ = p0.act(x, fe[p].aux, teacher=1)
        a.append(fe[p].to_actions(y))
        if step % 6 == 0: xs.append(x); auxs.append(fe[p].aux)
    sim.step(a[0], a[1])
print("boards", len(xs))
for name, pol in pols.items():
    agree = tot = 0
    for x, ax in zip(xs, auxs):
        y, _, _ = pol.act(x, ax, teacher=1)
        yb = S.act_b(pol, [x], [ax], [1])[0]
        m = y >= 0
        agree += int((y[m] == yb[m]).sum()); tot += int(m.sum())
    yb = S.act_b(pol, xs, auxs, [1] * len(xs))
    agree2 = sum(int((pol.act(x, ax, teacher=1)[0] == yb[i]).sum()) for i, (x, ax) in enumerate(zip(xs, auxs))) - 12 * len(xs) + tot
    print(name, "B=1 agree %d/%d  B=all agree %d/%d" % (agree, tot, agree2, tot))
    for B in (1, 4, 8, 12, 16, 24, 32):
        ts = time.perf_counter(); R = 3
        for r in range(R): S.act_b(pol, xs[:B], auxs[:B], [1] * B)
        dt = (time.perf_counter() - ts) / R
        print("  B=%2d  %.1f ms total  %.2f ms/board" % (B, 1000 * dt, 1000 * dt / B))
