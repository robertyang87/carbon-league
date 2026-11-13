"""cnet2_search_bench -- per-step costs: cnet2 encode/decode (b128, s64), Feat.observe, FastCarbon step/load."""
import os, sys, time, json
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cnet2_feat as F, cnet2_np as NP, cnet2_agent as CA
from fastsim import FastCarbon
import eval_v53
W = os.path.join(HERE, "..", "data", "weights")
pols = {n: NP.Policy(CA.load_weights(os.path.join(W, n + ".npz"))) for n in ("cnet2_b128_q8", "cnet2_s64")}
sim = FastCarbon(eval_v53.CFG).reset(seed=990001)
ags = [CA.make_agent(os.path.join(W, "cnet2_b128_q8.npz")) for _ in range(2)]
obs_l = []; t_sim = 0.0
for step in range(1, 301):
    o = [sim.obs(p) for p in range(2)]
    if step % 10 == 0: obs_l.append(o[0])
    a = [ags[p](o[p]) for p in range(2)]
    ts = time.perf_counter(); sim.step(a[0], a[1]); t_sim += time.perf_counter() - ts
print("fastsim step ms", 1000 * t_sim / 300)
st = sim.get_state()
# load_observation cost
full = sim.full_observation(0); ts = time.perf_counter()
for _ in range(100): FastCarbon(eval_v53.CFG).load_observation(full)
print("fastsim new+load ms", (time.perf_counter() - ts) * 10)
s2 = FastCarbon(eval_v53.CFG); ts = time.perf_counter()
for _ in range(100): s2.load_observation(full)
print("fastsim reload ms", (time.perf_counter() - ts) * 10)
for name, pol in pols.items():
    f = F.Feat(0); te = td = tf = 0.0; nu = 0
    for o in obs_l:
        ts = time.perf_counter(); x = f.observe(o); tf += time.perf_counter() - ts
        ts = time.perf_counter(); pol.encode(x, 1); te += time.perf_counter() - ts
        ts = time.perf_counter(); pol.act(x, f.aux, teacher=1); td += time.perf_counter() - ts
        nu += int(x["tok_mask"][:12].sum())
    n = len(obs_l)
    print(name, "observe ms %.2f encode ms %.2f act(enc+dec) ms %.2f units/step %.1f" % (1000*tf/n, 1000*te/n, 1000*td/n, nu/n))
