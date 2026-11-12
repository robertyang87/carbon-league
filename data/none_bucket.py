"""none_bucket OURS.py OUR.npz LABELER.npz LABELER_SLOT OPP.py SEED0 N [WORKERS]

Same games as dagger_probe (OURS vs OPP, seats alternate). For every planter decision where WE stayed (NONE) and the
labeler moved ("extra stop"), records our own net's P(NONE) (teacher slot 5 before LATE_STEP 240, 7 after, forced on
our actual joint action) and whether our net's greedy move was NONE at all (if not, the stop came from the guard).
Offline bias sweep: for b in (0.3, 0.6, 1.0), how many extra stops a NONE-logit penalty b would turn into a move (and
into the labeler's move), and how many of the stops the labeler agrees with it would wrongly flip. Nothing is trained.
"""
import json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("BC_W", "64")
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, "env_src", "carbon_challenge-master"))
BIAS = (0.3, 0.6, 1.0)


def play(task):
    import numpy as np
    import eval_v53, cnet2_feat as F, cnet2_np as NP
    from cnet2_agent import load_weights
    from fastsim import FastCarbon
    ours, our_w, lab_w, lab_slot, opp, seed, side = task
    random.seed(eval_v53.rng_seed(seed, side))
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    fa = eval_v53.load_agent_fn(os.path.join(CODE, ours)); fb = eval_v53.load_agent_fn(os.path.join(CODE, opp))
    lab = NP.Policy(load_weights(lab_w)); me = NP.Policy(load_weights(our_w))
    feat = F.Feat(side)
    extra = []; agree_stop = []      # (p_none, net_greedy_none, flips[b], flips_to_label[b]) / (flips[b])
    for step in range(1, 301):
        o_me = sim.obs(side)
        x = feat.observe(o_me)
        a_me = fa(o_me, eval_v53.CFG); a_op = fb(sim.obs(1 - side), eval_v53.CFG)
        y = feat.labels(a_me)
        _, lg, _ = lab.act(x, feat.aux, teacher=lab_slot, forced=y)
        _, mg, _ = me.act(x, feat.aux, teacher=5 if step - 1 < 240 else 7, forced=y)
        for s in range(10):
            if not x["tok_mask"][s] or int(x["tok_kind"][s]) != 1 or int(y[s]) != 0:
                continue
            t = int(lg[s].argmax()); z = mg[s].astype(np.float64)
            legal = z > -1e8
            p = np.exp(z - z[legal].max()) * legal; p /= p.sum()
            fl, fll = [], []
            for b in BIAS:
                z2 = z.copy(); z2[0] -= b; a2 = int(z2.argmax())
                fl.append(int(a2 != 0)); fll.append(int(a2 == t and t != 0))
            if t != 0:
                extra.append((float(p[0]), int(z.argmax() == 0), fl, fll, step))
            else:
                agree_stop.append((float(p[0]), fl))
        sim.step(*((a_me, a_op) if side == 0 else (a_op, a_me)))
    return {"seed": seed, "extra": extra, "agree": agree_stop}


def main():
    ours, our_w, lab_w, lab_slot, opp, s0, n = sys.argv[1:8]
    workers = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    tasks = [(ours, our_w, lab_w, int(lab_slot), opp, s, s % 2) for s in range(int(s0), int(s0) + int(n))]
    with ProcessPoolExecutor(workers) as ex:
        R = list(ex.map(play, tasks))
    E = [e for r in R for e in r["extra"]]; A = [a for r in R for a in r["agree"]]
    print(f"games {len(R)} | extra stops {len(E)} | agreed stops {len(A)}")
    print(f"  extra stops where our NET greedy was not NONE (guard forced the stop): {sum(1 - e[1] for e in E)}")
    for lo, hi in ((0, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01)):
        k = [e for e in E if lo <= e[0] < hi]
        print(f"  P(NONE) {lo:.1f}-{min(hi,1):.1f}: {len(k)} ({100*len(k)/max(1,len(E)):.0f}%)")
    for i, b in enumerate(BIAS):
        print(f"  bias {b}: extra stops turned into a move {sum(e[2][i] for e in E)} (into the labeler's move "
              f"{sum(e[3][i] for e in E)}) | agreed stops wrongly flipped {sum(a[1][i] for a in A)}")
    json.dump({"extra": E, "agree": A}, open(os.path.join(CODE, "..", "data", "none_bucket.json"), "w"))


if __name__ == "__main__":
    main()
