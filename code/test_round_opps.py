"""Regression fixture for --round-opps (per-iteration opponent subset) in rl2_ppo_train.

The inference server runs one forward per distinct model per step; with 13+ league models the
batches fall to ~10 positions per forward. --round-opps K lets one iteration play against at
most K non-SELF members. Pinned here:
- sample(rng) without keys consumes the RNG exactly as the legacy formula (default path unchanged);
- round_subset returns SELF plus exactly K other active members, never a retired one, and the
  whole active set when K is 0 or not smaller than the member count;
- sampling inside a subset only returns subset members;
- SELF keeps exactly its full-pool probability; over many iterations every other member's
  share of games stays within 2.5 points of its full-pool share (nobody is starved).

Run: python test_round_opps.py  (from code/, or an acceptance tree)
"""
import os
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "rl2_ppo_train.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))
sys.path.insert(0, CODE)

import numpy as np
import rl2_ppo_train as PT

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def league(td):
    pool = [[f"SNAP:cnet2:/w/m{i}.npz#T2", pr] for i, pr in enumerate((1.1, 0.55, 0.55, 0.5, 0.45, 0.45, 0.4, 0.3))]
    lg = PT.League(os.path.join(td, "league.json"), pool, 0.5, 1.0)
    lg.tab["SELF"] = {"prior": 0.5, "n": 0, "wr": 0.5, "margin": 0.0}
    lg.tab["SNAP:cnet2:/w/old.npz"] = {"prior": 0.0, "n": 9, "wr": 0.9, "margin": 0.0, "snap": True}   # retired
    for i, wr in enumerate((0.85, 0.78, 0.72, 0.49, 0.78, 0.78, 0.70, 0.68)):
        lg.tab[pool[i][0]]["wr"] = wr
    return lg


def legacy_sample(lg, rng):
    keys = [k for k, v in lg.tab.items() if v["prior"] > 0]
    w = np.array([lg.tab[k]["prior"] * (1.0 - lg.tab[k]["wr"] + 0.05) ** lg.pfsp for k in keys])
    return keys[rng.choice(len(keys), p=w / w.sum())]


def main():
    with tempfile.TemporaryDirectory() as td:
        lg = league(td)
        a = [lg.sample(np.random.default_rng(5)) for _ in range(1)]
        r1, r2 = np.random.default_rng(11), np.random.default_rng(11)
        check("default sample() draws exactly the legacy sequence",
              [lg.sample(r1) for _ in range(500)] == [legacy_sample(lg, r2) for _ in range(500)] and a)
        active = [k for k, v in lg.tab.items() if v["prior"] > 0]
        rng = np.random.default_rng(3)
        sub, sp = lg.round_subset(rng, 3)
        check("subset = SELF + exactly 3 others", "SELF" in sub and len(sub) == 4 and len(set(sub)) == 4)
        check("subset probabilities sum to 1 and align with the keys", len(sp) == 4 and abs(sp.sum() - 1) < 1e-12)
        check("subset never contains a retired member", all(lg.tab[k]["prior"] > 0 for k in sub))
        full_w0 = lg._weights(active)
        k0, p0 = lg.round_subset(rng, 0)
        check("K = 0 returns the whole active set with the full-pool probabilities",
              k0 == active and np.allclose(p0, full_w0 / full_w0.sum()))
        check("K >= member count returns the whole active set", lg.round_subset(rng, 8)[0] == active
              and lg.round_subset(rng, 50)[0] == active)
        check("sampling inside a subset stays inside it", {lg.sample(rng, sub, sp) for _ in range(300)} <= set(sub))

        full_w = lg._weights(active); full_p = dict(zip(active, full_w / full_w.sum()))
        counts = dict.fromkeys(active, 0); n_it, per = 4000, 40
        for _ in range(n_it):
            keys, probs = lg.round_subset(rng, 3)
            for _g in range(per):
                counts[lg.sample(rng, keys, probs)] += 1
        share = {k: counts[k] / (n_it * per) for k in active}
        nonself = [k for k in active if k != "SELF"]
        check("every member is played", all(counts[k] > 0 for k in active))
        # Within the non-SELF members the long-run order of shares follows the order of weights
        order_w = sorted(nonself, key=lambda k: -full_p[k]); order_s = sorted(nonself, key=lambda k: -share[k])
        check("heaviest and lightest members keep their rank", order_w[0] == order_s[0] and order_w[-1] == order_s[-1])
        print("  member weight(full pool) vs long-run share under --round-opps 3:")
        for k in active:
            print(f"    {k.split('/')[-1]:14s} {full_p[k]:.3f}  {share[k]:.3f}")
        check("SELF keeps its full-pool share", abs(share["SELF"] - full_p["SELF"]) < 0.01)
        check("every other member stays within 2.5 points of its full-pool share",
              all(abs(share[k] - full_p[k]) < 0.025 for k in nonself))

    src = open(os.path.join(CODE, "rl2_ppo_train.py")).read()
    check("flag defaults to 0 (off)", 'a("--round-opps", type=int, default=0' in src)
    print(f"{sum(RES)}/{len(RES)} passed", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
