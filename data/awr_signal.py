"""awr_signal -- is there a learnable signal in a whole-game awr_gen corpus, and which way does it point?

For every seed the corpus has one greedy game (k = 0) and K sampled games on the same map against the same opponent.
For each sampled game:  d_reward = its final margin - the greedy game's,  d_x = its macro counts - the greedy game's
(recruits and trees by phase, read from the replays with unit_ledger). Then
  * one line per feature: mean d_x, correlation with d_reward, and the univariate slope (margin per extra unit) +- SE;
  * a joint least-squares fit of d_reward on all d_x (slopes +- SE, R^2).
R^2 is printed next to what pure chance would give for this many features and games (k / (n - 1)); read the
adjusted value. It says how much of the game-to-game difference the macro counts explain at all: the rest is everything else the
sampling changed plus the chaotic divergence of two games that differ in one early move.

  python3 awr_signal.py CORPUS_DIR [CORPUS_DIR ...] [--workers W]
Reads CORPUS/awr.jsonl (or awr_raw.jsonl while the corpus is still being generated)."""
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unit_ledger as UL

PH = (("E", 0, 60), ("M", 60, 240), ("L", 240, 999))
FEATS = [f"{k}_{p}" for k in ("planter", "car", "plant", "seize") for p, _, _ in PH] + ["plant_price50"]


def feats(task):
    corpus, g = task
    d = json.load(gzip.open(os.path.join(corpus, g["file"])))
    units, trees = UL.score_side(d, g["side"])
    x = dict.fromkeys(FEATS, 0.0)
    for u in units:
        k = "planter" if u["kind"] == "PLANTER" else "car"
        x[f"{k}_{next(p for p, lo, hi in PH if lo <= u['step'] < hi)}"] += 1
    for t in trees:
        x[f"{t['kind']}_{next(p for p, lo, hi in PH if lo <= t['step'] < hi)}"] += 1
        if t["kind"] == "plant" and t["price"] >= 50:
            x["plant_price50"] += 1
    return corpus, g["seed"], g["k"], g["reward"], [x[f] for f in FEATS]


def main():
    a = sys.argv[1:]; W = os.cpu_count() or 4
    if "--workers" in a:
        i = a.index("--workers"); W = int(a[i + 1]); del a[i:i + 2]
    tasks = []
    for c in a:
        f = os.path.join(c, "awr.jsonl")
        f = f if os.path.exists(f) else os.path.join(c, "awr_raw.jsonl")
        tasks += [(c, json.loads(l)) for l in open(f) if l.strip()]
    with ProcessPoolExecutor(W) as ex:
        rows = list(ex.map(feats, tasks, chunksize=16))
    for c in a + (["ALL"] if len(a) > 1 else []):
        R = [r for r in rows if c == "ALL" or r[0] == c]
        nm = sum(1 for r in R if r[3] == 0.0)
        if nm:
            print(f"\n!! {os.path.basename(c.rstrip('/'))}: {nm} of {len(R)} games end exactly level (perfect mirror games against an identical "
                  f"deterministic opponent); they are left out, and seeds whose greedy game is one have no baseline")
        R = [r for r in R if r[3] != 0.0]
        base = {(r[0], r[1]): r for r in R if r[2] == 0}
        S = [r for r in R if r[2] > 0 and (r[0], r[1]) in base]
        if len(S) < len(FEATS) + 5:
            print(f"\n### {c}: only {len(S)} sampled games with a greedy partner"); continue
        y = np.array([r[3] - base[(r[0], r[1])][3] for r in S])
        X = np.array([np.array(r[4]) - np.array(base[(r[0], r[1])][4]) for r in S])
        n = len(y)
        print(f"\n### {os.path.basename(c.rstrip('/'))}: {n} sampled games, {len(base)} seeds | d_reward mean {y.mean():+.0f} +- {y.std(ddof=1) / n ** 0.5:.0f}, sd {y.std(ddof=1):.0f}")
        g0 = np.array([r[4] for r in base.values()]).mean(0)
        print("  feature         greedy/game  sampled-greedy   corr   margin per extra unit (alone)")
        for j, f in enumerate(FEATS):
            x = X[:, j]
            if x.std() == 0:
                print(f"  {f:14s} {g0[j]:8.1f}   (never differs)"); continue
            b = np.cov(x, y)[0, 1] / x.var(ddof=1); res = y - y.mean() - b * (x - x.mean())
            se = (res.var(ddof=2) / (x.var(ddof=1) * (n - 1))) ** 0.5
            print(f"  {f:14s} {g0[j]:8.1f}   {x.mean():+8.2f}      {np.corrcoef(x, y)[0, 1]:+.2f}   {b:+7.0f} +- {se:.0f}")
        keep = [j for j in range(len(FEATS)) if X[:, j].std() > 0]
        A = np.c_[np.ones(n), X[:, keep]]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef; s2 = res @ res / (n - A.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(A.T @ A)))
        r2 = 1 - res.var() / y.var(); k = len(keep)
        print(f"  joint fit: R^2 {r2:.3f} (pure chance would give about {k / (n - 1):.3f}; adjusted {1 - (1 - r2) * (n - 1) / (n - k - 1):+.3f}); "
              f"intercept {coef[0]:+.0f} +- {se[0]:.0f}")
        print("   " + "  ".join(f"{FEATS[j]} {coef[i + 1]:+.0f}+-{se[i + 1]:.0f}" for i, j in enumerate(keep)))


if __name__ == "__main__":
    main()
