"""diag_seize_risk_v06c: policy-computable risk features.

ep3: min distance from the tree cell to an enemy planter over frames
birth..birth+3 -- the trace analogue of the opponent model's
mode_path(steps=3) predicted-path distance available in-policy.

pheon: enemy-PLANTER heat at the tree cell at birth time, replicating
a slow-decay accumulation (0.99/step, +1.0 at planter pos, +0.4 at
4-neighbours) run forward over the trace -- the in-policy computable
"their planter territory" signal. Reported as max over the cell and
its 4-neighbours (zone pressure), bucketed.

Also: early-bucket (birth<50) distance-to-own-base, ours vs juzi
(en-route stopping check for the step-3 fix).
"""

import gzip
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 15
N = SIZE * SIZE
NBR4 = []
for _p in range(N):
    _r, _c = divmod(_p, SIZE)
    NBR4.append([((_r - 1) % SIZE) * SIZE + _c, ((_r + 1) % SIZE) * SIZE + _c,
                 _r * SIZE + (_c - 1) % SIZE, _r * SIZE + (_c + 1) % SIZE])


def dist(a, b):
    ra, ca = divmod(a, SIZE)
    rb, cb = divmod(b, SIZE)
    return min(abs(ra - rb), SIZE - abs(ra - rb)) + \
        min(abs(ca - cb), SIZE - abs(ca - cb))


def load(fname):
    with gzip.open(os.path.join(HERE, "traces", fname), "rt") as f:
        return json.load(f)


def table(sel, key, buckets):
    print(f"  {key}:")
    for lo, hi, lab in buckets:
        s = [r for r in sel if lo <= r[key] <= hi]
        if not s:
            print(f"    {lab}: n=0")
            continue
        sr = sum(r["seized"] for r in s) / len(s)
        ages = sorted(r["fsa"] for r in s if r["fsa"] is not None)
        med = ages[len(ages) // 2] if ages else None
        print(f"    {lab}: n={len(s):4d} seize_rate={sr:.3f} median_fsa={med}")


def main():
    files = sorted(f for f in os.listdir(os.path.join(HERE, "traces"))
                   if re.match(r"policy_v0_[45]_vs_reference_juzi", f))
    recs = {"ours": [], "juzi": []}
    early_dbase = {"ours": [], "juzi": []}
    for fn in files:
        us = int(re.search(r"side(\d)", fn).group(1))
        tr = load(fn)
        frames = tr["frames"]
        bases = {int(p): [pos for _b, pos in v]
                 for p, v in tr["meta"]["bases"].items()}
        # planter heat per enemy perspective: heat_for[p] = heat built
        # from player (1-p)'s planters, as p's policy would see it.
        heat = {0: [0.0] * N, 1: [0.0] * N}
        heat_snap = {0: [], 1: []}   # per frame, snapshot BEFORE update
        for fi, fr in enumerate(frames):
            for p in (0, 1):
                heat_snap[p].append(list(heat[p]))
                for c in range(N):
                    heat[p][c] *= 0.99
                for wid, w in fr["workers"].items():
                    if int(wid.split("-")[1]) == 1 - p and w[2] == "P":
                        heat[p][w[0]] += 1.0
                        for nb in NBR4[w[0]]:
                            heat[p][nb] += 0.4
        births = {}
        owner_hist = {}
        for fi, fr in enumerate(frames):
            for tid, (pos, owner, age) in fr["trees"].items():
                owner_hist.setdefault(tid, {})[fi] = owner
                if tid not in births:
                    births[tid] = (fi, pos, owner)
        for tid, (bfi, pos, planter) in births.items():
            enemy = 1 - planter
            seized = False
            fsa = None
            for fi in sorted(owner_hist[tid]):
                if owner_hist[tid][fi] == enemy:
                    seized = True
                    fsa = fi - bfi
                    break
            ep3 = 99
            for fi in range(bfi, min(bfi + 4, len(frames))):
                for wid, w in frames[fi]["workers"].items():
                    if int(wid.split("-")[1]) == enemy and w[2] == "P":
                        ep3 = min(ep3, dist(pos, w[0]))
            hs = heat_snap[planter][bfi]
            pheon = max([hs[pos]] + [hs[n] for n in NBR4[pos]])
            name = "ours" if planter == us else "juzi"
            recs[name].append(dict(ep3=ep3, pheon=round(pheon, 2),
                                   seized=seized, fsa=fsa,
                                   birth=frames[bfi]["step"]))
            if frames[bfi]["step"] < 50:
                early_dbase[name].append(
                    min(dist(pos, b) for b in bases[planter]))
    for name, sel in recs.items():
        print(f"\n== {name} (n={len(sel)})")
        table(sel, "ep3", [(0, 2, "0-2"), (3, 4, "3-4"), (5, 6, "5-6"),
                           (7, 98, "7+")])
        table(sel, "pheon", [(0.0, 0.05, "~0"), (0.05, 0.8, ".05-.8"),
                              (0.8, 2.0, ".8-2"), (2.0, 99, "2+")])
        d = early_dbase[name]
        print(f"  early(birth<50) dist-to-own-base: n={len(d)} "
              f"mean={sum(d)/max(1,len(d)):.2f}")


if __name__ == "__main__":
    main()
