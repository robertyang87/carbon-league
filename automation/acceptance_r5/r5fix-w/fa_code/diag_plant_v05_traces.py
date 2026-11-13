"""diag_plant_v05_traces: planting profile of the SHIPPED v0.5 from its
recorded traces (no re-simulation). Same metrics as diag_plant_v05.py
where the trace format allows (per-tree income is NOT in traces; the
eval ledger carries the totals). Also reports the example seed2001 P1
endgame stranded cargo (the unfreeze regression check; v0.4: 427.02).
"""

import gzip
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NBR4 = []
for _p in range(225):
    _r, _c = divmod(_p, 15)
    NBR4.append([((_r - 1) % 15) * 15 + _c, ((_r + 1) % 15) * 15 + _c,
                 _r * 15 + (_c - 1) % 15, _r * 15 + (_c + 1) % 15])
BASES = {0: (70, 154), 1: (74, 150)}  # placeholder, read from frames


def dist(a, b):
    ra, ca = divmod(a, 15)
    rb, cb = divmod(b, 15)
    dr = min(abs(ra - rb), 15 - abs(ra - rb))
    dc = min(abs(ca - cb), 15 - abs(ca - cb))
    return dr + dc


def load(fname):
    with gzip.open(os.path.join(HERE, "traces", fname), "rt") as f:
        return json.load(f)


def profile_game(tr, us):
    frames = tr["frames"]
    # bases: workers spawn there; use frame-0 worker positions of the
    # base cells from the trace meta if present, else infer: the two
    # cells where each player's workers first appear.
    births = {}       # tid -> dict
    flips = defaultdict(int)
    prev = {}
    for fr in frames:
        step = fr["step"]
        trees = {tid: tuple(v) for tid, v in fr["trees"].items()}
        for tid, (pos, owner, age) in trees.items():
            if tid not in births and tid not in prev:
                pre = frames[step - 1] if step >= 1 else fr
                carb = pre["carbon"]
                births[tid] = dict(
                    planter=owner, birth=step, pos=pos,
                    cell_carbon=carb[pos],
                    nb4=sum(carb[n] for n in NBR4[pos]),
                    flips=0)
            if tid in prev and prev[tid][1] != owner:
                flips[tid] += 1
        prev = trees
    out = {"us": [], "juzi": []}
    for tid, b in births.items():
        rel = "us" if b["planter"] == us else "juzi"
        b["seized_ever"] = flips.get(tid, 0) > 0
        out[rel].append(b)
    return out


def agg(recs):
    if not recs:
        return {}
    n = len(recs)
    buckets = [0] * 7
    for r in recs:
        buckets[min(r["birth"] // 50, 6)] += 1
    return dict(
        n=n,
        first5=sorted(r["birth"] for r in recs)[:5],
        buckets=buckets,
        nb4_mean=round(sum(r["nb4"] for r in recs) / n, 1),
        cell_mean=round(sum(r["cell_carbon"] for r in recs) / n, 1),
        seized_rate=round(sum(r["seized_ever"] for r in recs) / n, 3),
    )


def main():
    files = sorted(f for f in os.listdir(os.path.join(HERE, "traces"))
                   if f.startswith("policy_v0_5_vs_reference_juzi"))
    all_us, all_juzi = [], []
    for fn in files:
        m = re.search(r"side(\d)", fn)
        us = int(m.group(1))
        tr = load(fn)
        prof = profile_game(tr, us)
        all_us += prof["us"]
        all_juzi += prof["juzi"]
        print(fn, "-> us first5:", sorted(r["birth"] for r in prof["us"])[:5],
              "| juzi first5:", sorted(r["birth"] for r in prof["juzi"])[:5])
    print("\nv0.5 shipped, 8 games pooled:")
    print("  us  :", json.dumps(agg(all_us)))
    print("  juzi:", json.dumps(agg(all_juzi)))
    print("  (v0.4 reference: us n=129 buckets [3,29,41,37,17,2,0] nb4 95.6 "
          "seized 0.558; juzi n=219 nb4 143.1 seized 0.087)")

    tr = load("policy_v0_5_vs_example_seed2001_side1.json.gz")
    last = tr["frames"][-1]
    stranded = defaultdict(float)
    for wid, (pos, cargo, typ) in last["workers"].items():
        stranded[int(wid.split("-")[1])] += cargo
    print("\nexample seed2001 P1 (us=1) endgame stranded by player:",
          dict(stranded), "| us total:", round(stranded[1], 2),
          "(v0.4: 427.02)")


if __name__ == "__main__":
    main()
