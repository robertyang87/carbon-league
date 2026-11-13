"""diag_seize_risk_v06b: follow-up cuts on the v06 risk data.

1. ep0/epm1: seize rate by distance to nearest enemy planter AT the
   birth frame / one frame before (the policy-visible quantity).
2. Site probe split by version (v0.4 vs v0.5) -- the pooled probe
   mixes v0.4's late start into the early bucket.
3. Late-window state snapshots (steps 200/225/250): planters alive,
   own alive trees, cash -- us vs juzi, v0.5 games only. Why do we
   plant 4 trees at step 200-249 while juzi plants 27?
"""

import gzip
import json
import os
import re

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


def ep_tables():
    """Seize rate by enemy-planter distance at birth frame (ep0) and
    one frame earlier (epm1) -- what the policy sees when deciding."""
    files = sorted(f for f in os.listdir(os.path.join(HERE, "traces"))
                   if re.match(r"policy_v0_[45]_vs_reference_juzi", f))
    recs = {"ours": [], "juzi": []}
    for fn in files:
        us = int(re.search(r"side(\d)", fn).group(1))
        tr = load(fn)
        frames = tr["frames"]
        births = {}
        owner_hist = {}
        for fi, fr in enumerate(frames):
            for tid, (pos, owner, age) in fr["trees"].items():
                owner_hist.setdefault(tid, {})[fi] = owner
                if tid not in births:
                    births[tid] = (fi, pos, owner)
        def ep_at(fi, enemy):
            if fi < 0 or fi >= len(frames):
                return 99
            ds = [dist_cur for dist_cur in
                  [dist(pos_c, w[0]) for wid, w in frames[fi]["workers"].items()
                   if int(wid.split("-")[1]) == enemy and w[2] == "P"]
                  ]
            return min(ds) if ds else 99
        for tid, (bfi, pos, planter) in births.items():
            enemy = 1 - planter
            pos_c = pos
            seized = any(o == enemy for o in owner_hist[tid].values())
            fsa = None
            for fi in sorted(owner_hist[tid]):
                if owner_hist[tid][fi] == enemy:
                    fsa = fi - bfi
                    break
            recs["ours" if planter == us else "juzi"].append(
                dict(ep0=ep_at(bfi, enemy), epm1=ep_at(bfi - 1, enemy),
                     seized=seized, fsa=fsa))
    for name, sel in recs.items():
        print(f"\n== ep tables: {name} (n={len(sel)})")
        for key in ("epm1", "ep0"):
            print(f"  {key}:")
            for lo, hi, lab in [(0, 2, "0-2"), (3, 4, "3-4"), (5, 6, "5-6"),
                                (7, 98, "7+")]:
                s = [r for r in sel if lo <= r[key] <= hi]
                if not s:
                    print(f"    {lab}: n=0")
                    continue
                sr = sum(r["seized"] for r in s) / len(s)
                ages = sorted(r["fsa"] for r in s if r["fsa"] is not None)
                med = ages[len(ages) // 2] if ages else None
                print(f"    {lab}: n={len(s):4d} seize_rate={sr:.3f} "
                      f"median_fsa={med}")


def main():
    ep_tables()
    data = json.load(open(os.path.join(HERE, "diag_seize_risk_v06.json")))
    sites = data["sites"]
    # --- 2. site probe by version: recompute from traces with version tag
    for ver in ("policy_v0_4", "policy_v0_5"):
        files = sorted(f for f in os.listdir(os.path.join(HERE, "traces"))
                       if f.startswith(ver + "_vs_reference_juzi"))
        print(f"\n== site probe {ver}")
        # reuse diag_seize_risk_v06.site_probe via import
        import diag_seize_risk_v06 as d6
        all_s = []
        for fn in files:
            us = int(re.search(r"side(\d)", fn).group(1))
            all_s += d6.site_probe(load(fn), us)
        for name, sel_all in (("OURS", [s for s in all_s if s["ours"]]),
                              ("JUZI", [s for s in all_s if not s["ours"]])):
            print(f"  {name} (n={len(sel_all)})")
            for blo, bhi, blab in [(0, 49, "0-49"), (50, 99, "50-99"),
                                   (100, 149, "100-149"), (150, 300, "150+")]:
                sel = [s for s in sel_all if blo <= s["birth"] <= bhi]
                if not sel:
                    continue
                ch = sum(s["chosen"] for s in sel) / len(sel)
                bn = sum(s["best_near"] for s in sel) / len(sel)
                print(f"    birth {blab:>7}: n={len(sel):4d} chosen={ch:6.1f} "
                      f"best_near8={bn:6.1f} ratio={ch/bn:.2f}")
    # --- 3. late-window snapshots, v0.5 games
    files = sorted(f for f in os.listdir(os.path.join(HERE, "traces"))
                   if f.startswith("policy_v0_5_vs_reference_juzi"))
    print("\n== late-window snapshots (v0.5 games, pooled means)")
    for step in (150, 200, 225, 250):
        agg = {0: [], 1: []}
        for fn in files:
            tr = load(fn)
            fr = tr["frames"][step]
            for p in (0, 1):
                nplan = sum(1 for wid, w in fr["workers"].items()
                            if int(wid.split("-")[1]) == p and w[2] == "P")
                ncoll = sum(1 for wid, w in fr["workers"].items()
                            if int(wid.split("-")[1]) == p and w[2] == "C")
                ntree = sum(1 for t in fr["trees"].values() if t[1] == p)
                agg[p].append((nplan, ncoll, ntree, fr["cash"][p]))
        us = int(re.search(r"side(\d)", files[0]).group(1))
        # NOTE: policy side differs per file; redo per file properly below
        print(f"  step {step}: (per-file side handling in v2 below)")
        break
    for step in (150, 200, 225, 250):
        rows = {"us": [], "juzi": []}
        for fn in files:
            us = int(re.search(r"side(\d)", fn).group(1))
            tr = load(fn)
            fr = tr["frames"][step]
            for p, name in ((us, "us"), (1 - us, "juzi")):
                nplan = sum(1 for wid, w in fr["workers"].items()
                            if int(wid.split("-")[1]) == p and w[2] == "P")
                ncoll = sum(1 for wid, w in fr["workers"].items()
                            if int(wid.split("-")[1]) == p and w[2] == "C")
                ntree = sum(1 for t in fr["trees"].values() if t[1] == p)
                rows[name].append((nplan, ncoll, ntree, fr["cash"][p]))
        for name in ("us", "juzi"):
            r = rows[name]
            m = [sum(x[i] for x in r) / len(r) for i in range(4)]
            print(f"  step {step} {name:>5}: planters={m[0]:.2f} "
                  f"collectors={m[1]:.2f} trees={m[2]:.2f} cash={m[3]:.0f}")


if __name__ == "__main__":
    main()
