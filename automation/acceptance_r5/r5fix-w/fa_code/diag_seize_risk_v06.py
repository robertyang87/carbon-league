"""diag_seize_risk_v06: seized-risk model for our trees, from the 16
recorded v0.4 + v0.5 vs-juzi traces (no re-simulation).

For every tree planted in those games (both sides; "our" trees are the
policy's), extract:
  outcome   seized_ever (owner flipped to the enemy at least once),
            first-seize age, flip count, censoring flag (birth+50>300)
  features  (all relative to the planter's enemy)
    birth        planting step
    nb4          true 4-neighbour carbon sum at the birth cell,
                 frame birth-1 (same yardstick as diag_plant_v05)
    d_opp_base   torus distance to nearest enemy base
    d_own_base   torus distance to nearest own base
    hot_dist     distance to the enemy-planter hot zone (cells where an
                 enemy planter stood >= HOT_MIN frames all game)
    exposure3    frames in the owned window (birth .. first flip/death)
                 with an enemy planter within distance <= 3
    own_cover    share of owned-window frames with an own unit within
                 distance <= 3
    ep5          min distance to an enemy planter over frames
                 birth..birth+5
    arr1_age     age at which an enemy planter first came within
                 distance <= 1 (None = never)
Then: seize rate grouped by each feature's buckets, for our trees and
(mirrored) for juzi's trees.

Step-3 probe in the same pass: for each plant event, the chosen cell's
nb4 vs the best plantable nb4 on the map and the best within 8 steps
of the planting planter (frame birth-1) -- did dispatch/cash push us
to suboptimal cells, and in which step bucket?
"""

import gzip
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 15
N = SIZE * SIZE
HOT_MIN = 6

NBR4 = []
for _p in range(N):
    _r, _c = divmod(_p, SIZE)
    NBR4.append([((_r - 1) % SIZE) * SIZE + _c, ((_r + 1) % SIZE) * SIZE + _c,
                 _r * SIZE + (_c - 1) % SIZE, _r * SIZE + (_c + 1) % SIZE])


def dist(a, b):
    ra, ca = divmod(a, SIZE)
    rb, cb = divmod(b, SIZE)
    dr = min(abs(ra - rb), SIZE - abs(ra - rb))
    dc = min(abs(ca - cb), SIZE - abs(ca - cb))
    return dr + dc


def load(fname):
    with gzip.open(os.path.join(HERE, "traces", fname), "rt") as f:
        return json.load(f)


def game_records(tr, us):
    """Per-tree records for one game. us = policy side."""
    frames = tr["frames"]
    bases = {int(p): [pos for _bid, pos in v]
             for p, v in tr["meta"]["bases"].items()}
    # per-frame worker positions by (player, type)
    wpos = []   # frame -> {player: {"C": [pos], "P": [pos], "A": [pos]}}
    for fr in frames:
        d = {0: {"C": [], "P": [], "A": []}, 1: {"C": [], "P": [], "A": []}}
        for wid, (pos, cargo, typ) in fr["workers"].items():
            p = int(wid.split("-")[1])
            d[p][typ].append(pos)
            d[p]["A"].append(pos)
        wpos.append(d)
    # enemy-planter occupancy per player (planter p's enemy = 1-p)
    occ = {0: defaultdict(int), 1: defaultdict(int)}
    for fr in frames:
        for wid, (pos, cargo, typ) in fr["workers"].items():
            if typ == "P":
                occ[int(wid.split("-")[1])][pos] += 1
    hot = {p: {c for c, n in occ[p].items() if n >= HOT_MIN} for p in (0, 1)}

    births = {}
    owner_hist = defaultdict(dict)   # tid -> frame_idx -> owner
    for fi, fr in enumerate(frames):
        for tid, (pos, owner, age) in fr["trees"].items():
            owner_hist[tid][fi] = owner
            if tid not in births:
                births[tid] = (fi, pos, owner)
    recs = []
    for tid, (bfi, pos, planter) in births.items():
        enemy = 1 - planter
        birth = frames[bfi]["step"]
        hist = owner_hist[tid]
        last_fi = max(hist)
        # outcome
        seized_ever = False
        first_seize_fi = None
        flips = 0
        prev_owner = planter
        for fi in sorted(hist):
            o = hist[fi]
            if o != prev_owner:
                flips += 1
                if o == enemy and first_seize_fi is None:
                    seized_ever = True
                    first_seize_fi = fi
                prev_owner = o
        end_fi = first_seize_fi if first_seize_fi is not None else last_fi
        owned_frames = [fi for fi in sorted(hist) if fi <= end_fi]
        # features
        pre = frames[bfi - 1] if bfi >= 1 else frames[0]
        nb4 = sum(pre["carbon"][n] for n in NBR4[pos])
        d_opp = min(dist(pos, b) for b in bases[enemy])
        d_own = min(dist(pos, b) for b in bases[planter])
        hd = min((dist(pos, c) for c in hot[enemy]), default=99)
        exposure3 = 0
        cover_hits = 0
        for fi in owned_frames:
            if any(dist(pos, ep) <= 3 for ep in wpos[fi][enemy]["P"]):
                exposure3 += 1
            if any(dist(pos, op) <= 3 for op in wpos[fi][planter]["A"]):
                cover_hits += 1
        own_cover = cover_hits / max(1, len(owned_frames))
        ep5 = 99
        for fi in range(bfi, min(bfi + 6, len(frames))):
            for ep in wpos[fi][enemy]["P"]:
                ep5 = min(ep5, dist(pos, ep))
        arr1_age = None
        for fi in range(bfi, len(frames)):
            if any(dist(pos, ep) <= 1 for ep in wpos[fi][enemy]["P"]):
                arr1_age = fi - bfi
                break
        recs.append(dict(
            tid=tid, planter=planter, ours=(planter == us), birth=birth,
            pos=pos, nb4=round(nb4, 1), d_opp_base=d_opp, d_own_base=d_own,
            hot_dist=hd, exposure3=exposure3, own_cover=round(own_cover, 3),
            ep5=ep5, arr1_age=arr1_age, seized_ever=seized_ever,
            first_seize_age=(first_seize_fi - bfi)
            if first_seize_fi is not None else None,
            flips=flips, censored=(birth + 50 > 300),
            end_age=frames[last_fi]["trees"][tid][2]))
    return recs


def site_probe(tr, us):
    """Per plant event: chosen nb4 vs best plantable nb4 (global / near
    the planting planter), from frame birth-1 true carbon."""
    frames = tr["frames"]
    base_cells = {pos for v in tr["meta"]["bases"].values()
                  for _bid, pos in v}
    births = {}
    for fi, fr in enumerate(frames):
        for tid, (pos, owner, age) in fr["trees"].items():
            if tid not in births:
                births[tid] = (fi, pos, owner)
    out = []
    for tid, (bfi, pos, planter) in births.items():
        pre = frames[bfi - 1] if bfi >= 1 else frames[0]
        carb = pre["carbon"]
        occupied = {v[0] for v in pre["trees"].values()}
        nb4s = {}
        for c in range(N):
            if c in base_cells or c in occupied:
                continue
            nb4s[c] = sum(carb[n] for n in NBR4[c])
        chosen = nb4s.get(pos)
        if chosen is None:
            continue
        # the planting planter: planter-side PLANTER at pos in frame bfi
        ppos = None
        for wid, (wp, cargo, typ) in frames[bfi]["workers"].items():
            if typ == "P" and wp == pos and int(wid.split("-")[1]) == planter:
                ppos = wp
                break
        if ppos is None:
            ppos = pos
        near = [v for c, v in nb4s.items() if dist(c, ppos) <= 8]
        out.append(dict(ours=(planter == us), birth=frames[bfi]["step"],
                        chosen=round(chosen, 1),
                        best_global=round(max(nb4s.values()), 1),
                        best_near=round(max(near), 1) if near else None))
    return out


def bucket_stats(recs, key, buckets):
    rows = []
    for lo, hi, label in buckets:
        sel = [r for r in recs if lo <= r[key] <= hi]
        if not sel:
            rows.append((label, 0, None, None))
            continue
        sr = sum(r["seized_ever"] for r in sel) / len(sel)
        ages = sorted(r["first_seize_age"] for r in sel
                      if r["first_seize_age"] is not None)
        med = ages[len(ages) // 2] if ages else None
        rows.append((label, len(sel), round(sr, 3), med))
    return rows


def show(title, recs, key, buckets):
    print(f"\n  {title} (n={len(recs)})")
    for label, n, sr, med in bucket_stats(recs, key, buckets):
        print(f"    {label:>10}: n={n:4d}  seize_rate={sr}  "
              f"median_first_seize_age={med}")


def main():
    files = sorted(f for f in os.listdir(os.path.join(HERE, "traces"))
                   if re.match(r"policy_v0_[45]_vs_reference_juzi", f))
    print(f"games: {len(files)}")
    all_recs, all_sites = [], []
    per_ver = defaultdict(list)
    for fn in files:
        m = re.search(r"side(\d)", fn)
        us = int(m.group(1))
        ver = re.search(r"(policy_v0_[45])", fn).group(1)
        tr = load(fn)
        recs = game_records(tr, us)
        all_recs += recs
        per_ver[ver] += [r for r in recs if r["ours"]]
        all_sites += site_probe(tr, us)
    ours = [r for r in all_recs if r["ours"]]
    juzi = [r for r in all_recs if not r["ours"]]
    print(f"trees: ours={len(ours)} (v0.4 {len(per_ver['policy_v0_4'])}, "
          f"v0.5 {len(per_ver['policy_v0_5'])}), juzi={len(juzi)}")
    for name, recs in (("OURS v0.4", per_ver["policy_v0_4"]),
                       ("OURS v0.5", per_ver["policy_v0_5"]),
                       ("OURS pooled", ours), ("JUZI pooled", juzi)):
        base = sum(r["seized_ever"] for r in recs) / max(1, len(recs))
        print(f"\n== {name}: base seize rate {base:.3f}")
        show("d_opp_base", recs, "d_opp_base",
             [(0, 4, "0-4"), (5, 6, "5-6"), (7, 8, "7-8"), (9, 99, "9+")])
        show("hot_dist", recs, "hot_dist",
             [(0, 1, "0-1"), (2, 3, "2-3"), (4, 5, "4-5"), (6, 98, "6+")])
        show("exposure3", recs, "exposure3",
             [(0, 0, "0"), (1, 5, "1-5"), (6, 15, "6-15"), (16, 999, "16+")])
        show("own_cover", recs, "own_cover",
             [(0.0, 0.25, "<.25"), (0.25, 0.5, ".25-.5"),
              (0.5, 0.75, ".5-.75"), (0.75, 1.01, ">=.75")])
        show("birth", recs, "birth",
             [(0, 49, "0-49"), (50, 99, "50-99"), (100, 149, "100-149"),
              (150, 199, "150-199"), (200, 300, "200+")])
        show("nb4", recs, "nb4",
             [(0, 80, "<80"), (80, 120, "80-120"), (120, 160, "120-160"),
              (160, 9999, "160+")])
        show("ep5", recs, "ep5",
             [(0, 2, "0-2"), (3, 4, "3-4"), (5, 6, "5-6"), (7, 99, "7+")])
        ages = sorted(r["arr1_age"] for r in recs if r["arr1_age"] is not None)
        print(f"  arr1 (enemy planter within 1): reached "
              f"{len(ages)}/{len(recs)}, median age "
              f"{ages[len(ages)//2] if ages else None}")
    # 2D: d_opp_base x birth for ours
    print("\n== OURS 2D: seize rate by d_opp_base x birth bucket")
    for blo, bhi, blab in [(0, 49, "0-49"), (50, 99, "50-99"),
                           (100, 149, "100-149"), (150, 300, "150+")]:
        row = []
        for dlo, dhi, dlab in [(0, 5, "d<=5"), (6, 7, "d6-7"), (8, 99, "d8+")]:
            sel = [r for r in ours if blo <= r["birth"] <= bhi
                   and dlo <= r["d_opp_base"] <= dhi]
            sr = (sum(r["seized_ever"] for r in sel) / len(sel)) if sel else None
            row.append(f"{dlab}: n={len(sel)} sr={sr if sr is None else round(sr,2)}")
        print(f"  birth {blab:>7}: " + " | ".join(row))
    # site probe
    print("\n== site probe (chosen nb4 vs best available, frame birth-1)")
    for name, sel_all in (("OURS", [s for s in all_sites if s["ours"]]),
                          ("JUZI", [s for s in all_sites if not s["ours"]])):
        print(f"  {name} (n={len(sel_all)})")
        for blo, bhi, blab in [(0, 49, "0-49"), (50, 99, "50-99"),
                               (100, 149, "100-149"), (150, 300, "150+")]:
            sel = [s for s in sel_all if blo <= s["birth"] <= bhi]
            if not sel:
                continue
            ch = sum(s["chosen"] for s in sel) / len(sel)
            bg = sum(s["best_global"] for s in sel) / len(sel)
            bn = sum(s["best_near"] for s in sel) / len(sel)
            print(f"    birth {blab:>7}: n={len(sel):4d} chosen={ch:6.1f} "
                  f"best_near8={bn:6.1f} best_global={bg:6.1f} "
                  f"ratio_near={ch/bn:.2f}")
    with open(os.path.join(HERE, "diag_seize_risk_v06.json"), "w") as f:
        json.dump({"ours": ours, "juzi": juzi, "sites": all_sites}, f)


if __name__ == "__main__":
    main()
