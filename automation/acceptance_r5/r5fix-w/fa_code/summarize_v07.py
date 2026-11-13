"""summarize_v07: trace aggregates for the v0.7 report, computed with
ONE yardstick over both the v0.5 and the v0.7 recorded juzi games:
  - planter deaths by cause (diag_planter_deaths_v07.analyse)
  - alive planters at step 250
  - tree births in step buckets >= 200 / >= 250
  - seized-ever rate per side (tree born to A, owned by B at any
    later frame)
  - opening (step < 50) plant-site nb4: TRUE carbon sum over the
    birth cell's 4 neighbours in the frame BEFORE the birth step
    (decision-time map, mirror of the v0.5/v0.6 diag definition)
"""

import gzip
import json
import os
from collections import Counter

import diag_planter_deaths_v07 as D

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 15


def nbr4(pos):
    r, c = divmod(pos, SIZE)
    return [(((r + dy) % SIZE) * SIZE + ((c + dx) % SIZE))
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0))]


def trace_aggs(path):
    tr = json.load(gzip.open(path, "rt"))
    me = tr["meta"]["policy_side"]
    frames = tr["frames"]
    born = {}          # tid -> (owner, step, cell)
    seized = set()     # tids ever owned by the non-birth side
    nb4 = {0: [], 1: []}
    for t in range(1, 301):
        prev, cur = frames[t - 1], frames[t]
        for tid, v in cur["trees"].items():
            pos, owner, _age = v
            if tid not in born:
                born[tid] = (owner, t, pos)
                if t < 50:
                    nb4[owner].append(sum(prev["carbon"][n]
                                          for n in nbr4(pos)))
            elif born[tid][0] != owner:
                seized.add(tid)
    out = {}
    for side in (0, 1):
        mine = [tid for tid, (o, _s, _c) in born.items() if o == side]
        out[side] = {
            "trees_born": len(mine),
            "seized_ever": sum(1 for tid in mine if tid in seized),
            "seized_rate": (sum(1 for tid in mine if tid in seized)
                            / max(1, len(mine))),
            "opening_nb4": (sum(nb4[side]) / len(nb4[side])
                            if nb4[side] else None),
            "opening_n": len(nb4[side]),
        }
    return out, me


def run(prefix):
    files = [os.path.join(HERE, "traces",
                          f"{prefix}_vs_reference_juzi_seed{s}_side{d}.json.gz")
             for s in (1000, 1001, 1002, 1003) for d in (0, 1)]
    agg_us = Counter()
    agg_jz = Counter()
    nb4_us, nb4_jz = [], []
    deaths_us, deaths_jz = [], []
    alive_us = alive_jz = late_us = late_jz = l250_us = l250_jz = 0
    for f in files:
        deaths, stats, me = D.analyse(f)
        deaths_us += [d for d in deaths if d["is_us"]]
        deaths_jz += [d for d in deaths if not d["is_us"]]
        juzi = 1 - me
        alive_us += stats[me]["alive250"]
        alive_jz += stats[juzi]["alive250"]
        late_us += stats[me]["late_plants"]
        late_jz += stats[juzi]["late_plants"]
        aggs, _ = trace_aggs(f)
        for key, side, nb in (("us", me, nb4_us), ("jz", juzi, nb4_jz)):
            a = agg_us if key == "us" else agg_jz
            a["trees_born"] += aggs[side]["trees_born"]
            a["seized_ever"] += aggs[side]["seized_ever"]
            if aggs[side]["opening_nb4"] is not None:
                nb.append((aggs[side]["opening_nb4"],
                           aggs[side]["opening_n"]))
        # step>=250 plantings from births data inside trace_aggs is
        # not split out; recompute cheaply below via born steps.
    # >=250 split needs born steps: quick second pass
    def born250(path):
        tr = json.load(gzip.open(path, "rt"))
        me2 = tr["meta"]["policy_side"]
        seen, cnt = set(), {0: 0, 1: 0}
        for t in range(1, 301):
            for tid, v in tr["frames"][t]["trees"].items():
                if tid not in seen:
                    seen.add(tid)
                    if t >= 250:
                        cnt[v[1]] += 1
        return cnt[me2], cnt[1 - me2]
    for f in files:
        a, b = born250(f)
        l250_us += a
        l250_jz += b
    def wavg(nb):
        return sum(v * n for v, n in nb) / max(1, sum(n for _v, n in nb))
    print(f"\n### {prefix} (8 games vs juzi)")
    print(f"planter deaths/game: us {len(deaths_us)/8:.2f} "
          f"{dict(Counter(d['cause'] for d in deaths_us))}")
    print(f"                     jz {len(deaths_jz)/8:.2f} "
          f"{dict(Counter(d['cause'] for d in deaths_jz))}")
    print(f"alive planters @250: us {alive_us/8:.2f}  jz {alive_jz/8:.2f}")
    print(f"tree births >=200/game: us {late_us/8:.2f}  jz {late_jz/8:.2f}")
    print(f"tree births >=250/game: us {l250_us/8:.2f}  jz {l250_jz/8:.2f}")
    for lbl, a in (("us", agg_us), ("jz", agg_jz)):
        print(f"seized-ever {lbl}: {a['seized_ever']}/{a['trees_born']} "
              f"= {a['seized_ever']/max(1, a['trees_born']):.1%}")
    print(f"opening nb4 (step<50, true carbon): us {wavg(nb4_us):.1f} "
          f"(n={sum(n for _v, n in nb4_us)})  jz {wavg(nb4_jz):.1f} "
          f"(n={sum(n for _v, n in nb4_jz)})")


if __name__ == "__main__":
    run("policy_v0_5")
    run("policy_v0_7")
