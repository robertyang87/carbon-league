"""summarize_v08: trace aggregates for the v0.8 report, ONE yardstick
over recorded juzi games (default prefixes policy_v0_5 and
policy_v0_7_nosurv, seeds 1000-1003 x2):
  - seized-ever rate per side (summarize_v07.trace_aggs)
  - reseize latency: for each tree born to side A, a flip to B at t1
    and the FIRST flip back to A at t2 gives latency t2 - t1; trees
    never recovered count as unrecovered. Reported per birth side:
    flips, recovered, mean latency over recovered.
  - planter deaths by cause / alive @250 / late births
    (diag_planter_deaths_v07.analyse, via summarize_v07's approach)
  - opening (step < 50) plant-site nb4 true carbon
"""

import gzip
import json
import os
import sys
from collections import Counter

import diag_planter_deaths_v07 as D
from summarize_v07 import trace_aggs

HERE = os.path.dirname(os.path.abspath(__file__))


def latency_aggs(path):
    tr = json.load(gzip.open(path, "rt"))
    me = tr["meta"]["policy_side"]
    frames = tr["frames"]
    born = {}                     # tid -> birth owner
    owner = {}                    # tid -> current owner
    flip_step = {}                # tid -> step it flipped away from
                                  # birth owner (pending recovery)
    out = {0: {"flips": 0, "recovered": 0, "lat_sum": 0},
           1: {"flips": 0, "recovered": 0, "lat_sum": 0}}
    for t in range(1, 301):
        for tid, v in frames[t]["trees"].items():
            _pos, o, _age = v
            if tid not in born:
                born[tid] = o
                owner[tid] = o
            elif o != owner[tid]:
                b = born[tid]
                if o != b and owner[tid] == b:
                    out[b]["flips"] += 1
                    flip_step[tid] = t
                elif o == b and tid in flip_step:
                    out[b]["recovered"] += 1
                    out[b]["lat_sum"] += t - flip_step.pop(tid)
                owner[tid] = o
    return out, me


def run(prefix, seeds=(1000, 1001, 1002, 1003)):
    files = [os.path.join(HERE, "traces",
                          f"{prefix}_vs_reference_juzi_seed{s}_side{d}.json.gz")
             for s in seeds for d in (0, 1)]
    n = len(files)
    agg = {s: Counter() for s in ("us", "jz")}
    nb4 = {"us": [], "jz": []}
    deaths = {"us": [], "jz": []}
    alive = Counter()
    late = Counter()
    l250 = Counter()
    for f in files:
        dth, stats, me = D.analyse(f)
        juzi = 1 - me
        deaths["us"] += [d for d in dth if d["is_us"]]
        deaths["jz"] += [d for d in dth if not d["is_us"]]
        alive["us"] += stats[me]["alive250"]
        alive["jz"] += stats[juzi]["alive250"]
        late["us"] += stats[me]["late_plants"]
        late["jz"] += stats[juzi]["late_plants"]
        aggs, _ = trace_aggs(f)
        lat, _ = latency_aggs(f)
        for key, side in (("us", me), ("jz", juzi)):
            a = agg[key]
            a["trees_born"] += aggs[side]["trees_born"]
            a["seized_ever"] += aggs[side]["seized_ever"]
            a["flips"] += lat[side]["flips"]
            a["recovered"] += lat[side]["recovered"]
            a["lat_sum"] += lat[side]["lat_sum"]
            if aggs[side]["opening_nb4"] is not None:
                nb4[key].append((aggs[side]["opening_nb4"],
                                 aggs[side]["opening_n"]))
        tr = json.load(gzip.open(f, "rt"))
        seen, cnt = set(), {0: 0, 1: 0}
        for t in range(1, 301):
            for tid, v in tr["frames"][t]["trees"].items():
                if tid not in seen:
                    seen.add(tid)
                    if t >= 250:
                        cnt[v[1]] += 1
        l250["us"] += cnt[me]
        l250["jz"] += cnt[juzi]

    def wavg(lst):
        return sum(v * k for v, k in lst) / max(1, sum(k for _v, k in lst))

    print(f"\n### {prefix} ({n} games vs juzi)")
    for key in ("us", "jz"):
        a = agg[key]
        lat_mean = a["lat_sum"] / a["recovered"] if a["recovered"] else None
        print(f"{key}: trees_born {a['trees_born']}  seized-ever "
              f"{a['seized_ever']}/{a['trees_born']} = "
              f"{a['seized_ever'] / max(1, a['trees_born']):.1%}  "
              f"flips {a['flips']}  recovered {a['recovered']}  "
              f"reseize latency {lat_mean if lat_mean is None else round(lat_mean, 2)}")
        print(f"    planter deaths/game {len(deaths[key]) / n:.2f} "
              f"{dict(Counter(d['cause'] for d in deaths[key]))}")
        print(f"    alive@250/game {alive[key] / n:.2f}  births>=200/game "
              f"{late[key] / n:.2f}  births>=250/game {l250[key] / n:.2f}  "
              f"opening nb4 {wavg(nb4[key]):.1f}")


if __name__ == "__main__":
    prefixes = sys.argv[1:] or ["policy_v0_5", "policy_v0_7_nosurv"]
    for p in prefixes:
        run(p)
