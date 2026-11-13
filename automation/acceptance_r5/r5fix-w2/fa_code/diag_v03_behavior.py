"""diag_v03_behavior: behavioural readout from the v0.3 vs juzi traces.

Per game: tree-owner flips (enemy seizes of our trees, our seizes --
by tree id, owner changes between consecutive frames), whether the
3-planter corps ever fielded (max simultaneous friendly planters,
frames with >= 3), and the death structure under the v2 classifier
(diag_deaths_v03.analyse): deaths/game, ledger death cargo, recovered
share, true loss, hunt_kill share.
"""

import gzip
import json
import os
from statistics import mean

import diag_deaths_v03 as D

HERE = os.path.dirname(os.path.abspath(__file__))


def flips(path):
    with gzip.open(path, "rt") as f:
        tr = json.load(f)
    me = tr["meta"]["policy_side"]
    frames = tr["frames"]
    foe_seize = my_seize = 0
    max_p = 0
    frames_p3 = 0
    for t in range(300):
        cur, nxt = frames[t], frames[t + 1]
        for tid, (pos, own, age) in cur["trees"].items():
            if tid in nxt["trees"]:
                nown = nxt["trees"][tid][1]
                if own == me and nown != me:
                    foe_seize += 1
                elif own != me and nown == me:
                    my_seize += 1
        np_ = sum(1 for w, v in frames[t + 1]["workers"].items()
                  if w.startswith(f"player-{me}-") and v[2] == "P")
        max_p = max(max_p, np_)
        if np_ >= 3:
            frames_p3 += 1
    return foe_seize, my_seize, max_p, frames_p3


def main():
    files = sorted(f for f in os.listdir(os.path.join(HERE, "traces"))
                   if f.startswith("policy_v0_3_vs_reference_juzi"))
    print(f"{len(files)} v0.3 juzi traces")
    agg = []
    for fn in files:
        path = os.path.join(HERE, "traces", fn)
        fs, ms, maxp, fp3 = flips(path)
        meta, ev = D.analyse(path)
        tot = sum(e["cargo"] for e in ev)
        rec = sum(e["cargo"] for e in ev if e["recovered"])
        hk = sum(e["cargo"] for e in ev
                 if e["cause"] == "hunt_kill")
        agg.append((fs, ms, maxp, fp3, len(ev), tot, rec, hk))
        print(f"seed{meta['seed']} side{meta['policy_side']}: "
              f"foe_seize {fs} my_seize {ms} max_planters {maxp} "
              f"frames_p>=3 {fp3} deaths {len(ev)} deathcargo {tot:.0f} "
              f"recovered {rec:.0f} huntkill {hk:.0f}")
    if agg:
        print(f"\nmeans: foe_seize {mean(a[0] for a in agg):.1f} "
              f"my_seize {mean(a[1] for a in agg):.1f} "
              f"frames_p>=3 {mean(a[3] for a in agg):.0f} "
              f"deaths {mean(a[4] for a in agg):.1f} "
              f"deathcargo {mean(a[5] for a in agg):.0f} "
              f"recovered {mean(a[6] for a in agg):.0f} "
              f"huntkill {mean(a[7] for a in agg):.0f}")


if __name__ == "__main__":
    main()
