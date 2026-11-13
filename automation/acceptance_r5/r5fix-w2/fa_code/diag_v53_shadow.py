"""diag_v53_shadow: per-phase pathology metrics from recorded traces.

For the v53-vs-shadow_alpha_v2 games (NN-style opponent) and, for
contrast, the v53-vs-juzi games on the same seed, compute per phase
(steps 1-100 / 101-200 / 201-300) from the trace frames alone:
  * deaths per phase (worker id disappears) and the cargo it carried
    (death cargo), per player;
  * worker counts (C/P) at steps 100/200/300;
  * tree counts per owner at steps 100/200/300 and trees gained/lost;
  * total field cargo (sum of worker cargo) at steps 100/200/299 --
    the "stranded trajectory";
  * cash at steps 0/100/200/300.
Frames store workers keyed by wid WITHOUT owner; owner is recovered by
tracking each wid's first-seen base side is impossible from frames
alone, so ownership is inferred per game from the eval ledger protocol:
the trace meta gives policy/opp and policy_side; worker ownership is
recovered by matching worker spawn positions to the nearest base at
first sight is unreliable -- instead we use the frames' worker ids
grouped by continuity with the step-0..5 spawns: bases in meta give
both players' base cells; a new wid first appears on a base cell,
which identifies its owner exactly.
"""

import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "traces")
PHASES = ((1, 100), (101, 200), (201, 300))


def load_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def owner_map(trace):
    """wid -> player, from the base cell it first appears on."""
    meta = trace["meta"]
    base_owner = {}
    for p, blist in meta["bases"].items():
        for _bid, pos in blist:
            base_owner[pos] = int(p)
    owner = {}
    for fr in trace["frames"]:
        for wid, w in fr["workers"].items():
            if wid not in owner and w[0] in base_owner:
                owner[wid] = base_owner[w[0]]
        if len(owner) == len(fr["workers"]) and fr["step"] > 5:
            # keep scanning cheaply only while new wids may appear
            pass
    return owner


def analyze(path):
    tr = load_trace(path)
    meta = tr["meta"]
    frames = tr["frames"]
    owner = owner_map(tr)
    pol_side = meta["policy_side"]
    out = {"file": os.path.basename(path), "policy": meta["policy"],
           "opp": meta["opp"], "seed": meta["seed"],
           "policy_side": pol_side, "final_cash": meta["final_cash"]}
    # deaths & death cargo per phase, per player
    dd = {0: [], 1: []}
    for pi, (lo, hi) in enumerate(PHASES):
        for p in (0, 1):
            deaths = 0
            dcargo = 0.0
            for t in range(lo, hi + 1):
                prev = frames[t - 1]["workers"]
                cur = frames[t]["workers"]
                for wid, w in prev.items():
                    if wid not in cur and owner.get(wid) == p:
                        deaths += 1
                        dcargo += w[1]
            dd[p].append((deaths, round(dcargo, 1)))
    out["deaths_phases"] = dd
    # snapshots
    snaps = {}
    for s in (100, 200, 300):
        fr = frames[s]
        cnt = {0: [0, 0], 1: [0, 0]}
        cargo = {0: 0.0, 1: 0.0}
        for wid, w in fr["workers"].items():
            p = owner.get(wid)
            if p is None:
                continue
            cnt[p][0 if w[2] == "C" else 1] += 1
            cargo[p] += w[1]
        trees = {0: 0, 1: 0}
        for _tid, t in fr["trees"].items():
            trees[t[1]] += 1
        snaps[s] = {"workers_CP": cnt, "cargo": {k: round(v, 1)
                                                 for k, v in cargo.items()},
                    "trees": trees, "cash": fr["cash"]}
    out["snaps"] = snaps
    return out


def show(res):
    ps = res["policy_side"]
    print(f"== {res['file']}  policy={res['policy']} (P{ps}) "
          f"vs {res['opp']}  final {res['final_cash']}")
    for label, p in (("POLICY", ps), ("OPP", 1 - ps)):
        d = res["deaths_phases"][p]
        print(f"  {label} P{p}: deaths/phase "
              + " | ".join(f"{a} ({b})" for a, b in d)
              + f"  total {sum(a for a, _ in d)} "
                f"({sum(b for _, b in d):.0f} cargo)")
        for s in (100, 200, 300):
            sn = res["snaps"][s]
            print(f"    step{s}: C/P={sn['workers_CP'][p]} "
                  f"cargo={sn['cargo'][p]} trees={sn['trees'][p]} "
                  f"cash={sn['cash'][p]}")


if __name__ == "__main__":
    pats = sys.argv[1:]
    files = sorted(f for f in os.listdir(TR)
                   if any(p in f for p in pats) and f.endswith(".json.gz"))
    for f in files:
        show(analyze(os.path.join(TR, f)))
