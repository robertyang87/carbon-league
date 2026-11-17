"""bench_encode_only.py -- trunk-only (encode) decomposition companion to
bench_latency.py. Times CachePolicy.encode (the trunk forward that act()
builds on) for teacher vs student-B prototype on the SAME single-pass
positions (sealed 994400, steps 200-207, both seats + step 0 = 18).
Purpose: separate the trunk cost from act()'s fixed per-call overhead
(Prefix build + 12-step decode loop), so the section-4 caliber question
(trunk vs act) can be ruled on with both numbers on the table.
DGX seat, nice'd by the caller; NOT a section-4 verdict."""

import argparse
import json
import os
import sys
import time

import numpy as np


def stats(ts):
    a = np.sort(np.array(ts) * 1000.0)
    n = len(a)
    return {"n_calls": n, "mean_ms": float(a.mean()),
            "p95_ms": float(a[min(n - 1, int(0.95 * n))]),
            "min_ms": float(a[0]), "max_ms": float(a[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--teacher-weights", required=True)
    ap.add_argument("--student-npz", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--cycles", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import gzip
    import cnet2_feat as F
    from eval_v53 import CFG
    from fastsim import FastCarbon
    from cnet2_search import CachePolicy, load_weights

    with gzip.open(a.game, "rt", encoding="utf-8") as fh:
        steps = json.load(fh)["steps"]
    sim = FastCarbon(CFG).reset(seed=1, full_carbon=steps[0][0]["observation"]["full_carbon"])
    feats = [F.Feat(0), F.Feat(1)]
    positions = []

    def snap():
        for seat in (0, 1):
            x = feats[seat].observe(sim.obs(seat))
            positions.append({k: (v.copy() if hasattr(v, "copy") else v) for k, v in x.items()})

    snap()
    for t in range(207):
        sim.step(steps[t + 1][0].get("action") or {}, steps[t + 1][1].get("action") or {})
        if t + 1 >= 200:
            snap()
    print(f"positions: {len(positions)}", flush=True)

    out = {"positions": len(positions), "cycles": a.cycles,
           "loadavg_start": list(os.getloadavg())}
    for name, wpath in (("teacher", a.teacher_weights), ("student_B_random", a.student_npz)):
        pol = CachePolicy(load_weights(wpath))
        for x in positions:  # warmup cycle
            pol.encode(x, 2)
        ts = []
        for _ in range(a.cycles):
            for x in positions:
                t0 = time.perf_counter()
                pol.encode(x, 2)
                ts.append(time.perf_counter() - t0)
        out[name] = stats(ts)
        print(name, out[name], flush=True)
    out["ratio_mean"] = out["teacher"]["mean_ms"] / out["student_B_random"]["mean_ms"]
    out["ratio_p95"] = out["teacher"]["p95_ms"] / out["student_B_random"]["p95_ms"]
    out["loadavg_end"] = list(os.getloadavg())
    print("ratio_mean", out["ratio_mean"], "ratio_p95", out["ratio_p95"])
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
