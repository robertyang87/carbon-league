"""bench_latency.py -- section-4 REACHABILITY prototype (not the formal section-4
measurement, which is L20's artifact on spot-36 in the war-room window).

Same seat, same real positions, batch=1: production teacher (06a03d9c) vs
B-tier random-weight student prototype, through the production inference path
(cnet2_search.CachePolicy.act, pinned stack from --code-dir). Positions are
real (x, aux) pairs extracted from the sealed selfplay-994400 replay by the
pinned gate's extract_scene (steps 200-207 x both seats = 32 positions).
Cycling 32 distinct positions defeats CachePolicy's last-encode cache, so
every call pays the full encode+decode, as in production branch evaluation.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bench(pol, positions, cycles, teacher_style=2):
    times = []
    for _ in range(cycles):
        for (x, aux) in positions:
            t0 = time.perf_counter()
            pol.act(x, aux, teacher=teacher_style, want_u=True)
            times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {"n_calls": int(t.size), "mean_ms": float(t.mean()),
            "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "min_ms": float(t.min()), "max_ms": float(t.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--teacher-weights", required=True)
    ap.add_argument("--student-npz", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.code_dir)
    mods = {}
    for m in ("cnet2_search", "cnet2_np", "cnet2_feat", "fastsim", "stage1_teacher_gate", "stage0_teacher"):
        mods[m] = md5_of(os.path.join(a.code_dir, m + ".py"))
    import gzip
    import cnet2_feat as F
    from eval_v53 import CFG
    from fastsim import FastCarbon
    from cnet2_search import CachePolicy, load_weights

    # Single-pass replay (the gate's extract_scene discipline: both seats'
    # Feats observe every step exactly once), snapshotting both seats'
    # (x, aux) at steps 200-207 -> 16 real positions. (Per-scene
    # extract_scene calls would replay the prefix 16 times.)
    with gzip.open(a.game, "rt", encoding="utf-8") as fh:
        steps = json.load(fh)["steps"]
    sim = FastCarbon(CFG).reset(seed=1, full_carbon=steps[0][0]["observation"]["full_carbon"])
    feats = [F.Feat(0), F.Feat(1)]
    positions = []

    def snap(step):
        for seat in (0, 1):
            x = feats[seat].observe(sim.obs(seat))
            xc = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in x.items()}
            auxc = {k: np.array(v, copy=True) for k, v in feats[seat].aux.items()}
            positions.append((xc, auxc))

    snap(0)
    for t in range(207):
        sim.step(steps[t + 1][0].get("action") or {}, steps[t + 1][1].get("action") or {})
        if t + 1 >= 200:
            snap(t + 1)
    print(f"positions: {len(positions)}", flush=True)

    cpu = ""
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.lower().startswith(("model name", "model", "hardware")):
                cpu = ln.split(":", 1)[1].strip()
                if cpu:
                    break
    except OSError:
        pass
    load0 = os.getloadavg()
    res = {"seat": "DGX", "cpu": cpu, "loadavg_before": load0,
           "module_md5": mods,
           "teacher_weights_md5": md5_of(a.teacher_weights),
           "student_npz_md5": md5_of(a.student_npz),
           "thread_env": {k: os.environ.get(k) for k in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "positions": len(positions), "cycles": a.cycles}

    tw = load_weights(a.teacher_weights)
    sw = load_weights(a.student_npz)
    for name, w in (("teacher", tw), ("student_B_random", sw)):
        pol = CachePolicy(w)
        bench(pol, positions, 1)  # warmup cycle, discarded
        r = bench(pol, positions, a.cycles)
        res[name] = r
        print(name, r)
    res["ratio_mean"] = res["teacher"]["mean_ms"] / res["student_B_random"]["mean_ms"]
    res["ratio_p95"] = res["teacher"]["p95_ms"] / res["student_B_random"]["p95_ms"]
    res["loadavg_after"] = os.getloadavg()
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("ratio_mean", res["ratio_mean"], "ratio_p95", res["ratio_p95"])


if __name__ == "__main__":
    main()
