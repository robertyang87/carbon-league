#!/usr/bin/env python3
"""smoke_train -- checkpoint-1 flux version (small-steps directive).

Same data / architecture / recipe as train_distill.py, but a short
first segment (worker 4 epochs, base 6 epochs) whose ONLY job is to
expose distortions early: offline fidelity slices, a 4-game mini
pairing vs base, and the distortion checklist (tree steals, endgame
leftover, opening double-kill stress).  Writes distilled_v0_weights
.npz + smoke_train_metrics.json.  The full run (train_distill.py,
20/30 epochs, checkpointed) follows after checkpoint-1 attribution.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

import train_distill as TD  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    t0 = time.time()
    d = np.load(TD.DATA)
    out, metrics = {}, {}
    wp, wm = TD.train_head(
        "worker_smoke", d["W_x"], d["W_y"], d["W_split"], 247, 256, 5,
        epochs=4, batch=8192,
        meta_slices={"phase": d["W_phase"], "state": d["W_state"],
                     "type": d["W_type"]})
    for k, v in wp.items():
        out["w_" + k] = v
    # w_mean/w_std come from train_head (train-split stats)
    metrics["worker"] = wm
    bp, bm = TD.train_head(
        "base_smoke", d["B_x"], d["B_y"], d["B_split"], 108, 128, 3,
        epochs=6, batch=4096)
    for k, v in bp.items():
        out["b_" + k] = v
    # b_mean/b_std come from train_head (train-split stats)
    metrics["base"] = bm
    wpath = HERE / "distilled_v0_weights.npz"
    np.savez(wpath, **out)
    metrics["weights_bytes"] = os.path.getsize(wpath)
    metrics["wall_sec"] = round(time.time() - t0, 1)
    (HERE / "smoke_train_metrics.json").write_text(
        json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    print(f"wrote {wpath} ({metrics['weights_bytes']} bytes)")


if __name__ == "__main__":
    main()
