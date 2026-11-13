#!/usr/bin/env python3
"""train_distill2 -- Case-B segment 2 (after checkpoint-1 verdict).

Changes vs segment 1, per the checkpoint-1 attribution:
  * training data = base corpus + DAgger correction games x3
    (bc_data/distill_dataset_dagger.npz);
  * base head class weights use FULL inverse frequency (cw_power
    1.0) instead of sqrt -- segment-1 RECP recall was 0.119;
  * full schedule, no truncation: worker 20 epochs, base 30,
    per-epoch checkpoints via train_distill.train_head (resumable).
Outputs the production names: distilled_v1_weights.npz +
distill_train_metrics.json (heads named worker/base so the base
per-class recalls are recorded).
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

import train_distill as TD  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "bc_data" / "distill_dataset_dagger.npz"


def main():
    t0 = time.time()
    d = np.load(DATA)
    out, metrics = {}, {}
    wp, wm = TD.train_head(
        "worker", d["W_x"], d["W_y"], d["W_split"], 247, 256, 5,
        epochs=20, batch=8192, cw_power=0.5,
        meta_slices={"phase": d["W_phase"], "state": d["W_state"],
                     "type": d["W_type"]})
    for k, v in wp.items():
        out["w_" + k] = v
    # w_mean/w_std come from train_head (train-split stats)
    metrics["worker"] = wm
    bp, bm = TD.train_head(
        "base", d["B_x"], d["B_y"], d["B_split"], 108, 128, 3,
        epochs=30, batch=4096, cw_power=1.0)
    for k, v in bp.items():
        out["b_" + k] = v
    # b_mean/b_std come from train_head (train-split stats)
    metrics["base"] = bm
    np.savez(TD.WEIGHTS_OUT, **out)
    size = os.path.getsize(TD.WEIGHTS_OUT)
    metrics["weights_bytes"] = size
    metrics["wall_sec"] = round(time.time() - t0, 1)
    metrics["dataset"] = str(DATA)
    TD.METRICS_OUT.write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    print(f"wrote {TD.WEIGHTS_OUT} ({size} bytes)")


if __name__ == "__main__":
    main()
