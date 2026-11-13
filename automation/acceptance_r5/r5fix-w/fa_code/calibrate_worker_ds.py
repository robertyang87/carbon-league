"""calibrate_worker -- stay-logit bias calibration for the worker net.

Argmax BC over-predicts 'stay' (val: pred 48.8% vs true 37.3%), and in
closed loop that becomes an absorbing state: collectors park on drained
cells and never relocate or return to deposit. This script searches a
constant penalty delta on the stay logit (w_b3[0] -= delta) that makes
the val predicted stay rate match the true stay rate, subject to a
small accuracy budget, and bakes it into the weights file.

Run: venv/bin/python calibrate_worker.py   (env BC_NPZ / BC_WEIGHTS as
in train_bc.py)
"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from train_bc_ds import build_dataset  # noqa: E402
from calibrate_base_ds import forward, NPZ_PATH, WEIGHTS  # noqa: E402


def main():
    _z = np.load(NPZ_PATH, allow_pickle=True)
    d = {k: _z[k] for k in _z.files}
    Xw, yw, tw, gw, Xb, yb, gb = build_dataset(d)
    W = dict(np.load(os.path.join(HERE, WEIGHTS)))
    logits = forward(W, "w_", Xw[gw])
    yv, tv = yw[gw], tw[gw]
    true_stay = float(np.mean(yv == 0))
    print(f"worker val: n={len(yv)} true stay rate {true_stay:.4f}")
    print(" delta  acc     stay_pred  stay_coll  stay_plan")
    best = None
    for delta in np.arange(0.0, 2.01, 0.1):
        lg = logits.copy()
        lg[:, 0] -= delta
        pred = lg.argmax(1)
        acc = float(np.mean(pred == yv))
        sp = float(np.mean(pred == 0))
        sc = float(np.mean(pred[tv == 0] == 0))
        spl = float(np.mean(pred[tv == 1] == 0))
        print(f"  {delta:3.1f}  {acc:.4f}  {sp:.4f}    {sc:.4f}    {spl:.4f}")
        gap = abs(sp - true_stay)
        if best is None or gap < best[0]:
            best = (gap, delta, acc, sp)
    gap, delta, acc, sp = best
    base_acc = float(np.mean(logits.argmax(1) == yv))
    print(f"\nchosen delta={delta:.1f}: stay_pred {sp:.4f} "
          f"(true {true_stay:.4f}), acc {acc:.4f} (was {base_acc:.4f})")
    W["w_b3"] = W["w_b3"].copy()
    W["w_b3"][0] -= np.float32(delta)
    np.savez(os.path.join(HERE, WEIGHTS), **W)
    with open(os.path.join(
            HERE, WEIGHTS.replace("_weights.npz",
                                  "_worker_calibration.json")), "w") as f:
        json.dump({"stay_bias_delta": float(delta),
                   "val_stay_pred": sp, "val_stay_true": true_stay,
                   "val_acc_with_bias": acc,
                   "val_acc_without_bias": base_acc}, f)
    print(f"baked stay bias into {WEIGHTS} (w_b3[0] -= delta)")


if __name__ == "__main__":
    main()
