"""calibrate_base -- post-hoc logit-bias calibration for the base net.

The base MLP was trained with inverse-frequency class weights; raw
argmax then recruits on ~25% of base-steps vs the true ~4%, which
would flood the live shadow with collectors. This script measures the
val-set trade-off of adding a bias delta to the class-0 (none) logit
and bakes the chosen delta into shadow_weights.npz (b_b3), subject to
predicted positive rate <= 10% while maximizing RECC+RECP recall.

Also dumps the worker net's predicted action distribution on val
(diagnostic for the report; worker argmax is left untouched).

Run: venv/bin/python calibrate_base.py
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from train_bc_ds_cached import build_dataset, VAL_FILES  # noqa: E402


def forward(W, prefix, X):
    h = (X - W[prefix + "mean"]) / W[prefix + "std"]
    for i in (1, 2):
        h = np.maximum(h @ W[f"{prefix}W{i}"].T + W[f"{prefix}b{i}"], 0.0)
    return h @ W[prefix + "W3"].T + W[prefix + "b3"]


NPZ_PATH = os.environ.get(
    "BC_NPZ", os.path.join(HERE, "bc_data", "claude_family.npz"))
WEIGHTS = os.environ.get("BC_WEIGHTS", "shadow_weights.npz")


def main():
    _z = np.load(NPZ_PATH, allow_pickle=True)
    d = {k: _z[k] for k in _z.files}
    Xw, yw, tw, gw, Xb, yb, gb = build_dataset(d)
    W = dict(np.load(os.path.join(HERE, WEIGHTS)))

    # worker predicted distribution (diagnostic only)
    wpred = forward(W, "w_", Xw[gw]).argmax(1)
    names = ["stay", "UP", "RIGHT", "DOWN", "LEFT"]
    print("worker val action distribution (pred vs true):")
    for c in range(5):
        print(f"  {names[c]:6s} pred {np.mean(wpred == c):.3f}  "
              f"true {np.mean(yw[gw] == c):.3f}")

    logits = forward(W, "b_", Xb[gb])
    yv = yb[gb]
    print(f"\nbase val: n={len(yv)} true pos rate {np.mean(yv > 0):.4f}")
    print(" delta  pos_rate  RECC_rec  RECP_rec  precision")
    best = None
    for delta in np.arange(0.0, 4.01, 0.25):
        lg = logits.copy()
        lg[:, 0] += delta
        pred = lg.argmax(1)
        pos = float(np.mean(pred > 0))
        rec1 = float(np.mean(pred[yv == 1] == 1)) if np.any(yv == 1) else 0.0
        rec2 = float(np.mean(pred[yv == 2] == 2)) if np.any(yv == 2) else 0.0
        prec = float(np.mean(yv[pred > 0] == pred[pred > 0])) \
            if np.any(pred > 0) else 0.0
        print(f"  {delta:4.2f}   {pos:.4f}   {rec1:.3f}    {rec2:.3f}"
              f"    {prec:.3f}")
        if pos <= 0.10:
            score = rec1 + rec2
            if best is None or score > best[0]:
                best = (score, delta, pos, rec1, rec2, prec)
    score, delta, pos, rec1, rec2, prec = best
    print(f"\nchosen delta={delta:.2f}: pos_rate {pos:.4f}, RECC recall "
          f"{rec1:.3f}, RECP recall {rec2:.3f}, precision {prec:.3f}")
    W["b_b3"] = W["b_b3"].copy()
    W["b_b3"][0] += np.float32(delta)
    np.savez(os.path.join(HERE, WEIGHTS), **W)
    with open(os.path.join(
            HERE, WEIGHTS.replace("_weights.npz", "_calibration.json")),
            "w") as f:
        import json
        json.dump({"base_bias_delta": float(delta),
                   "val_pos_rate": pos, "val_recc_recall": rec1,
                   "val_recp_recall": rec2, "val_precision": prec}, f)
    print(f"baked bias into {WEIGHTS} (b_b3[0] += delta)")


if __name__ == "__main__":
    main()
