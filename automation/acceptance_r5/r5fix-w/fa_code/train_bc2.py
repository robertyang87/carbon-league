"""train_bc2 -- stage-2 BC A/B: control features vs belief-augmented
features, identical architecture/hyperparameters/split (train_bc's
train_mlp and class-weight rules), on bc_data/bc2_dataset.npz.

Gate from the stage-2 work order: the belief arm's val top-1 must not
drop vs control before the belief features may enter PPO. The belief
arm's weights are exported as the PPO warm start
(rl2_init_weights.npz, shadow npz format).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_bc import train_mlp  # noqa: E402


def run_arm(Xw, yw, tw, gw, Xb, yb, gb, label):
    out = {"arm": label}
    counts = np.bincount(yw[~gw], minlength=5).astype(np.float64)
    cw = (len(yw[~gw]) / (5 * counts)) ** 0.5
    w_weights, wpred, wacc = train_mlp(
        Xw[~gw], yw[~gw], Xw[gw], yw[gw], 5, 192, 40, 1e-3, 1e-4, cw)
    out["worker_val_acc"] = float(wacc)
    out["worker_val_stay_baseline"] = float(np.mean(yw[gw] == 0))
    for tname, tval in (("collector", 0), ("planter", 1)):
        m = tw[gw] == tval
        out[f"worker_{tname}_acc"] = float(np.mean(wpred[m] == yw[gw][m]))
        out[f"worker_{tname}_stay_baseline"] = float(
            np.mean(yw[gw][m] == 0))
    counts_b = np.bincount(yb[~gb], minlength=3).astype(np.float64)
    cwb = np.minimum(len(yb[~gb]) / (3 * counts_b), 13.0)
    b_weights, bpred, bacc = train_mlp(
        Xb[~gb], yb[~gb], Xb[gb], yb[gb], 3, 96, 40, 1e-3, 1e-4, cwb,
        seed=1)
    out["base_val_acc"] = float(bacc)
    out["base_val_none_baseline"] = float(np.mean(yb[gb] == 0))
    for cname, cval in (("RECCOLLECTOR", 1), ("RECPLANTER", 2)):
        m = yb[gb] == cval
        out[f"base_{cname}_recall"] = float(
            np.mean(bpred[m] == cval)) if np.sum(m) else float("nan")
    print(json.dumps(out, indent=1), flush=True)
    return out, w_weights, b_weights


def main():
    z = np.load(os.path.join(HERE, "bc_data", "bc2_dataset.npz"))
    d = {k: z[k] for k in z.files}
    yw, tw, gw = d["yw"], d["tw"], d["gw"]
    yb, gb = d["yb"], d["gb"]
    results = {}
    results["control"], _, _ = run_arm(
        d["Xw"], yw, tw, gw, d["Xb"], yb, gb, "control")
    results["belief"], w_w, b_w = run_arm(
        d["Xw2"], yw, tw, gw, d["Xb2"], yb, gb, "belief")
    out = {}
    for k, v in w_w.items():
        out[f"w_{k}"] = v
    for k, v in b_w.items():
        out[f"b_{k}"] = v
    np.savez(os.path.join(HERE, "rl2_init_weights.npz"), **out)
    with open(os.path.join(HERE, "bc2_train_metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    c, b = results["control"], results["belief"]
    print("=== A/B summary (val top-1) ===")
    for k in ("worker_val_acc", "worker_collector_acc",
              "worker_planter_acc", "base_val_acc",
              "base_RECCOLLECTOR_recall", "base_RECPLANTER_recall"):
        print(f"{k:28s} control {c[k]:.4f}  belief {b[k]:.4f}  "
              f"delta {b[k] - c[k]:+.4f}")


if __name__ == "__main__":
    main()
