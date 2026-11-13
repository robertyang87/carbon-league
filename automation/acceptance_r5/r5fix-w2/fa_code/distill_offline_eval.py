#!/usr/bin/env python3
"""distill_offline_eval -- offline fidelity of the distilled weights,
computed through the DEPLOYMENT path (NumPy forward on the exported
npz), on the whole-game validation split.

Slices (design 1.9 + shadow-DS additions): overall, by phase, by
game-state (favourable/contested/adverse), by worker type, the
step>=270 deep-endgame slice, and the tree-seize slice (planter
samples whose teacher destination is an enemy tree cell), including
the NN's implied seize rate vs the teacher's.  Confusion matrices
are dumped for the failure-attribution section of the report.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def np_forward(w, prefix, x):
    h = (x - w[prefix + "mean"]) / w[prefix + "std"]
    for i in (1, 2):
        h = h @ w[f"{prefix}W{i}"].T + w[f"{prefix}b{i}"]
        h = np.maximum(h, 0.0)
    return h @ w[prefix + "W3"].T + w[f"{prefix}b3"]


def predict(w, prefix, X, batch=65536):
    out = []
    for s in range(0, len(X), batch):
        out.append(np_forward(w, prefix,
                              X[s:s + batch].astype(np.float32)))
    return np.concatenate(out).argmax(axis=1)


def top1(pred, y, sel=None):
    if sel is None:
        sel = np.ones(len(y), dtype=bool)
    if not sel.any():
        return None
    return float((pred[sel] == y[sel]).mean())


def confusion(pred, y, ncls):
    cm = np.zeros((ncls, ncls), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def main():
    d = np.load(HERE / "bc_data" / "distill_dataset.npz")
    w = dict(np.load(HERE / "distilled_v1_weights.npz"))
    va = d["W_split"] == 1
    X, y = d["W_x"][va], d["W_y"][va]
    phase, state = d["W_phase"][va], d["W_state"][va]
    wtype, step = d["W_type"][va], d["W_step"][va]
    dest_tree, dest5 = d["W_dest_tree"][va], d["W_dest5"][va]
    pred = predict(w, "w_", X)
    m = {}
    m["worker_val_n"] = int(len(y))
    m["worker_top1"] = top1(pred, y)
    m["worker_top1_by_phase"] = {
        str(k): top1(pred, y, phase == k) for k in (0, 1, 2)}
    m["worker_top1_by_state"] = {
        str(k): top1(pred, y, state == k) for k in (-1, 0, 1)}
    m["worker_top1_by_type"] = {
        str(k): top1(pred, y, wtype == k) for k in (0, 1)}
    m["worker_top1_step_ge_270"] = top1(pred, y, step >= 270)
    m["worker_top1_step_200_269"] = top1(
        pred, y, (step >= 200) & (step < 270))
    seize = (wtype == 1) & (dest_tree == 2)
    m["seize_slice_n"] = int(seize.sum())
    m["seize_slice_top1"] = top1(pred, y, seize)
    planter = wtype == 1
    m["teacher_seize_rate_planter"] = float(
        (dest_tree[planter] == 2).mean())
    nn_dest = dest5[np.arange(len(y)), pred]
    m["nn_seize_rate_planter"] = float((nn_dest[planter] == 2).mean())
    m["worker_confusion"] = confusion(pred, y, 5).tolist()

    bva = d["B_split"] == 1
    BX, by = d["B_x"][bva], d["B_y"][bva]
    bpred = predict(w, "b_", BX)
    m["base_val_n"] = int(len(by))
    m["base_top1"] = top1(bpred, by)
    m["base_recall"] = {
        str(c): (float((bpred[by == c] == c).mean())
                 if (by == c).any() else None)
        for c in range(3)}
    m["base_confusion"] = confusion(bpred, by, 3).tolist()
    out = HERE / "distill_offline_metrics.json"
    out.write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
