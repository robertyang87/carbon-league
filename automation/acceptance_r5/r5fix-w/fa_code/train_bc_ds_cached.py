"""train_bc_ds -- behaviour-cloning training for the Shadow DeepSeek policy (fork of train_bc.py; val split = bc_data/deepseek_val_files.json: 2 RL-policy-1 + 2 RL-light-1 + 3 NN-light games).

Data: bc_data/claude_family.npz (23 Claude-family games, 6,877 decision
points). Features come from bc_features.featurize ONLY -- the player's
own visibility-filtered carbon plus the shared, fully-visible players
block. full_carbon is never touched.

Split is BY GAME (no same-game leakage): 18 train games / 5 val games,
the val set holding one AlphaCarbon-2 game, one AlphaCarbon game, and
three opus-variant games.

Trains two small MLPs (worker action 5-way, base recruit 3-way) with
PyTorch on CPU (2 threads -- a rule-line job shares this box), exports
weights to shadow_weights.npz and generates shadow_alpha.py with the
bc_features source embedded verbatim.

Run: venv/bin/python train_bc.py
"""

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from bc_features import featurize, WORKER_DIM, BASE_DIM  # noqa: E402

VAL_FILES = set(json.load(open(os.path.join(
    HERE, "bc_data", "deepseek_val_files.json"))))


def build_dataset(d):
    """Cache-backed replacement: featurization done once by
    featurize_cache_ds.py (verified bitwise-equal to the original
    build_dataset). Cache path from env BC_CACHE."""
    c = np.load(os.environ["BC_CACHE"], allow_pickle=True)
    print(f"loaded featurization cache {os.environ['BC_CACHE']}")
    return (c["Xw"], c["yw"], c["tw"], c["gw"],
            c["Xb"], c["yb"], c["gb"])


def train_mlp(Xtr, ytr, Xva, yva, out_dim, hidden, epochs, lr, wd,
              class_w, seed=0, patience=8):
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    mean = Xtr.mean(0)
    std = Xtr.std(0) + 1e-6
    xtr = torch.tensor((Xtr - mean) / std)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    xva = torch.tensor((Xva - mean) / std)
    yva_t = torch.tensor(yva, dtype=torch.long)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, out_dim))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor(class_w, dtype=torch.float32))
    bs = 1024
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for k in range(0, len(xtr), bs):
            idx = perm[k:k + bs]
            opt.zero_grad()
            loss = lossf(net(xtr[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xva).argmax(1) == yva_t).float().mean().item()
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep + 1}")
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(xva).argmax(1).numpy()
    weights = {}
    linears = [m for m in net if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linears):
        weights[f"W{i + 1}"] = lin.weight.detach().numpy().astype(np.float32)
        weights[f"b{i + 1}"] = lin.bias.detach().numpy().astype(np.float32)
    weights["mean"] = mean.astype(np.float32)
    weights["std"] = std.astype(np.float32)
    return weights, pred, best_acc


NPZ_PATH = os.environ.get(
    "BC_NPZ", os.path.join(HERE, "bc_data", "claude_family.npz"))
WEIGHTS_OUT = os.environ.get("BC_WEIGHTS", "shadow_weights.npz")
METRICS_OUT = os.environ.get(
    "BC_METRICS", WEIGHTS_OUT.replace("_weights.npz", "_train_metrics.json"))


def main():
    _z = np.load(NPZ_PATH, allow_pickle=True)
    # Materialize every field ONCE: NpzFile.__getitem__ re-decompresses
    # the whole array from the zip on every access, which made the
    # per-sample loop in build_dataset ~40x slower than featurize itself.
    d = {k: _z[k] for k in _z.files}
    Xw, yw, tw, gw, Xb, yb, gb = build_dataset(d)
    print(f"worker samples: {len(yw)} (train {np.sum(~gw)}, val {np.sum(gw)})")
    print(f"base samples:   {len(yb)} (train {np.sum(~gb)}, val {np.sum(gb)})")

    metrics = {}

    # ---- worker policy ----
    counts = np.bincount(yw[~gw], minlength=5).astype(np.float64)
    cw = (len(yw[~gw]) / (5 * counts)) ** 0.5
    print("worker class weights:", np.round(cw, 2))
    w_weights, wpred, wacc = train_mlp(
        Xw[~gw], yw[~gw], Xw[gw], yw[gw], 5, 192, 40, 1e-3, 1e-4, cw)
    print(f"worker val top-1: {wacc:.4f}")
    for tname, tval in (("collector", 0), ("planter", 1)):
        m = tw[gw] == tval
        acc = float(np.mean(wpred[m] == yw[gw][m]))
        stay = float(np.mean(yw[gw][m] == 0))
        print(f"  {tname}: acc {acc:.4f}  stay-baseline {stay:.4f}  "
              f"random 0.2000  n={int(np.sum(m))}")
        metrics[f"worker_{tname}_acc"] = acc
        metrics[f"worker_{tname}_stay_baseline"] = stay
    metrics["worker_val_acc"] = float(wacc)
    metrics["worker_val_stay_baseline"] = float(np.mean(yw[gw] == 0))

    # ---- base policy ----
    counts_b = np.bincount(yb[~gb], minlength=3).astype(np.float64)
    cwb = np.minimum(len(yb[~gb]) / (3 * counts_b), 13.0)
    print("base class weights:", np.round(cwb, 2),
          "train class counts:", counts_b)
    b_weights, bpred, bacc = train_mlp(
        Xb[~gb], yb[~gb], Xb[gb], yb[gb], 3, 96, 40, 1e-3, 1e-4, cwb,
        seed=1)
    print(f"base val top-1: {bacc:.4f}  always-none baseline: "
          f"{float(np.mean(yb[gb] == 0)):.4f}")
    for cname, cval in (("RECCOLLECTOR", 1), ("RECPLANTER", 2)):
        m = yb[gb] == cval
        rec = float(np.mean(bpred[m] == cval)) if np.sum(m) else float("nan")
        print(f"  {cname}: recall {rec:.4f}  n={int(np.sum(m))}")
        metrics[f"base_{cname}_recall"] = rec
    metrics["base_val_acc"] = float(bacc)
    metrics["base_val_none_baseline"] = float(np.mean(yb[gb] == 0))
    metrics["base_val_pred_positive_rate"] = float(np.mean(bpred > 0))
    metrics["base_val_true_positive_rate"] = float(np.mean(yb[gb] > 0))

    out = {}
    for k, v in w_weights.items():
        out[f"w_{k}"] = v
    for k, v in b_weights.items():
        out[f"b_{k}"] = v
    np.savez(os.path.join(HERE, WEIGHTS_OUT), **out)
    with open(os.path.join(HERE, METRICS_OUT), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"saved {WEIGHTS_OUT} + {METRICS_OUT}")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
