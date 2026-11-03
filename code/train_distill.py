#!/usr/bin/env python3
"""train_distill -- Case-B stage 1: distill base_v53 into two MLPs
(worker head 247->256->256->5, base head 108->128->128->3).

Fixed recipe (design section 1.6): AdamW lr 1e-3 cosine, class weights
= sqrt inverse frequency on the train split, worker 20 epochs /
base 30 epochs, batch 8192 / 4096.  Early stop on VALIDATION loss
(patience 5, whole-game split, ~10.8k val points) is the only
selection -- no game-based checkpoint picking (RL stage-1 lesson 3).

Run with the venv python (torch).  Threads capped at 2.
Outputs: distilled_v1_weights.npz (rl_common.np_forward layout:
mean/std + W1,b1,W2,b2,W3,b3 with W stored [out,in]) and
distill_train_metrics.json (torch-side val fidelity; the headline
offline numbers are recomputed through the NumPy deploy path by
distill_offline_eval.py).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS",
                      os.environ.get("DISTILL_THREADS", "4"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("DISTILL_THREADS", "4")))
HERE = Path(__file__).resolve().parent
DATA = HERE / "bc_data" / "distill_dataset.npz"
WEIGHTS_OUT = HERE / "distilled_v1_weights.npz"
METRICS_OUT = HERE / "distill_train_metrics.json"
SEED = 20261103


def class_weights(y, ncls, power=0.5):
    cnt = np.bincount(y, minlength=ncls).astype(np.float64)
    w = 1.0 / np.power(np.maximum(cnt, 1.0), power)
    return (w / w.mean()).astype(np.float32), cnt


def train_head(name, X, y, split, in_dim, hidden, ncls, epochs, batch,
               meta_slices=None, cw_power=0.5):
    torch.manual_seed(SEED)
    tr = split == 0
    va = split == 1
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    # Standardize with TRAIN-split stats -- the deploy path
    # (rl_common.np_forward layout) applies (x-mean)/std before the
    # first Linear, so training MUST see the same standardized
    # inputs.  Segment 1/2a skipped this: torch val looked healthy
    # (0.698) while the deployed net was a distorted copy (NumPy
    # path 0.605, base RECC recall 0.85 -> 0.40) and every game
    # collapsed.  The returned stats are what the exporter writes.
    mean = Xtr.astype(np.float32).mean(axis=0)
    std = Xtr.astype(np.float32).std(axis=0)
    std[std < 1e-6] = 1.0
    Xtr = (Xtr.astype(np.float32) - mean) / std
    Xva = (Xva.astype(np.float32) - mean) / std
    cw, cnt = class_weights(ytr, ncls, power=cw_power)
    print(f"[{name}] train={len(ytr)} val={len(yva)} "
          f"label counts={cnt.tolist()} cw={cw.tolist()}", flush=True)
    net = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, ncls))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw), reduction="none")
    best = (float("inf"), None, -1)
    bad = 0
    start_ep = 0
    # Durable checkpointing: save after every epoch (atomic rename);
    # on restart, resume exactly where the process died (VM reboots
    # are a fact of life on this box -- user directive 2026-11-03:
    # full epochs, no truncation to save time).
    ckpt_path = HERE / f"distill_ckpt_{name}.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best = (ck["best_loss"], ck["best_sd"], ck["best_ep"])
        bad = ck["bad"]
        start_ep = ck["next_ep"]
        print(f"[{name}] resumed from checkpoint at ep{start_ep} "
              f"(best ep{best[2]} val_ce={best[0]:.4f})", flush=True)
    ntr = len(ytr)
    for ep in range(start_ep, epochs):
        net.train()
        perm = np.random.default_rng(SEED + ep).permutation(ntr)
        tot = 0.0
        for s in range(0, ntr, batch):
            idx = perm[s:s + batch]
            xb = torch.tensor(Xtr[idx].astype(np.float32))
            yb = torch.tensor(ytr[idx])
            logits = net(xb)
            loss = lossf(logits, yb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            vtot, vacc = 0.0, 0
            for s in range(0, len(yva), 16384):
                xb = torch.tensor(Xva[s:s + 16384].astype(np.float32))
                yb = torch.tensor(yva[s:s + 16384])
                logits = net(xb)
                vtot += float(lossf(logits, yb).sum())
                vacc += int((logits.argmax(1) == yb).sum())
        vloss = vtot / len(yva)
        vacc = vacc / len(yva)
        print(f"[{name}] ep{ep} train_ce={tot / ntr:.4f} "
              f"val_ce={vloss:.4f} val_top1={vacc:.4f}", flush=True)
        if vloss < best[0] - 1e-4:
            best = (vloss, {k: v.detach().clone()
                            for k, v in net.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "best_loss": best[0],
                    "best_sd": best[1], "best_ep": best[2], "bad": bad,
                    "next_ep": ep + 1}, tmp)
        os.replace(tmp, ckpt_path)
        if bad >= 5:
            print(f"[{name}] early stop at ep{ep} "
                  f"(best ep{best[2]})", flush=True)
            break
    if ckpt_path.exists():
        ckpt_path.unlink()
    net.load_state_dict(best[1])
    net.eval()
    metrics = {"best_epoch": best[2], "val_ce": best[0]}
    # detailed val metrics with the best weights
    with torch.no_grad():
        preds = []
        for s in range(0, len(yva), 16384):
            xb = torch.tensor(Xva[s:s + 16384].astype(np.float32))
            preds.append(net(xb).argmax(1).numpy())
    pred = np.concatenate(preds)
    metrics["val_top1"] = float((pred == yva).mean())
    if meta_slices:
        for sname, sarr in meta_slices.items():
            sva = sarr[va]
            m = {}
            for v in sorted(set(sva.tolist())):
                sel = sva == v
                m[str(v)] = float((pred[sel] == yva[sel]).mean())
            metrics[f"val_top1_by_{sname}"] = m
    if name == "base":
        rec = {}
        for c in range(ncls):
            sel = yva == c
            rec[str(c)] = float((pred[sel] == c).mean()) \
                if sel.any() else None
        metrics["val_recall_per_class"] = rec
    sd = net.state_dict()
    params = {
        "W1": sd["0.weight"].numpy(), "b1": sd["0.bias"].numpy(),
        "W2": sd["2.weight"].numpy(), "b2": sd["2.bias"].numpy(),
        "W3": sd["4.weight"].numpy(), "b3": sd["4.bias"].numpy(),
        "mean": mean.astype(np.float32), "std": std.astype(np.float32),
    }
    return params, metrics


def main():
    t0 = time.time()
    d = np.load(DATA)
    out = {}
    metrics = {}
    wp, wm = train_head(
        "worker", d["W_x"], d["W_y"], d["W_split"], 247, 256, 5,
        epochs=20, batch=8192,
        meta_slices={"phase": d["W_phase"], "state": d["W_state"],
                     "type": d["W_type"]})
    for k, v in wp.items():
        out["w_" + k] = v
    # w_mean/w_std come from train_head (train-split stats)
    metrics["worker"] = wm
    bp, bm = train_head(
        "base", d["B_x"], d["B_y"], d["B_split"], 108, 128, 3,
        epochs=30, batch=4096)
    for k, v in bp.items():
        out["b_" + k] = v
    # b_mean/b_std come from train_head (train-split stats)
    metrics["base"] = bm
    np.savez(WEIGHTS_OUT, **out)
    size = os.path.getsize(WEIGHTS_OUT)
    metrics["weights_bytes"] = size
    metrics["wall_sec"] = round(time.time() - t0, 1)
    METRICS_OUT.write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    print(f"wrote {WEIGHTS_OUT} ({size} bytes)")


if __name__ == "__main__":
    main()
