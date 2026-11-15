#!/usr/bin/env python3
"""Stage 1 deliverable C, part 2: KL trajectory on L20 (offline only).

Caliber (PREREG + K0 joint_logp_kl): per-slot categorical KL summed
over valid slots (y >= 0), computed in float64 from log_softmax of
the model's raw (legal-masked) logits, teacher-forced on each root
row's recorded action prefix y. Direction: KL(pi_test || pi_ref).
No temperature (the K0 screen caliber).

Models are built through the repo's own adapter path
(rl2_ppo_cnet2.Cnet2Adapter.make_model) from the copied ckpt npz
files in /root/ppo_attrib/state. Batch assembly follows
Cnet2Buffer.batch's dtype/reshape rules, except "teacher" is taken
from each root row's own field (the K0 root-row caliber; the
trainer's constant-teacher batch is for learner rollout rows).

Pipeline self-check (PREREG): an independent second forward of the
anchor must give KL(pi_anchor || pi_anchor) == 0.0 exactly, else the
script stops before producing numbers.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root/carbon/code")
WORK = "/root/ppo_attrib"
MB = 512

z = np.load(os.path.join(WORK, "roots.npz"), allow_pickle=True)
N = len(z["y"])
print("roots:", N)

import torch  # noqa: E402
from rl2_ppo_cnet2 import Cnet2Adapter  # noqa: E402
import cnet2_feat as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)


def batch(idx):
    d = {}
    for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
              "tok_cargo", "hist_cell", "hist_cargo", "hist_move",
              "uf", "mf", "legal", "y", "teacher"):
        a = z[k][idx]
        if a.dtype == np.float16:
            a = a.astype(np.float32)
        d[k] = torch.from_numpy(np.ascontiguousarray(a))
    B = len(idx)
    d["maps"] = (d["maps"].float() / 255.0).reshape(B, F.P, F.S, F.S)
    for k in ("tok_pos", "tok_kind", "hist_cell", "hist_move", "y",
              "teacher"):
        d[k] = d[k].long()
    d["tok_mask"] = d["tok_mask"].bool()
    d["legal"] = d["legal"].bool()
    return {k: v.to(DEV) for k, v in d.items()}


def forward(ckpt):
    w = np.load(os.path.join(WORK, "state", f"ckpt_{ckpt}.npz"))
    weights = {k: w[k] for k in w.keys()}
    net = Cnet2Adapter().make_model(weights)
    net = net.to(DEV).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, N, MB):
            idx = np.arange(i, min(i + MB, N))
            lg, _v, _a = net(batch(idx))
            outs.append(lg.float().cpu().numpy())
    del net
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outs)


def kl_rows(lg_test, lg_ref, y):
    """per-root KL(pi_test || pi_ref), float64, K0 caliber."""
    lt = torch.from_numpy(lg_test).double()
    lr = torch.from_numpy(lg_ref).double()
    logp = torch.log_softmax(lt, dim=-1)
    logq = torch.log_softmax(lr, dim=-1)
    valid = torch.from_numpy(y >= 0).double()
    kl_slot = (logp.exp() * (logp - logq)).sum(dim=-1)
    return (kl_slot * valid).sum(dim=1).numpy()


y = z["y"]
# ---- self-check: two independent forwards of the anchor
lg_a1 = forward(74)
lg_a2 = forward(74)
maxdiff = float(np.abs(lg_a1 - lg_a2).max())
self_kl = kl_rows(lg_a1, lg_a2, y)
print(f"self-check: max|logit diff|={maxdiff}  "
      f"KL(anchor||anchor) max={self_kl.max()} mean={self_kl.mean()}")
assert maxdiff == 0.0 and self_kl.max() == 0.0, "SELF-CHECK FAILED"
np.save(os.path.join(WORK, "kl", "logits_74.npy"), lg_a1)

results = {"n_roots": int(N), "self_check_max_logit_diff": maxdiff,
           "anchor": {}, "segments": {}}
prev = lg_a1
for k in range(75, 97):
    lg = forward(k)
    np.save(os.path.join(WORK, "kl", f"logits_{k}.npy"), lg)
    ka = kl_rows(lg, lg_a1, y)
    results["anchor"][str(k)] = {
        "mean": float(ka.mean()), "p99": float(np.percentile(ka, 99)),
        "max": float(ka.max())}
    ks = kl_rows(lg, prev, y)
    results["segments"][f"{k-1}->{k}"] = {
        "mean": float(ks.mean()), "p99": float(np.percentile(ks, 99)),
        "max": float(ks.max())}
    print(f"ckpt {k}: KL vs anchor mean={ka.mean():.6f} "
          f"p99={np.percentile(ka, 99):.6f} | seg {k-1}->{k} "
          f"mean={ks.mean():.6f} p99={np.percentile(ks, 99):.6f}")
    prev = lg

json.dump(results, open(os.path.join(WORK, "kl", "kl_results.json"), "w"),
          indent=1)
print("DONE")
