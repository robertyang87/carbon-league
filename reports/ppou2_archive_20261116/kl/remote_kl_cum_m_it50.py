#!/usr/bin/env python3
"""ppou2 cumulative KL: KL(pi_ckpt20 || pi_start) on the Stage 1 rootset32.

Same caliber as /root/ppo_attrib/kl_measure.py (K0): per-slot categorical KL
summed over valid slots (y >= 0), float64, log_softmax of raw legal-masked
logits, teacher-forced on each root row's own teacher field. Direction:
KL(pi_test || pi_ref). No temperature.

test = /root/carbon/data/rl3/ppou2_m/ckpt_50.npz (main arm checkpoint it=30)
ref  = /root/carbon/data/weights/cnet2_m3s2r1L.npz (frozen start, m3s2r1)
Pipeline self-check: two independent forwards of the reference must give
KL(ref||ref) == 0.0 exactly, else the script stops before producing numbers.
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

TEST = "/root/carbon/data/rl3/ppou2_m/ckpt_50.npz"
REF = "/root/carbon/data/weights/cnet2_m3s2r1L.npz"


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


def forward(path):
    w = np.load(path)
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
# ---- self-check: two independent forwards of the reference
lg_r1 = forward(REF)
lg_r2 = forward(REF)
maxdiff = float(np.abs(lg_r1 - lg_r2).max())
self_kl = kl_rows(lg_r1, lg_r2, y)
print(f"self-check: max|logit diff|={maxdiff}  "
      f"KL(ref||ref) max={self_kl.max()} mean={self_kl.mean()}")
assert maxdiff == 0.0 and self_kl.max() == 0.0, "SELF-CHECK FAILED"

lg_t = forward(TEST)
ka = kl_rows(lg_t, lg_r1, y)
res = {"test": TEST, "ref": REF, "n_roots": int(N),
       "self_check_max_logit_diff": maxdiff,
       "kl_mean": float(ka.mean()),
       "kl_p99": float(np.percentile(ka, 99)),
       "kl_max": float(ka.max())}
print(json.dumps(res, indent=1))
json.dump(res, open("/root/ppo_attrib/kl_cum_ppou2_m_it50.json", "w"),
          indent=1)
print("DONE")
