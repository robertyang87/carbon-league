"""Verify ppou2 anchor state actor == init (warmup must not move the policy).

Compares state_anchor_frozen.pt's net state_dict against the init npz
(cnet2_m3s2r1L.npz) key by key. npz keys are model state_dict keys (export()
writes state_dict items verbatim) plus 'meta'.
Usage on L20: python3 check_anchor_equal.py <anchor.pt> <init.npz>
Prints max abs diff per key group and exits nonzero on any mismatch.
"""
import sys
import numpy as np
import torch

anchor_path, init_path = sys.argv[1], sys.argv[2]
stt = torch.load(anchor_path, map_location="cpu")
net_sd = stt["net"]
w0 = np.load(init_path)
keys = [k for k in w0.files if k != "meta"]
missing = [k for k in keys if k not in net_sd]
extra = [k for k in net_sd if k not in keys and k != "meta"]
worst = 0.0
worst_key = None
for k in keys:
    if k in missing:
        continue
    a = net_sd[k].detach().cpu().float().numpy()
    b = w0[k].astype(np.float32)
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH {k}: state {a.shape} vs init {b.shape}")
        sys.exit(2)
    d = float(np.abs(a - b).max()) if a.size else 0.0
    if d > worst:
        worst, worst_key = d, k
print(f"compared {len(keys)} keys; missing_in_state={missing}; extra_in_state={extra}")
print(f"max_abs_diff={worst} at {worst_key}")
print("ACTOR_EQUALS_INIT" if worst == 0.0 and not missing else "ACTOR_DIFFERS_FROM_INIT")
sys.exit(0 if worst == 0.0 and not missing else 1)
