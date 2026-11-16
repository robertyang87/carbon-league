"""Load-test the probe round-41 checkpoints (read-only)."""
import numpy as np
import torch

P = "/Users/arm/junyang/carbon/data/ppou2_probe_studio/out"
z = np.load(f"{P}/ckpt_41.npz")
print("NPZ_KEYS", sorted(z.files))
for k in sorted(z.files):
    a = z[k]
    if a.size <= 3:
        print(" ", k, a.shape, a)
    else:
        print(" ", k, a.shape, a.dtype)
st = torch.load(f"{P}/state_last.pt", map_location="cpu", weights_only=False)
print("STATE_TYPE", type(st))
if isinstance(st, dict):
    print("STATE_KEYS", sorted(st.keys()))
    print("STATE_IT", st.get("it"))
print("LOAD_OK")
