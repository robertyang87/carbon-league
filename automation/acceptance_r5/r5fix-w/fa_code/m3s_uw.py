"""m3s_uw.py DS -- planters and base slots learn from the teacher, our collectors do not (unit weight 0; BC_SELF_KL anchors
them to the initial net). Overwrites DS/uw.npy (keeps uw_build.npy)."""
import os, sys
import numpy as np
ds = sys.argv[1]
uw = np.load(os.path.join(ds, "uw.npy"))
if not os.path.exists(os.path.join(ds, "uw_build.npy")):
    np.save(os.path.join(ds, "uw_build.npy"), uw)
kind = np.asarray(np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r")[:, :10])
mask = np.asarray(np.load(os.path.join(ds, "tok_mask.npy"), mmap_mode="r")[:, :uw.shape[1]])
new = np.zeros_like(uw)
new[:, :10] = ((kind != 0) & mask[:, :10]).astype(np.float32)
new[:, 10:12] = mask[:, 10:12].astype(np.float32)
np.save(os.path.join(ds, "uw.npy"), new.astype(np.float32))
print({"rows": len(uw), "uw_shape": list(uw.shape), "planter_tokens": int(((kind == 1) & mask[:, :10]).sum()),
       "collector_tokens_zeroed": int(((kind == 0) & mask[:, :10]).sum()), "base_tokens": int(mask[:, 10:12].sum())})
