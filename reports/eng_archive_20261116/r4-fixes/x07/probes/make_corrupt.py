"""Build corrupted copies of the canonical heads-v2 weights for the
X07 repro: several tensors (spread over stem / trunk / attention /
heads) wholly replaced by NaN (variant nan) or +Inf (variant inf).
Tensor inventory and all other tensors are byte-identical to the
canonical file."""
import numpy as np

SRC = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
TARGETS = ["stem.weight", "blocks.3.c1.weight", "tattn.1.qkv.weight",
           "value.weight", "whead.weight"]

d = dict(np.load(SRC))
for tag, bad in (("nan", np.nan), ("inf", np.inf)):
    out = dict(d)
    for t in TARGETS:
        a = np.array(out[t], dtype=np.float32)
        a[...] = bad
        out[t] = a
    p = f"/home/hatch/workspace/r4-fixes/x07/fixtures/corrupt_{tag}.npz"
    np.savez(p, **out)
    chk = dict(np.load(p))
    badn = {t: int((~np.isfinite(chk[t])).sum()) for t in TARGETS}
    print(f"corrupt_{tag}.npz written; non-finite counts: {badn}; "
          f"total tensors: {len(chk)}")
