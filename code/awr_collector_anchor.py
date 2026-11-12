"""awr_collector_anchor DS OUT [SCALE] -- copy of an awr_weights dataset in which our collectors keep their own moves.

In whole-game AWR only planters and base slots are sampled, so awr_weights gives our collectors unit weight 0. They then
get no gradient at all, yet share the decoder with the units being trained, and drift (first round 2026-11-11:
collector accuracy 0.992 -> 0.930). Here collector tokens get unit weight SCALE * mean(w) / w[row], so their loss weight
is a flat SCALE * mean(w), independent of the game's advantage: plain self-distillation of the net's own greedy car moves,
an anchor built from the data itself. OUT links every file of DS except uw.npy."""
import os, sys
import numpy as np
ds, out = sys.argv[1], sys.argv[2]; scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
os.makedirs(out, exist_ok=True)
for f in os.listdir(ds):
    if f != "uw.npy" and not os.path.exists(os.path.join(out, f)):
        os.symlink(os.path.abspath(os.path.join(ds, f)), os.path.join(out, f))
w = np.load(os.path.join(ds, "w.npy")); uw = np.load(os.path.join(ds, "uw.npy"))
kind = np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r")[:, :10]; mask = np.load(os.path.join(ds, "tok_mask.npy"), mmap_mode="r")[:, :10]
col = (np.asarray(kind) == 0) & np.asarray(mask)
inv = np.divide(scale * w.mean(), w, out=np.zeros_like(w), where=w > 0)   # w = 0 rows (dropped mirror games) stay out
uw[:, :10] = np.where(col, inv[:, None], uw[:, :10])
np.save(os.path.join(out, "uw.npy"), uw.astype(np.float32))
print({"rows": len(w), "collector_tokens": int(col.sum()), "per_row": round(float(col.sum() / len(w)), 2), "w_mean": round(float(w.mean()), 3), "scale": scale, "zero_w_rows": int((w == 0).sum()), "finite": bool(np.isfinite(uw).all())})
