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
# F06 (integration): this script is an independent entry that can consume a w.npy NOT produced by the
# current awr_weights (pre-floor or hand-built), so it must not rely on the producer's floor: refuse
# non-finite input outright, and floor the inversion denominator at the same AWR_WMIN as awr_weights.
if not (np.isfinite(w).all() and np.isfinite(uw).all()):
    sys.exit(f"awr_collector_anchor: non-finite w/uw in dataset {ds!r} -- refusing to build an anchor from it")
w_floor = float(os.environ.get("AWR_WMIN", "1e-3"))
kind = np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r")[:, :10]; mask = np.load(os.path.join(ds, "tok_mask.npy"), mmap_mode="r")[:, :10]
col = (np.asarray(kind) == 0) & np.asarray(mask)
floored_rows = int(((w > 0) & (w < w_floor)).sum())
inv = np.divide(scale * w.mean(), np.maximum(w, w_floor), out=np.zeros_like(w), where=w > 0)   # w = 0 rows (dropped mirror games) stay out
uw[:, :10] = np.where(col, inv[:, None], uw[:, :10])
if not np.isfinite(uw).all():
    sys.exit(f"awr_collector_anchor: non-finite uw produced for dataset {ds!r} -- not saving")
np.save(os.path.join(out, "uw.npy"), uw.astype(np.float32))
print({"rows": len(w), "collector_tokens": int(col.sum()), "per_row": round(float(col.sum() / len(w)), 2), "w_mean": round(float(w.mean()), 3), "scale": scale, "zero_w_rows": int((w == 0).sum()), "finite": bool(np.isfinite(uw).all()), "w_floor": w_floor, "floored_rows": floored_rows})
