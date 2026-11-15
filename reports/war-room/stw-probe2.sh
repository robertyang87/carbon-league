#!/bin/bash
echo "PROBE2_BEGIN"
cd ~/junyang/carbon/code || exit 1
echo "--- trainer data-loading greps ---"
grep -n "np\.load\|meta\.json\|shards\|def load\|val\b" cnet2_train.py | head -50
echo "--- imports ---"
grep -n "^import \|^from " cnet2_train.py | head
echo "--- ds_m3s2 full listing ---"
ls ~/junyang/carbon/data/ds_m3s2
echo "--- shards dir? ---"
ls ~/junyang/carbon/data/ds_m3s2/shards 2>/dev/null | head -3 || echo "NO_SHARDS_DIR"
~/junyang/carbon/.venv/bin/python - <<'PYEOF'
import json, numpy as np, os
d = os.path.expanduser("~/junyang/carbon/data/ds_m3s2")
meta = json.load(open(os.path.join(d, "meta.json")))
games = meta["games"]
print("meta top keys:", list(meta.keys()))
print("games:", len(games), "sum n:", sum(g["n"] for g in games))
print("val games:", sum(1 for g in games if g.get("val")), "val n:", sum(g["n"] for g in games if g.get("val")))
print("train n:", sum(g["n"] for g in games if not g.get("val")))
for f in sorted(os.listdir(d)):
    if f.endswith(".npy"):
        a = np.load(os.path.join(d, f), mmap_mode="r")
        print("NPY", f, a.shape, a.dtype)
step = np.load(os.path.join(d, "step.npy")); gid = np.load(os.path.join(d, "gid.npy"))
print("step range", int(step.min()), int(step.max()), "gid range", int(gid.min()), int(gid.max()))
val_gi = sorted(g["gi"] for g in games if g.get("val"))
is_val = np.isin(gid, np.array(val_gi))
print("samples: val", int(is_val.sum()), "train", int((~is_val).sum()))
tr = ~is_val
eg = tr & (step >= 270)
print("train step>=270:", int(eg.sum()), "share of train:", round(float(eg.sum() / tr.sum()), 4))
print("val step>=270:", int((is_val & (step >= 270)).sum()))
PYEOF
echo "PROBE2_END"
