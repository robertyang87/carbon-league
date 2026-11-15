#!/bin/bash
echo "BUILD_BEGIN"
ls -la ~/junyang/carbon/data/ds_m3s2 | head -3
~/junyang/carbon/.venv/bin/python - <<'PYEOF'
import json, numpy as np, os, shutil, hashlib, sys
H = os.path.expanduser("~")
src = os.path.join(H, "junyang/carbon/data/ds_m3s2")
dst = os.path.join(H, "junyang/carbon/data/ds_m3s2_w")
if os.path.exists(dst):
    print("ABORT: ds_m3s2_w already exists"); sys.exit(3)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

src_meta_md5_before = md5(os.path.join(src, "meta.json"))
src_step_md5_before = md5(os.path.join(src, "step.npy"))

meta = json.load(open(os.path.join(src, "meta.json")))
val_g = np.array([g["val"] for g in meta["games"]], dtype=bool)
gid = np.load(os.path.join(src, "gid.npy"))
step = np.load(os.path.join(src, "step.npy"))
is_val = val_g[gid]
N = len(gid)
eg_idx = np.nonzero((~is_val) & (step >= 270))[0]
extra = np.concatenate([eg_idx, eg_idx])  # 2 extra copies -> 3 total
print("orig: N", N, "train", int((~is_val).sum()), "val", int(is_val.sum()),
      "train_endgame", len(eg_idx), "share", round(len(eg_idx) / float((~is_val).sum()), 4))

shutil.copytree(src, dst)

files = sorted(f for f in os.listdir(src) if f.endswith(".npy"))
for f in files:
    a = np.load(os.path.join(src, f))
    assert a.shape[0] == N, (f, a.shape)
    b = np.concatenate([a, a[extra]], axis=0)
    np.save(os.path.join(dst, f), b)
    del a, b
print("arrays duplicated:", len(files))

# update meta total in the COPY only
mpath = os.path.join(dst, "meta.json")
m2 = json.load(open(mpath))
m2["total"] = N + len(extra)
json.dump(m2, open(mpath, "w"))

# ---- verification ----
gid2 = np.load(os.path.join(dst, "gid.npy")); step2 = np.load(os.path.join(dst, "step.npy"))
is_val2 = val_g[gid2]
N2 = len(gid2)
assert N2 == N + len(extra)
assert np.array_equal(gid2[:N], gid) and np.array_equal(step2[:N], step)
assert int(is_val2.sum()) == int(is_val.sum()) == 3289, "val count changed"
assert not is_val2[N:].any(), "appended rows must all be train"
for f in ["maps.npy", "y.npy", "uw.npy", "mf.npy"]:
    a = np.load(os.path.join(src, f)); b = np.load(os.path.join(dst, f))
    assert np.array_equal(b[:N], a), f"prefix mismatch {f}"
    va = a[np.nonzero(is_val)[0]]; vb = b[np.nonzero(is_val2)[0]]
    assert np.array_equal(va, vb), f"val rows mismatch {f}"
eg2 = (~is_val2) & (step2 >= 270)
print("new: N", N2, "train", int((~is_val2).sum()), "val", int(is_val2.sum()),
      "train_endgame", int(eg2.sum()), "share", round(int(eg2.sum()) / float((~is_val2).sum()), 4))
print("src meta md5 same:", md5(os.path.join(src, "meta.json")) == src_meta_md5_before,
      "| src step md5 same:", md5(os.path.join(src, "step.npy")) == src_step_md5_before)
print("BUILD_OK")
PYEOF
du -sh ~/junyang/carbon/data/ds_m3s2_w
echo "BUILD_END"
