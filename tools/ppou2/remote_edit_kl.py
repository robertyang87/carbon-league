#!/usr/bin/env python3
"""Edit kl_coef inside a trainer state file (state_last.pt) and verify.
Usage: remote_edit_kl.py <state_path> <new_kl> [bak_path]
Only the top-level 'kl_coef' entry is changed. Before editing, the
original file is copied byte-for-byte to bak_path when given and not
already present (an existing backup is never overwritten); after saving,
the file is reloaded and kl_coef is read back, and every other top-level
entry is compared against the backup (tensors by torch.equal).
Fix 2026-11-15 (strategy-group fix-on-touch): the script used to expect
the caller to pre-create the backup; a missing bak path crashed the
post-edit deep-compare after the edit had already landed.
"""
import os, shutil, sys, torch


def deep_equal(a, b):
    # dict.__eq__ truth-tests tensor values and raises on multi-element
    # tensors, so compare nested structures recursively.
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


path, new_kl = sys.argv[1], float(sys.argv[2])
bak = sys.argv[3] if len(sys.argv) > 3 else None

st = torch.load(path, map_location="cpu", weights_only=False)
print("before kl_coef:", st.get("kl_coef"), "it:", st.get("it"))
if bak and not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print("backup created:", bak)
st["kl_coef"] = new_kl
torch.save(st, path)
st2 = torch.load(path, map_location="cpu", weights_only=False)
print("after kl_coef:", st2.get("kl_coef"), "it:", st2.get("it"))
assert st2["kl_coef"] == new_kl, "read-back mismatch"
if bak:
    old = torch.load(bak, map_location="cpu", weights_only=False)
    assert set(old) == set(st2), f"key set changed: {set(old) ^ set(st2)}"
    diff = [k for k in old if k != "kl_coef" and not deep_equal(old[k], st2[k])]
    print("other-keys identical:", not diff, ("differing: " + str(diff)) if diff else "")
    assert not diff, f"non-kl entries changed: {diff}"
print("EDIT_OK", path, new_kl)
