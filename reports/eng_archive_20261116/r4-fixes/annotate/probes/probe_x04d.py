"""X04(d) probe: cnet2_exit_train consumption must eat the bytes its
verification certified -- not a later by-path re-read (mmap).

Part 1 (sealed): consume the d1 sealed fixture through
load_exit(resolve_exit_arg(root)) while a wrapper swaps value.npy on
disk AFTER _verify_exit_generation has certified it. Compare:
  - the consumed array's first value vs the certified original;
  - load_exit's recorded identity sha for 'value' vs the manifest's
    declared sha256 (record A / run B shows up as a mismatch).
Part 2 (legacy flat): same trick one level down -- swap value.npy
after load_exit's own _sha256_path hashed it; check whether the
consumed bytes are the ones the identity record describes.

Usage: ~/.venvs/k0/bin/python probe_x04d.py RUN_DIR
"""
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np  # noqa: E402
import cnet2_exit_train as E  # noqa: E402

SRC = "/tmp/x04d_fixture_src"
WORK = "/tmp/x04d_probe"
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------- part 1: sealed --------------------------------------------
root1 = os.path.join(WORK, "ds")
shutil.copytree(SRC, root1)
ptr = json.load(open(os.path.join(root1, "current.json")))
gen_dir = os.path.join(root1, ptr["dir"])
man = json.load(open(os.path.join(gen_dir, "manifest.json")))
vfile = os.path.join(gen_dir, "value.npy")
v_orig = np.load(vfile).copy()
man_sha = man["fields"]["value"]["sha256"]

orig_verify = E._verify_exit_generation
swapped = {"done": False}


def swapping_verify(gd, p):
    out = orig_verify(gd, p)
    evil = v_orig + np.float32(1000.0)
    np.save(vfile, evil)
    swapped["done"] = True
    return out


E._verify_exit_generation = swapping_verify
try:
    res = E.resolve_exit_arg(root1)
finally:
    E._verify_exit_generation = orig_verify
D, is_val, meta = E.load_exit(res)
got = float(np.asarray(D["value"]).flat[0])
want = float(v_orig.flat[0])
print(f"[sealed] swap planted: {swapped['done']}; value[0] certified="
      f"{want!r} consumed={got!r} consumed_certified={got == want}")

# ---------------- part 2: legacy flat ---------------------------------------
flat = os.path.join(WORK, "flat")
os.makedirs(flat)
for fn in os.listdir(gen_dir):
    if fn != "manifest.json":
        shutil.copy(os.path.join(gen_dir, fn), os.path.join(flat, fn))
# restore the pristine value.npy (part 1 swapped the copy in gen_dir of
# root1 only; flat was copied AFTER the swap -> re-copy from the source)
shutil.copy(os.path.join(SRC, ptr["dir"], "value.npy"),
            os.path.join(flat, "value.npy"))
v_orig2 = np.load(os.path.join(flat, "value.npy")).copy()
orig_sha = sha(open(os.path.join(flat, "value.npy"), "rb").read())

orig_shapath = E._sha256_path
state = {"done": False}


def swapping_sha(p):
    h = orig_shapath(p)
    if p.endswith("value.npy") and not state["done"]:
        evil = v_orig2 + np.float32(1000.0)
        np.save(p, evil)
        state["done"] = True
    return h


E._sha256_path = swapping_sha
try:
    D2, _, _ = E.load_exit(flat)
finally:
    E._sha256_path = orig_shapath
got2 = float(np.asarray(D2["value"]).flat[0])
want2 = float(v_orig2.flat[0])
print(f"[legacy] swap planted: {state['done']}; value[0] hashed="
      f"{want2!r} consumed={got2!r} consumed_hashed={got2 == want2}")
print(f"[legacy] identity sha of original value.npy: {orig_sha[:16]}..")
