#!/usr/bin/env python3
"""N2 baseline/fixed probe for cnet2_train.load() single-read binding.

Part 1 (generation FIELDS): monkeypatch T._resolve_load_source so the
field file 'scal.npy' in the certified generation dir is swapped AFTER
verification returns (the verify-then-reopen window), with different
finite values of the same shape/dtype. Baseline: load() consumes the
swapped bytes B although the manifest verified A. Fixed: load() must
refuse on its own sha256 check of the bytes it read.
Then (when load succeeds) a post-load on-disk swap tests whether the
consumed arrays track the disk (mmap) or stay resident.

Part 2 (aux bundle, BC_AUX_W=1): load a fresh clone, then flip the aux
label file on disk; check whether D['aux_die'] follows the disk.

Usage: probe_load_swap.py <run_dir> <tag> [--swap]
Evidence JSON on stdout (and the caller saves it).
"""
import hashlib
import io
import json
import os
import shutil
import sys

import numpy as np

RUN = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
SWAP = "--swap" in sys.argv
FIX = "/home/hatch/workspace/k0-codex-verify/fixture/ds_k0"
WORK = f"/home/hatch/workspace/n2-fix/evidence/load_{TAG}"
sys.path.insert(0, RUN)
os.environ.setdefault("BC_DEVICE", "cpu")

import cnet2_train as T  # noqa: E402


def clone(name):
    ds = os.path.join(WORK, name)
    if os.path.exists(ds):
        shutil.rmtree(ds)
    os.makedirs(WORK, exist_ok=True)
    shutil.copytree(FIX, ds)
    return ds


def gen_dir_of(ds):
    ptr = json.load(open(os.path.join(ds, "current.json")))
    return os.path.join(ds, ptr["dir"])


def arr_eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


out = {"tag": TAG, "swap": SWAP}

# ---------------- part 1: FIELDS swap-after-verify ----------------
ds = clone("ds_fields")
gen = gen_dir_of(ds)
man = json.load(open(os.path.join(gen, "manifest.json")))
scal_p = os.path.join(gen, "scal.npy")
A = np.load(scal_p)
assert hashlib.sha256(open(scal_p, "rb").read()).hexdigest() == man["fields"]["scal"]["sha256"]
B = (A * np.float16(0.5)).astype(A.dtype)
assert not arr_eq(A, B) and np.isfinite(B).all()
C = (A * np.float16(0.25)).astype(A.dtype)

if SWAP:
    orig_resolve = T._resolve_load_source

    def patched(dirn):
        src, m = orig_resolve(dirn)
        np.save(scal_p, B)          # swap AFTER verification, BEFORE load() reads
        return src, m

    T._resolve_load_source = patched

p1 = {}
try:
    D, is_val, meta = T.load(ds)
    p1["load_ok"] = True
    got = np.asarray(D["scal"])
    p1["consumed_eq_A"] = arr_eq(got, A)
    p1["consumed_eq_B"] = arr_eq(got, B)
    np.save(scal_p, C)              # post-load disk swap
    got2 = np.asarray(D["scal"])
    p1["after_disk_swap_eq_C"] = arr_eq(got2, C)
    p1["after_disk_swap_still_first"] = arr_eq(got2, got)
    p1["scal_is_memmap"] = isinstance(D["scal"], np.memmap)
except SystemExit as e:
    p1["load_ok"] = False
    p1["systemexit"] = str(e)
out["fields"] = p1

# ---------------- part 2: aux post-load disk swap ----------------
ds2 = clone("ds_aux")
os.environ["BC_AUX_W"] = "1"
p2 = {}
try:
    D2, _, _ = T.load(ds2)
    p2["load_ok"] = True
    before = np.asarray(D2["aux_die"]).copy()
    die_p = os.path.join(ds2, "aux_die.npy")
    cur = np.load(die_p)
    np.save(die_p, (1 - cur).astype(cur.dtype))
    after = np.asarray(D2["aux_die"])
    p2["aux_follows_disk_after_swap"] = not arr_eq(after, before)
    p2["aux_is_memmap"] = isinstance(D2["aux_die"], np.memmap)
except SystemExit as e:
    p2["load_ok"] = False
    p2["systemexit"] = str(e)
out["aux"] = p2

print(json.dumps(out, indent=1))
