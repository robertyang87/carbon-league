"""X04(a) probe: load_annotated must return the arrays verify_generation
actually verified -- not a second, unverified by-path read.

Two checks against the real product gen-1dcc9611575bfbe3 (out1):
  1. count np.load calls per field file during load_annotated
     (verify loads each field once; a discarding load_annotated loads
     every field a second time);
  2. race: swap q.npy on disk AFTER verify_generation returns but
     BEFORE load_annotated's own reads (interposed by wrapping the
     module's verify_generation). The consumer must still receive the
     verified values.

Usage: python3 probe_x04a.py RUN_DIR
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np  # noqa: E402
import k0_annotate as A  # noqa: E402

SRC = "/home/hatch/workspace/k0-round2/annotate/fixture/out1"
WORK = "/tmp/x04a_probe"
shutil.rmtree(WORK, ignore_errors=True)
shutil.copytree(SRC, os.path.join(WORK, "out1"))
OUT = os.path.join(WORK, "out1")
GEN = os.path.join(OUT, "generations", "gen-1dcc9611575bfbe3")

# --- check 1: load counting -------------------------------------------------
counts = {}
real_load = np.load


def counting_load(file, *a, **kw):
    if isinstance(file, (str, os.PathLike)):
        counts[os.path.basename(str(file))] = \
            counts.get(os.path.basename(str(file)), 0) + 1
    return real_load(file, *a, **kw)


np.load = counting_load
try:
    meta, arrs = A.load_annotated(OUT)
finally:
    np.load = real_load
field_loads = {k: v for k, v in counts.items() if k.endswith(".npy")}
print("np.load counts per field file:", dict(sorted(field_loads.items())))
print("max loads of one field:", max(field_loads.values()))

# --- check 2: post-verify swap ----------------------------------------------
orig_verify = A.verify_generation
qpath = os.path.join(GEN, "q.npy")
q_orig = np.load(qpath).copy()


def swapping_verify(gen_dir, ptr, *a, **kw):
    out = orig_verify(gen_dir, ptr, *a, **kw)
    evil = q_orig.copy()
    evil[0, 0] = 0.123456789  # different but plausible values
    np.save(qpath, evil)
    return out


A.verify_generation = swapping_verify
try:
    meta2, arrs2 = A.load_annotated(OUT)
finally:
    A.verify_generation = orig_verify
got = float(arrs2["q"][0, 0])
print(f"q[0,0] verified value: {float(q_orig[0, 0])!r}; "
      f"consumer received: {got!r}; "
      f"received_verified={got == float(q_orig[0, 0])}")
