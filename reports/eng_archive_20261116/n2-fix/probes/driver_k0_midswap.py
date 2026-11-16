#!/usr/bin/env python3
"""N2 mid-window swap driver for the FIXED k0_train: pre-imports
cnet2_train from the run dir and wraps T.load so that the moment
T.load returns (generation pinned + hash-verified, k0's own u_raw
read still ahead), the generation's u_raw.npy is rewritten on disk
(x0.5). Fixed k0 must refuse at its single-read verification of
u_raw; the recorded SystemExit lands in the hook JSON.

Env: N2_HOOK_OUT (json evidence path).
Usage: driver_k0_midswap.py <run_dir> <k0_train.py> <k0 args...>
"""
import hashlib
import io
import json
import os
import sys

import numpy as np

RUN_DIR = os.path.abspath(sys.argv[1])
KT = os.path.abspath(sys.argv[2])
sys.argv = [KT] + sys.argv[3:]
HOOK_OUT = os.environ["N2_HOOK_OUT"]
sys.path.insert(0, RUN_DIR)
os.chdir(RUN_DIR)

import cnet2_train as T0  # noqa: E402

cap = {}
_orig_load = T0.load


def wrapped(dirn):
    D, is_val, meta = _orig_load(dirn)
    gen = meta["dataset_source"]
    up = os.path.join(gen, "u_raw.npy")
    with open(up, "rb") as fh:
        raw_before = fh.read()
    arr = np.load(io.BytesIO(raw_before))
    np.save(up, (arr * np.float32(0.5)).astype(arr.dtype))
    with open(up, "rb") as fh:
        raw_after = fh.read()
    cap["midswap"] = {
        "path": up,
        "verified_sha256": hashlib.sha256(raw_before).hexdigest(),
        "swapped_sha256": hashlib.sha256(raw_after).hexdigest(),
    }
    return D, is_val, meta


T0.load = wrapped
rc = 0
try:
    code = open(KT).read()
    exec(compile(code, KT, "exec"), {"__name__": "__main__", "__file__": KT})
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    cap["systemexit"] = repr(e.code)
except BaseException as e:  # noqa: BLE001 - evidence, then record
    cap["exception"] = f"{type(e).__name__}: {e}"
    rc = 1
cap["exit"] = rc
with open(HOOK_OUT, "w") as fh:
    json.dump(cap, fh, indent=1)
sys.exit(rc if isinstance(rc, int) else 1)
