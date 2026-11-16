#!/usr/bin/env python3
"""W02 swap-hook driver: execs a k0_train.py as __main__ with np.load
wrapped. The wrapper recognises the two aux label loads (path loads
named aux_die.npy / aux_die_m.npy, or BytesIO loads parsing to an
(n, 10, 3) array), snapshots the exact bytes + array each load
consumed, and -- once BOTH aux arrays have been consumed -- rewrites
the on-disk aux files with different-but-valid bytes (aux_die flipped
0<->1, aux_die_m rolled by one row), when W02_SWAP=1.

Whatever a run records or trains on after that point reveals which
read it came from: a second by-path read sees the flipped bytes B;
a single-read binding keeps the consumed bytes A.

Env: W02_HOOK_OUT (json evidence path), W02_AUX_DIE, W02_AUX_DIE_M
(on-disk paths to corrupt), W02_SWAP ("1" to corrupt).
Usage: driver_swap.py <k0_train.py> <k0 args...>
"""
import hashlib
import io
import json
import os
import sys

import numpy as np

KT = os.path.abspath(sys.argv[1])
sys.argv = [KT] + sys.argv[2:]
HOOK_OUT = os.environ["W02_HOOK_OUT"]
DIE_P = os.environ["W02_AUX_DIE"]
DIEM_P = os.environ["W02_AUX_DIE_M"]
DO_SWAP = os.environ.get("W02_SWAP") == "1"

_real_load = np.load
cap = {}


def _arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _classify(file):
    """(kind, raw_bytes) if this np.load call is an aux label load."""
    raw = None
    if isinstance(file, (str, os.PathLike)):
        if os.path.basename(str(file)) not in ("aux_die.npy", "aux_die_m.npy"):
            return None
        with open(file, "rb") as fh:
            raw = fh.read()
    elif isinstance(file, io.BytesIO):
        raw = file.getvalue()
    if raw is None:
        return None
    try:
        a = _real_load(io.BytesIO(raw))
    except Exception:
        return None
    if not isinstance(a, np.ndarray) or a.ndim != 3 or a.shape[1:] != (10, 3):
        return None
    return ("die_m" if a.dtype == bool else "die"), raw


def hooked(file, *args, **kw):
    m = _classify(file)
    out = _real_load(file, *args, **kw)
    if m is not None:
        kind, raw = m
        if kind not in cap:
            cap[kind] = {"raw_sha256": hashlib.sha256(raw).hexdigest(),
                         "arr_sha256": _arr_sha(np.asarray(out))}
            if DO_SWAP and len(cap) == 2 and "swapped" not in cap:
                d = _real_load(DIE_P)
                np.save(DIE_P, (1 - d).astype(d.dtype))
                mm = _real_load(DIEM_P)
                np.save(DIEM_P, np.roll(mm, 1, axis=0))
                cap["swapped"] = True
    return out


np.load = hooked
rc = 0
try:
    code = open(KT).read()
    exec(compile(code, KT, "exec"), {"__name__": "__main__", "__file__": KT})
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    cap["systemexit"] = repr(e.code)
except BaseException as e:  # noqa: BLE001 - evidence, then propagate
    cap["exception"] = f"{type(e).__name__}: {e}"
    json.dump(cap, open(HOOK_OUT, "w"), indent=1)
    raise
json.dump(cap, open(HOOK_OUT, "w"), indent=1)
sys.exit(rc)
