#!/usr/bin/env python3
"""N2 swap-hook driver for the k0_train u_raw/label_status residual:
execs a k0_train.py as __main__ with np.load wrapped. The first by-path
load of u_raw.npy (k0_train main's own load, after T.load has pinned
and hash-verified the generation) is preceded -- when N2_SWAP=1 -- by
rewriting the on-disk file with different finite values (x0.5), i.e.
the file is swapped inside the verify-then-reopen window. The hook
records the sha256 of the generation's verified bytes (A, on disk at
hook entry) and of the bytes the load actually consumes.

Env: N2_HOOK_OUT (json evidence path), N2_SWAP ("1" to corrupt).
Usage: driver_uraw.py <k0_train.py> <k0 args...>
"""
import hashlib
import io
import json
import os
import sys

import numpy as np

KT = os.path.abspath(sys.argv[1])
sys.argv = [KT] + sys.argv[2:]
HOOK_OUT = os.environ["N2_HOOK_OUT"]
DO_SWAP = os.environ.get("N2_SWAP") == "1"

_real_load = np.load
cap = {}


def hooked(file, *args, **kw):
    if isinstance(file, (str, os.PathLike)):
        base = os.path.basename(str(file))
        if base in ("u_raw.npy", "label_status.npy") and base not in cap:
            with open(file, "rb") as fh:
                raw_before = fh.read()
            ent = {"path": str(file),
                   "disk_sha256_at_load": hashlib.sha256(raw_before).hexdigest()}
            if DO_SWAP and base == "u_raw.npy":
                arr = _real_load(io.BytesIO(raw_before))
                np.save(str(file), (arr * np.float32(0.5)).astype(arr.dtype))
                with open(file, "rb") as fh:
                    raw_after = fh.read()
                ent["swapped"] = True
                ent["consumed_sha256"] = hashlib.sha256(raw_after).hexdigest()
            else:
                ent["consumed_sha256"] = ent["disk_sha256_at_load"]
            cap[base] = ent
    return _real_load(file, *args, **kw)


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
