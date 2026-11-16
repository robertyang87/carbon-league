#!/usr/bin/env python3
"""N2-b manifest swap driver: pre-imports cnet2_train from the run dir
and wraps T.load so that the moment T.load returns (generation pinned +
hash-verified against manifest A), the generation's manifest.json on
disk is -- when N2B_SWAP=1 -- replaced with a prepared manifest B
(env N2B_MAN_B). B differs from A in exactly one entry: fields.u_raw
names a planted u_raw_evil.npy and declares that file's true sha256;
total / generation / cfg_fp / meta_sha256 are A's values unchanged.

Baseline k0 re-reads manifest.json by path after T.load, so B steers
its _load_gen_field: the evil u_raw is consumed under the verified
generation's identity. Fixed k0 consumes the manifest T.load hands it
(meta['generation_manifest'], i.e. A) and never re-opens the file --
the swap is then inert. The hook records the on-disk manifest sha256
at T.load return and after the swap, plus the canonical hash of the
manifest T.load actually handed over in meta.

Env: N2_HOOK_OUT (json evidence path), N2B_SWAP ("1" to swap),
     N2B_MAN_B (path to manifest B bytes; required when swapping).
Usage: driver_k0_manswap.py <run_dir> <k0_train.py> <k0 args...>
"""
import hashlib
import json
import os
import sys

RUN_DIR = os.path.abspath(sys.argv[1])
KT = os.path.abspath(sys.argv[2])
sys.argv = [KT] + sys.argv[3:]
HOOK_OUT = os.environ["N2_HOOK_OUT"]
DO_SWAP = os.environ.get("N2B_SWAP") == "1"
sys.path.insert(0, RUN_DIR)
os.chdir(RUN_DIR)

import cnet2_train as T0  # noqa: E402

cap = {}
_orig_load = T0.load


def _canon_sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def wrapped(dirn):
    D, is_val, meta = _orig_load(dirn)
    gen = meta["dataset_source"]
    mp = os.path.join(gen, "manifest.json")
    with open(mp, "rb") as fh:
        raw_before = fh.read()
    ent = {"manifest_path": mp,
           "disk_sha256_at_load_return": hashlib.sha256(raw_before).hexdigest(),
           "meta_has_generation_manifest": "generation_manifest" in meta,
           "meta_generation_manifest_sha256": (
               _canon_sha(meta["generation_manifest"])
               if meta.get("generation_manifest") is not None else None)}
    if DO_SWAP:
        with open(os.environ["N2B_MAN_B"], "rb") as fh:
            b = fh.read()
        with open(mp, "wb") as fh:
            fh.write(b)
        ent["swapped"] = True
        ent["disk_sha256_after_swap"] = hashlib.sha256(b).hexdigest()
    cap["manswap"] = ent
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
