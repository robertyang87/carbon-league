#!/usr/bin/env python3
"""T.load equivalence dump: per-field content sha256 + meta + is_val
for one cnet2_train.py, against the pristine fixture dataset.

Usage: probe_load_equiv.py <path-to-cnet2_train.py>
Prints one JSON object on the last line.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

TARGET = os.path.abspath(sys.argv[1])
FIX = "/home/hatch/workspace/k0-codex-verify/fixture/ds_k0"
sys.path.insert(0, os.path.dirname(TARGET))
os.environ.setdefault("BC_DEVICE", "cpu")

spec = importlib.util.spec_from_file_location("cnet2_train", TARGET)
CT = importlib.util.module_from_spec(spec)
sys.modules["cnet2_train"] = CT
spec.loader.exec_module(CT)

D, is_val, meta = CT.load(FIX)
out = {"fields": {}, "meta_sha256": None, "is_val_sha256": None}
for k in sorted(D):
    a = np.ascontiguousarray(np.asarray(D[k]))
    out["fields"][k] = {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                        "dtype": str(a.dtype), "shape": list(a.shape)}
m = {k: v for k, v in meta.items()}
out["meta_sha256"] = hashlib.sha256(
    json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()
out["meta_keys"] = sorted(m)
iv = np.ascontiguousarray(np.asarray(is_val))
out["is_val_sha256"] = hashlib.sha256(iv.tobytes()).hexdigest()
out["is_val_sum"] = int(np.asarray(is_val).sum())
print(json.dumps(out, indent=1, sort_keys=True))
