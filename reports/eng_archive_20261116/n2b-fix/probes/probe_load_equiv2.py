#!/usr/bin/env python3
"""T.load equivalence dump (N2-b form): per-field content sha256 +
is_val for one cnet2_train.py against the pristine fixture dataset,
with meta reported in two parts -- every pre-existing key (canonical
hash + key list) and the N2-b addition 'generation_manifest' kept
separate (canonical hash + verbatim equality against the fixture's
on-disk manifest A, the one load() verified against).

Usage: probe_load_equiv2.py <path-to-cnet2_train.py>
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
out = {"fields": {}}
for k in sorted(D):
    a = np.ascontiguousarray(np.asarray(D[k]))
    out["fields"][k] = {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                        "dtype": str(a.dtype), "shape": list(a.shape)}


def canon(o):
    return hashlib.sha256(
        json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


rest = {k: v for k, v in meta.items() if k != "generation_manifest"}
out["meta_preexisting_sha256"] = canon(rest)
out["meta_preexisting_keys"] = sorted(rest)
gm = meta.get("generation_manifest")
out["generation_manifest_present"] = "generation_manifest" in meta
out["generation_manifest_sha256"] = canon(gm) if gm is not None else None
ptr = json.load(open(os.path.join(FIX, "current.json")))
man_disk = json.load(open(os.path.join(FIX, ptr["dir"], "manifest.json")))
out["generation_manifest_eq_disk"] = (gm == man_disk)
iv = np.ascontiguousarray(np.asarray(is_val))
out["is_val_sha256"] = hashlib.sha256(iv.tobytes()).hexdigest()
out["is_val_sum"] = int(np.asarray(is_val).sum())
print(json.dumps(out, indent=1, sort_keys=True))
