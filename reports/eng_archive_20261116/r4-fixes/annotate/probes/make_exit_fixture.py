"""Build a small contract-conforming sealed exit generation for the
X04(d) probe (the d1 fixture's dummy q predates the V01 semantic gate
and is refused by the current loader on pristine bytes, so the probe
uses this hand-sealed generation instead: all rows unsearched --
searched=False, q all-NaN, pi_slot=-1, pi all zero -- which satisfies
_check_exit_q trivially).

Usage: python3 make_exit_fixture.py RUN_DIR OUT_ROOT
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np  # noqa: E402
from cnet2_build import FIELDS  # noqa: E402
from exit_gen import EXTRA  # noqa: E402

OUT = os.path.abspath(sys.argv[2])
N = 6
KEYS = list(FIELDS) + list(EXTRA)


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


arrs = {}
for k in KEYS:
    dt, tail = (FIELDS.get(k) or EXTRA[k])
    arrs[k] = np.zeros((N,) + tuple(tail), dtype=dt)
arrs["q"][:] = np.nan                      # unsearched sentinel
arrs["pi_slot"][:] = -1
arrs["gid"][:] = 0
arrs["step"][:] = np.arange(N)
arrs["value"][:] = np.array([0.5 * i for i in range(N)],
                            dtype=arrs["value"].dtype)

gen_id = "gen-probe-x04d-0001"
gen_dir = os.path.join(OUT, "generations", gen_id)
os.makedirs(gen_dir, exist_ok=True)
finfo = {}
for k in KEYS:
    p = os.path.join(gen_dir, k + ".npy")
    np.save(p, arrs[k])
    raw = open(p, "rb").read()
    finfo[k] = {"file": k + ".npy", "bytes": len(raw), "rows": N,
                "sha256": hashlib.sha256(raw).hexdigest()}
meta = {"games": [{"game": "probe-game", "val": False}], "total": N,
        "format": "cnet2-exit-generation/1"}
with open(os.path.join(gen_dir, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1, sort_keys=True)
meta_sha = hashlib.sha256(
    open(os.path.join(gen_dir, "meta.json"), "rb").read()).hexdigest()
man = {"format": "cnet2-exit-generation/1", "total": N,
       "producer": {"probe": "make_exit_fixture"},
       "meta_sha256": meta_sha, "fields": finfo,
       "generation": gen_id, "created": 0}
man["self_sha256"] = hashlib.sha256(canon(man)).hexdigest()
with open(os.path.join(gen_dir, "manifest.json"), "w") as fh:
    json.dump(man, fh, indent=1, sort_keys=True)
man_sha = hashlib.sha256(
    open(os.path.join(gen_dir, "manifest.json"), "rb").read()).hexdigest()
ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
       "manifest_sha256": man_sha, "total": N, "created": 0}
with open(os.path.join(OUT, "current.json"), "w") as fh:
    json.dump(ptr, fh)
print("fixture sealed:", OUT, gen_id, "fields:", len(finfo))
