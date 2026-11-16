#!/usr/bin/env python3
"""k0 u_raw PRE-swap probe: the generation's u_raw.npy is replaced on
disk (x0.5) BEFORE the run starts, manifest untouched. Baseline k0
consumes the swapped bytes silently; fixed k0 must refuse at its
single-read verification, naming the field.

Usage: probe_k0_preswap.py <run_dir> <tag>
Writes evidence/k0pre_<tag>.json next to CWD.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

RUN_DIR = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
FIX = "/home/hatch/workspace/k0-codex-verify/fixture/ds_k0"
INIT = "/home/hatch/workspace/k0-codex-verify/runs/init.npz"
PY = "/home/hatch/.venvs/k0/bin/python"

outdir = os.path.abspath(os.path.join("evidence", f"k0pre_{TAG}"))
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir)
ds = os.path.join(outdir, "ds")
shutil.copytree(FIX, ds, symlinks=True)
import glob
_hits = glob.glob(os.path.join(ds, "generations", "*", "u_raw.npy"))
assert len(_hits) == 1, _hits
up = _hits[0]
arr = np.load(up)
np.save(up, (arr * 0.5).astype(np.float32))

env = dict(os.environ, BC_DEVICE="cpu")
r = subprocess.run(
    [PY, os.path.join(RUN_DIR, "k0_train.py"),
     "--data", ds, "--init", INIT, "--out", os.path.join(outdir, "out.pt"),
     "--max-updates", "2", "--seed", "7", "--aux-route", "detach",
     "--bs", "128", "--max-games", "2", "--grad-log-every", "100000"],
    cwd=RUN_DIR, env=env, capture_output=True, text=True, timeout=270)
res = {"tag": TAG, "exit": r.returncode,
       "stderr_tail": [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-4:],
       "stdout_tail": [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-4:]}
print(json.dumps(res, indent=1))
with open(os.path.join(os.path.dirname(outdir), f"k0pre_{TAG}.json"), "w") as fh:
    json.dump(res, fh, indent=1)
