#!/usr/bin/env python3
"""Screen-path variant of the mid-window swap: T.load returns, the
generation's u_raw.npy is rewritten (x0.5), then run_screen reads
u_raw. Fixed k0 must refuse with the 'k0 screen:' verification
message.

Usage: probe_k0_midswap_screen.py <run_dir> <tag>
"""
import json
import os
import shutil
import subprocess
import sys

RUN_DIR = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
FIX = "/home/hatch/workspace/k0-codex-verify/fixture/ds_k0"
INIT = "/home/hatch/workspace/k0-codex-verify/runs/init.npz"
ARM = "/home/hatch/workspace/w-fixes/k0/evidence/w02_base/pristine.pt"
PY = "/home/hatch/.venvs/k0/bin/python"
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "driver_k0_midswap.py")

outdir = os.path.abspath(os.path.join("evidence", f"k0mids_{TAG}"))
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir)
ds = os.path.join(outdir, "ds")
shutil.copytree(FIX, ds, symlinks=True)
hook = os.path.join(outdir, "hook.json")
env = dict(os.environ, BC_DEVICE="cpu", N2_HOOK_OUT=hook)
r = subprocess.run(
    [PY, DRIVER, RUN_DIR, os.path.join(RUN_DIR, "k0_train.py"),
     "--screen", "--data", ds, "--init", INIT, "--checkpoint", ARM,
     "--seed", "7", "--max-games", "8"],
    cwd=RUN_DIR, env=env, capture_output=True, text=True, timeout=270)
res = {"tag": TAG, "exit": r.returncode,
       "hook": json.load(open(hook)) if os.path.exists(hook) else {},
       "stderr_tail": [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-4:]}
print(json.dumps(res, indent=1))
with open(os.path.join(os.path.dirname(outdir), f"k0mids_{TAG}.json"), "w") as fh:
    json.dump(res, fh, indent=1)
