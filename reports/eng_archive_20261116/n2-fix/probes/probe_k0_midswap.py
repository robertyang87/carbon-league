#!/usr/bin/env python3
"""Run driver_k0_midswap against a fresh fixture clone.

Usage: probe_k0_midswap.py <run_dir> <tag>
Writes evidence/k0mid_<tag>.json (+ case dir evidence/k0mid_<tag>/).
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
PY = "/home/hatch/.venvs/k0/bin/python"
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "driver_k0_midswap.py")

outdir = os.path.abspath(os.path.join("evidence", f"k0mid_{TAG}"))
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir)
ds = os.path.join(outdir, "ds")
shutil.copytree(FIX, ds, symlinks=True)
hook = os.path.join(outdir, "hook.json")
env = dict(os.environ, BC_DEVICE="cpu", N2_HOOK_OUT=hook)
r = subprocess.run(
    [PY, DRIVER, RUN_DIR, os.path.join(RUN_DIR, "k0_train.py"),
     "--data", ds, "--init", INIT, "--out", os.path.join(outdir, "out.pt"),
     "--max-updates", "2", "--seed", "7", "--aux-route", "detach",
     "--bs", "128", "--max-games", "2", "--grad-log-every", "100000"],
    cwd=RUN_DIR, env=env, capture_output=True, text=True, timeout=270)
res = {"tag": TAG, "exit": r.returncode,
       "hook": json.load(open(hook)) if os.path.exists(hook) else {},
       "stderr_tail": [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-4:]}
print(json.dumps(res, indent=1))
with open(os.path.join(os.path.dirname(outdir), f"k0mid_{TAG}.json"), "w") as fh:
    json.dump(res, fh, indent=1)
