#!/usr/bin/env python3
"""N2 runner: short k0_train run (2 updates) through driver_uraw.py on a
scratch clone of the ds_k0 fixture.

Usage: probe_k0_uraw.py <run_dir> <tag> [--swap]
Verdict material on stdout: exit code, update losses, the hook record
(u_raw verified bytes A vs consumed bytes), the manifest's u_raw sha.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

RUN = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
SWAP = "--swap" in sys.argv
PY = "/home/hatch/.venvs/k0/bin/python"
FIX = "/home/hatch/workspace/k0-codex-verify/fixture/ds_k0"
INIT = "/home/hatch/workspace/k0-codex-verify/runs/init.npz"
DRIVER = "/home/hatch/workspace/n2-fix/probes/driver_uraw.py"
E = f"/home/hatch/workspace/n2-fix/evidence/k0_{TAG}"
os.makedirs(E, exist_ok=True)

ds = os.path.join(E, "ds")
if os.path.exists(ds):
    shutil.rmtree(ds)
shutil.copytree(FIX, ds)

ptr = json.load(open(os.path.join(ds, "current.json")))
man = json.load(open(os.path.join(ds, ptr["dir"], "manifest.json")))
man_uraw_sha = man["fields"]["u_raw"]["sha256"]

env = dict(os.environ, N2_HOOK_OUT=f"{E}/hook.json",
           N2_SWAP="1" if SWAP else "0", BC_DEVICE="cpu")
p = subprocess.run([PY, DRIVER, os.path.join(RUN, "k0_train.py"),
                    "--data", ds, "--init", INIT, "--out", f"{E}/out.pt",
                    "--max-updates", "2", "--seed", "7", "--aux-route", "detach",
                    "--bs", "128", "--max-games", "2", "--grad-log-every", "100000"],
                   cwd=RUN, capture_output=True, text=True, env=env)
open(f"{E}/stdout", "w").write(p.stdout)
open(f"{E}/stderr", "w").write(p.stderr)

losses, eff = [], None
for line in p.stdout.splitlines():
    if '"k0_effective"' in line:
        eff = json.loads(line)["k0_effective"]
    elif '"update"' in line:
        losses.append(json.loads(line)["loss"])
hook = json.load(open(f"{E}/hook.json")) if os.path.exists(f"{E}/hook.json") else {}
print(json.dumps({"tag": TAG, "swap": SWAP, "exit": p.returncode,
                  "losses": losses,
                  "manifest_u_raw_sha256": man_uraw_sha,
                  "hook": hook,
                  "stderr_tail": p.stderr.strip().splitlines()[-3:]}, indent=1))
