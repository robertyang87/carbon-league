#!/usr/bin/env python3
"""N2-b runner (training path): short k0_train run (2 updates) through
driver_k0_manswap.py on a scratch clone of the ds_k0 fixture. The
clone's generation dir carries a planted u_raw_evil.npy (u_raw x0.5 --
the exact transform the N2 file-swap probe used, so a consumed-evil
run must reproduce that probe's u losses) and, with --swap, manifest
B is installed over manifest.json the moment T.load returns.

Usage: probe_k0_man.py <run_dir> <tag> [--swap]
Writes evidence/<tag>/ (case dir) + evidence/<tag>.json.
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np

RUN_DIR = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
SWAP = "--swap" in sys.argv
FIX = "/home/hatch/workspace/k0-codex-verify/fixture/ds_k0"
INIT = "/home/hatch/workspace/k0-codex-verify/runs/init.npz"
PY = "/home/hatch/.venvs/k0/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "driver_k0_manswap.py")
EVID = os.path.abspath(os.path.join(HERE, "..", "evidence"))

outdir = os.path.join(EVID, TAG)
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir)
ds = os.path.join(outdir, "ds")
shutil.copytree(FIX, ds, symlinks=True)

ptr = json.load(open(os.path.join(ds, "current.json")))
gen = os.path.join(ds, ptr["dir"])
man_a = json.load(open(os.path.join(gen, "manifest.json")))

# Plant the evil u_raw (x0.5, same transform as the N2 file swap).
raw_u = open(os.path.join(gen, "u_raw.npy"), "rb").read()
arr = np.load(io.BytesIO(raw_u))
evil_path = os.path.join(gen, "u_raw_evil.npy")
np.save(evil_path, (arr * np.float32(0.5)).astype(arr.dtype))
evil_sha = hashlib.sha256(open(evil_path, "rb").read()).hexdigest()

man_b = json.loads(json.dumps(man_a))
man_b["fields"]["u_raw"] = {"file": "u_raw_evil.npy",
                            "rows": man_a["fields"]["u_raw"]["rows"],
                            "sha256": evil_sha}
man_b_path = os.path.join(outdir, "manifest_B.json")
with open(man_b_path, "w") as fh:
    json.dump(man_b, fh, indent=1, sort_keys=True)

hook = os.path.join(outdir, "hook.json")
env = dict(os.environ, BC_DEVICE="cpu", N2_HOOK_OUT=hook,
           N2B_SWAP="1" if SWAP else "0", N2B_MAN_B=man_b_path)
r = subprocess.run(
    [PY, DRIVER, RUN_DIR, os.path.join(RUN_DIR, "k0_train.py"),
     "--data", ds, "--init", INIT, "--out", os.path.join(outdir, "out.pt"),
     "--max-updates", "2", "--seed", "7", "--aux-route", "detach",
     "--bs", "128", "--max-games", "2", "--grad-log-every", "100000"],
    cwd=RUN_DIR, env=env, capture_output=True, text=True, timeout=270)
open(os.path.join(outdir, "stdout"), "w").write(r.stdout)
open(os.path.join(outdir, "stderr"), "w").write(r.stderr)

losses, eff = [], None
for line in r.stdout.splitlines():
    if '"k0_effective"' in line:
        eff = json.loads(line)["k0_effective"]
    elif '"update"' in line:
        losses.append(json.loads(line)["loss"])
res = {"tag": TAG, "swap": SWAP, "exit": r.returncode,
       "losses": losses,
       "effective": eff,
       "man_a_u_raw": man_a["fields"]["u_raw"],
       "man_b_u_raw": man_b["fields"]["u_raw"],
       "evil_sha256": evil_sha,
       "hook": json.load(open(hook)) if os.path.exists(hook) else {},
       "stderr_tail": [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-4:]}
print(json.dumps(res, indent=1))
with open(os.path.join(EVID, f"{TAG}.json"), "w") as fh:
    json.dump(res, fh, indent=1)
