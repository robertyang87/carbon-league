#!/usr/bin/env python3
"""N2-b runner (screen path): k0_train --screen through
driver_k0_manswap.py on a scratch clone of the ds_k0 fixture, with the
same planted u_raw_evil.npy / manifest B construction as the training
path probe. Full stdout is saved verbatim (the regression compare is
byte-level zero-diff between runs).

Usage: probe_k0_man_screen.py <run_dir> <tag> [--swap]
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
# --variant x05 (default): evil file is u_raw x0.5 (finite; on the
# screen path u_raw values feed only the input finite gate, so this
# variant is consumption-invisible there by construction).
# --variant nan: evil file carries NaN at flat index 0 (a usable row),
# which the baseline screen's u_raw finite gate refuses -- B-driven
# refusal proves the re-read manifest steered the load.
VARIANT = "x05"
if "--variant" in sys.argv:
    VARIANT = sys.argv[sys.argv.index("--variant") + 1]
FIX = "/home/hatch/workspace/k0-codex-verify/fixture/ds_k0"
INIT = "/home/hatch/workspace/k0-codex-verify/runs/init.npz"
ARM = "/home/hatch/workspace/w-fixes/k0/evidence/w02_base/pristine.pt"
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

raw_u = open(os.path.join(gen, "u_raw.npy"), "rb").read()
arr = np.load(io.BytesIO(raw_u))
if VARIANT == "nan":
    ls = np.load(os.path.join(gen, "label_status.npy"))
    assert int(ls.reshape(-1)[0]) == 0, "flat row 0 must be a usable row"
    evil = arr.copy()
    evil.reshape(-1)[0] = np.nan
    evil_name = "u_raw_nan.npy"
else:
    evil = (arr * np.float32(0.5)).astype(arr.dtype)
    evil_name = "u_raw_evil.npy"
evil_path = os.path.join(gen, evil_name)
np.save(evil_path, evil)
evil_sha = hashlib.sha256(open(evil_path, "rb").read()).hexdigest()

man_b = json.loads(json.dumps(man_a))
man_b["fields"]["u_raw"] = {"file": evil_name,
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
     "--screen", "--data", ds, "--init", INIT, "--checkpoint", ARM,
     "--seed", "7", "--max-games", "8"],
    cwd=RUN_DIR, env=env, capture_output=True, text=True, timeout=270)
open(os.path.join(outdir, "stdout"), "w").write(r.stdout)
open(os.path.join(outdir, "stderr"), "w").write(r.stderr)

res = {"tag": TAG, "swap": SWAP, "variant": VARIANT, "exit": r.returncode,
       "stdout_sha256": hashlib.sha256(r.stdout.encode()).hexdigest(),
       "stdout": r.stdout,
       "man_b_u_raw": man_b["fields"]["u_raw"],
       "evil_sha256": evil_sha,
       "hook": json.load(open(hook)) if os.path.exists(hook) else {},
       "stderr_tail": [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-4:]}
print(json.dumps({k: v for k, v in res.items() if k != "stdout"}, indent=1))
with open(os.path.join(EVID, f"{TAG}.json"), "w") as fh:
    json.dump(res, fh, indent=1)
