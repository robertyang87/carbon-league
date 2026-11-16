#!/usr/bin/env python3
"""W02 probe runner: fresh 2-update training on a scratch clone of the
ds_k0 fixture, driven through driver_swap.py.

Two runs per version under test:
  pristine (W02_SWAP=0): baseline behaviour, no interference.
  swap     (W02_SWAP=1): aux files on disk are flipped AFTER both aux
    arrays have been consumed by np.load.
Verdict reads the emitted k0_effective line:
  recorded files_sha256 vs the sha of the bytes the loads consumed (A,
  from the hook) and of the bytes left on disk after the run (B when
  swapped):  BUG = recorded == B while training consumed A (and the
  swap run's losses equal the pristine run's, i.e. training used A).
  FIXED = recorded == A under swap, losses still equal pristine.

Usage: probe_w02.py <k0_train.py> <tag>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

KT = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
PY = "/home/hatch/.venvs/k0/bin/python"
FIX = "/home/hatch/workspace/k0-codex-verify/fixture/ds_k0"
INIT = "/home/hatch/workspace/k0-codex-verify/runs/init.npz"
RUN = "/home/hatch/workspace/w-fixes/k0/run"
DRIVER = "/home/hatch/workspace/w-fixes/k0/probes/driver_swap.py"
E = f"/home/hatch/workspace/w-fixes/k0/evidence/w02_{TAG}"
os.makedirs(E, exist_ok=True)
os.makedirs(RUN, exist_ok=True)
for f in os.listdir("/home/hatch/workspace/v07v08-fix/run"):
    if f.endswith(".py") and f != "k0_train.py":
        shutil.copy(os.path.join("/home/hatch/workspace/v07v08-fix/run", f),
                    os.path.join(RUN, f))
shutil.copy(KT, os.path.join(RUN, "k0_train.py"))

BASE = ["--seed", "7", "--aux-route", "detach", "--bs", "128",
        "--max-games", "2", "--grad-log-every", "100000"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def one_run(name, swap):
    ds = os.path.join(E, f"ds_{name}")
    if os.path.exists(ds):
        shutil.rmtree(ds)
    shutil.copytree(FIX, ds)
    env = dict(os.environ,
               W02_HOOK_OUT=f"{E}/{name}.hook.json",
               W02_AUX_DIE=os.path.join(ds, "aux_die.npy"),
               W02_AUX_DIE_M=os.path.join(ds, "aux_die_m.npy"),
               W02_SWAP="1" if swap else "0")
    p = subprocess.run([PY, DRIVER, os.path.join(RUN, "k0_train.py"),
                        "--data", ds, "--init", INIT, "--out", f"{E}/{name}.pt",
                        "--max-updates", "2"] + BASE,
                       cwd=RUN, capture_output=True, text=True, env=env)
    open(f"{E}/{name}.stdout", "w").write(p.stdout)
    open(f"{E}/{name}.stderr", "w").write(p.stderr)
    eff, losses = None, []
    for l in p.stdout.splitlines():
        if '"k0_effective"' in l:
            eff = json.loads(l)["k0_effective"]
        elif '"update"' in l:
            losses.append(json.loads(l)["loss"])
    hook = json.load(open(f"{E}/{name}.hook.json")) if os.path.exists(f"{E}/{name}.hook.json") else {}
    return {"exit": p.returncode,
            "recorded_files_sha256": (eff or {}).get("aux", {}).get("files_sha256"),
            "recorded_sidecar_sha256": (eff or {}).get("aux", {}).get("sidecar_sha256"),
            "losses": losses,
            "hook": hook,
            "disk_after": {"aux_die.npy": sha256_file(os.path.join(ds, "aux_die.npy")),
                           "aux_die_m.npy": sha256_file(os.path.join(ds, "aux_die_m.npy"))}}


results = {"pristine": one_run("pristine", False), "swap": one_run("swap", True)}
print(json.dumps(results, indent=1))
json.dump(results, open(f"{E}/probe_results.json", "w"), indent=1)
