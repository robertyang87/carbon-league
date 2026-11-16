#!/usr/bin/env python3
"""W01 regression battery:
  A. Post-B2 produce+resume pairs under the base file and under the
     fixed file (ds128, cap 64, seed 7): fresh effective configs must
     be identical dicts and all update losses (fresh + resumed) must
     be bit-identical -- the fix must not perturb a recorded run.
  B. Tampered record: a post-B2 checkpoint whose effective.max_games
     is edited 64 -> 32 (subset sha left at the cap-64 value), resumed
     with --max-games 32 so the scalar protocol check passes: the W01
     subset assertion must die naming games_used_sha256 and both shas.
  C. Plain CLI change: untampered checkpoint resumed with
     --max-games 32 dies via the pre-existing scalar protocol check.

Usage: probe_w01_regress.py <base_kt> <test_kt> <tag>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

BASE_KT = os.path.abspath(sys.argv[1])
TEST_KT = os.path.abspath(sys.argv[2])
TAG = sys.argv[3]
PY = "/home/hatch/.venvs/k0/bin/python"
FIX = "/home/hatch/workspace/b2-sampling/fixture/ds128"
INIT = "/home/hatch/workspace/k0-codex-verify/runs/init.npz"
RUN = "/home/hatch/workspace/w-fixes/k0/run"
E = f"/home/hatch/workspace/w-fixes/k0/evidence/w01regress_{TAG}"
os.makedirs(E, exist_ok=True)
os.makedirs(RUN, exist_ok=True)
for f in os.listdir("/home/hatch/workspace/v07v08-fix/run"):
    if f.endswith(".py") and f != "k0_train.py":
        shutil.copy(os.path.join("/home/hatch/workspace/v07v08-fix/run", f),
                    os.path.join(RUN, f))

COMMON = ["--data", FIX, "--seed", "7", "--aux-route", "detach", "--bs", "128",
          "--grad-log-every", "100000"]
results = {}


def stage(kt):
    shutil.copy(kt, os.path.join(RUN, "k0_train.py"))


def run(args, name):
    p = subprocess.run([PY, "k0_train.py"] + args, cwd=RUN, capture_output=True, text=True)
    open(f"{E}/{name}.stdout", "w").write(p.stdout)
    open(f"{E}/{name}.stderr", "w").write(p.stderr)
    eff, losses = None, []
    for l in p.stdout.splitlines():
        if '"k0_effective"' in l:
            eff = json.loads(l)["k0_effective"]
        elif '"update"' in l:
            losses.append(json.loads(l)["loss"])
    return p.returncode, eff, losses


def pair(kt, name):
    stage(kt)
    rc1, eff1, l1 = run(COMMON + ["--init", INIT, "--out", f"{E}/{name}.pt",
                                  "--max-games", "64", "--max-updates", "2"], f"{name}_fresh")
    rc2, eff2, l2 = run(["--data", FIX, "--resume", f"{E}/{name}.pt", "--out", f"{E}/{name}_r.pt",
                         "--seed", "7", "--aux-route", "detach", "--bs", "128",
                         "--max-games", "64", "--max-updates", "4",
                         "--grad-log-every", "100000"], f"{name}_resume")
    return {"fresh_exit": rc1, "resume_exit": rc2, "fresh_eff": eff1,
            "resume_eff": eff2, "losses": l1 + l2}


results["base"] = pair(BASE_KT, "base")
results["test"] = pair(TEST_KT, "test")
results["A_fresh_eff_identical"] = results["base"]["fresh_eff"] == results["test"]["fresh_eff"]
results["A_resume_eff_identical"] = results["base"]["resume_eff"] == results["test"]["resume_eff"]
results["A_losses_identical"] = results["base"]["losses"] == results["test"]["losses"]

# B. tampered max_games in the record (subset sha left stale)
stage(TEST_KT)
code = (
    "import torch, hashlib, json\n"
    f"ck = torch.load({f'{E}/test.pt'!r}, map_location='cpu', weights_only=False)\n"
    "ck['effective']['max_games'] = 32\n"
    f"torch.save(ck, {f'{E}/tampered.pt'!r})\n"
    f"raw = open({f'{E}/tampered.pt'!r}, 'rb').read()\n"
    "m = {'checkpoint': 'tampered.pt', 'sha256': hashlib.sha256(raw).hexdigest(),\n"
    "     'updates': int(ck['updates']), 'complete': True, 'time': 0}\n"
    f"json.dump(m, open({f'{E}/tampered.pt.complete.json'!r}, 'w'))\n"
    "print(ck['effective']['games_used_sha256'])\n"
)
p = subprocess.run([PY, "-c", code], capture_output=True, text=True)
recorded_sha = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else None
rc, _, _ = run(["--data", FIX, "--resume", f"{E}/tampered.pt", "--out", f"{E}/tampered_r.pt",
                "--seed", "7", "--aux-route", "detach", "--bs", "128",
                "--max-games", "32", "--max-updates", "4", "--grad-log-every", "100000"],
               "tampered_resume")
err = open(f"{E}/tampered_resume.stderr").read()
results["B_tamper_exit"] = rc
results["B_tamper_names_field"] = "games_used_sha256" in err
results["B_tamper_names_recorded_sha"] = bool(recorded_sha) and recorded_sha in err

# C. plain CLI max-games change
rc, _, _ = run(["--data", FIX, "--resume", f"{E}/test.pt", "--out", f"{E}/cli_r.pt",
                "--seed", "7", "--aux-route", "detach", "--bs", "128",
                "--max-games", "32", "--max-updates", "4", "--grad-log-every", "100000"],
               "cli_resume")
err = open(f"{E}/cli_resume.stderr").read()
results["C_cli_exit"] = rc
results["C_cli_names_max_games"] = "max_games (--max-games)" in err

print(json.dumps({k: v for k, v in results.items()
                  if not k.endswith("_eff") and k != "losses"}, indent=1))
json.dump(results, open(f"{E}/regress_results.json", "w"), indent=1)
