#!/usr/bin/env python3
"""W01 repro: a pre-B2 checkpoint (prefix subset, no games_used_sha256 in
its effective record) resumed under the current code silently continues
on the NEW seeded subset -- same --seed/--max-games, no signal anywhere.

Steps:
  1. fresh train 2 updates with the PRE-B2 file (e49dcf33, prefix
     selection) on ds128, cap 64 -> old.pt; its eff must carry NO
     games_used_sha256.
  2. compute the prefix subset sha (what the old run actually trained
     on) and the seeded subset sha (B2 algorithm) directly from the
     dataset gids.
  3. resume old.pt with the file under test, same seed/cap, +2 updates.
     BUG (base): exit 0 and the resume eff's games_used_sha256 == the
     seeded sha (!= prefix sha) -- the training set changed silently.
     FIXED: exit != 0 naming the missing subset record (default), or
     with --accept-unrecorded-subset exit 0 and the new checkpoint's
     effective carries the subset sha + the acceptance record.

Usage: repro_w01.py <old_kt> <test_kt> <tag> [--accept]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

OLD_KT = os.path.abspath(sys.argv[1])
TEST_KT = os.path.abspath(sys.argv[2])
TAG = sys.argv[3]
ACCEPT = "--accept" in sys.argv[4:]
PY = "/home/hatch/.venvs/k0/bin/python"
FIX = "/home/hatch/workspace/b2-sampling/fixture/ds128"
INIT = "/home/hatch/workspace/k0-codex-verify/runs/init.npz"
RUN = "/home/hatch/workspace/w-fixes/k0/run"
E = f"/home/hatch/workspace/w-fixes/k0/evidence/w01_{TAG}"
os.makedirs(E, exist_ok=True)
os.makedirs(RUN, exist_ok=True)
for f in os.listdir("/home/hatch/workspace/v07v08-fix/run"):
    if f.endswith(".py") and f != "k0_train.py":
        shutil.copy(os.path.join("/home/hatch/workspace/v07v08-fix/run", f),
                    os.path.join(RUN, f))

COMMON = ["--data", FIX, "--seed", "7", "--aux-route", "detach", "--bs", "128",
          "--max-games", "64", "--grad-log-every", "100000"]


def stage(kt):
    shutil.copy(kt, os.path.join(RUN, "k0_train.py"))


def run(args, name):
    p = subprocess.run([PY, "k0_train.py"] + args, cwd=RUN,
                       capture_output=True, text=True)
    open(f"{E}/{name}.stdout", "w").write(p.stdout)
    open(f"{E}/{name}.stderr", "w").write(p.stderr)
    return p


def eff_of(stdout):
    for l in stdout.splitlines():
        if '"k0_effective"' in l:
            return json.loads(l)["k0_effective"]
    return None


def subset_shas():
    code = (
        "import sys, json, hashlib\n"
        "sys.path.insert(0, '/home/hatch/workspace/w-fixes/k0/run')\n"
        "import numpy as np, cnet2_train as T\n"
        f"D, is_val, meta = T.load({FIX!r})\n"
        "gids = np.unique(np.asarray(D['gid']))\n"
        "pre = np.sort(gids[:64])\n"
        "rs = np.random.RandomState((7 * 1000003 + 977) % (2 ** 31))\n"
        "seeded = np.sort(rs.choice(gids, size=64, replace=False))\n"
        "h = lambda a: hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()\n"
        "print(json.dumps({'prefix_sha': h(pre), 'seeded_sha': h(seeded),\n"
        "                  'same': bool((pre == seeded).all()), 'n_games': int(len(gids))}))\n"
    )
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("subset sha computation failed: " + p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


results = {}

# 1. genuine pre-B2 checkpoint
stage(OLD_KT)
p = run(COMMON + ["--init", INIT, "--out", f"{E}/old.pt", "--max-updates", "2"], "old_fresh")
results["old_fresh_exit"] = p.returncode
eff_old = eff_of(p.stdout)
results["old_eff_has_subset_sha"] = bool(eff_old and eff_old.get("games_used_sha256"))
results["old_eff_games_used"] = eff_old.get("games_used") if eff_old else None

# 2. the two candidate subsets
shas = subset_shas()
results.update(shas)

# 3. resume the old checkpoint with the file under test
stage(TEST_KT)
args = ["--data", FIX, "--resume", f"{E}/old.pt", "--out", f"{E}/old_resumed.pt",
        "--seed", "7", "--aux-route", "detach", "--bs", "128", "--max-games", "64",
        "--max-updates", "4", "--grad-log-every", "100000"]
if ACCEPT:
    args.append("--accept-unrecorded-subset")
p = run(args, "resume")
results["resume_exit"] = p.returncode
eff_r = eff_of(p.stdout)
results["resume_subset_sha"] = eff_r.get("games_used_sha256") if eff_r else None
results["resume_sampling"] = eff_r.get("games_sampling") if eff_r else None
results["resume_subset_migration"] = eff_r.get("subset_migration") if eff_r else None
results["resume_stderr_tail"] = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""

if os.path.exists(f"{E}/old_resumed.pt"):
    code = (
        "import torch, json\n"
        f"ck = torch.load({f'{E}/old_resumed.pt'!r}, map_location='cpu', weights_only=False)\n"
        "eff = ck.get('effective') or {}\n"
        "print(json.dumps({'ck_subset_sha': eff.get('games_used_sha256'),\n"
        "                  'ck_subset_migration': eff.get('subset_migration')}))\n"
    )
    p2 = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    if p2.returncode == 0:
        results.update(json.loads(p2.stdout.strip().splitlines()[-1]))

print(json.dumps(results, indent=1))
json.dump(results, open(f"{E}/repro_results.json", "w"), indent=1)
