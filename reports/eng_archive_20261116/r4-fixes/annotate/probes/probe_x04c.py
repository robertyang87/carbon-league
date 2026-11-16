"""X04(c) probe: publish() assembles from shard files it re-reads by
path. If a shard file is replaced (after _verify_shard recorded its
sha256) by a DIFFERENT but internally self-consistent npz, base
publish assembles the unverified bytes silently; fixed publish must
die naming the game.

Setup: a real tiny run's out dir (produced beforehand, arg 2) is
copied; one shard is doctored: q value changed + embedded _meta array
hashes recomputed so the npz is self-consistent; its file sha256 then
differs from the recorded shard_sha256 in the metas handed to publish.

Usage: python3 probe_x04c.py RUN_DIR OUT_DIR
"""
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np  # noqa: E402
import k0_annotate as A  # noqa: E402

SRC_OUT = os.path.abspath(sys.argv[2])
WORK = "/tmp/x04c_probe"
shutil.rmtree(WORK, ignore_errors=True)
shutil.copytree(SRC_OUT, os.path.join(WORK, "out"))
OUT = os.path.join(WORK, "out")

progress = json.load(open(os.path.join(OUT, "progress.json")))
identity = progress["identity"]
prog_games = progress["games"]
WIN = identity.get("window", [206, 208])

# rebuild the plan the same way main() does (same args as the tiny run)
import argparse
args = argparse.Namespace(
    games="/tmp/x04c_games.json",
    replays="/home/hatch/workspace/gen1r_smoke/replays",
    weights="/home/hatch/workspace/s12work/heads_v2.npz",
    window_lo=WIN[0], window_hi=WIN[1], pi0=1.0, pi1=1.0, seed=12345,
    K=4, M=2, H=1, teacher_style=2, belief="sample",
    train_temp=1.0, behavior_temp=1.0, seed_teacher=7000,
    isolation_file=[])
games = sorted(prog_games)
json.dump(games, open("/tmp/x04c_games.json", "w"))
plan = A.build_plan(args)
shard_dir = os.path.join(OUT, "shards")
import inspect
VS_KW = ({"accept_unbound": True}
         if "accept_unbound" in inspect.signature(A._verify_shard).parameters
         else {})
metas = {}
for pg in plan["per_game"]:
    rec = prog_games[pg["game"]]
    metas[pg["game"]] = A._verify_shard(shard_dir, pg, identity,
                                        rec.get("shard_sha256"), **VS_KW)
print("shards verified:", {g: m["shard_sha256"][:12]
                           for g, m in metas.items()})

# doctor one shard: change a Q_std value (a diagnostic field no
# record gate inspects, so the doctored shard passes every content
# check -- only its file sha256 betrays it), re-seal embedded hashes
g0 = games[0]
sp = os.path.join(shard_dir, g0 + ".npz")
with np.load(sp) as D:
    data = {k: D[k] for k in D.files}
meta = json.loads(str(data["_meta"].item()))
arrs = {k: data[k] for k in A.FIELDS}
arrs["Q_std"] = arrs["Q_std"].copy()
arrs["Q_std"][0, 0] = 123.456
meta["arrays"] = {k: A._array_sha(arrs[k]) for k in A.FIELDS}
tmp = sp + ".doc"
with open(tmp, "wb") as fh:
    np.savez(fh, _meta=np.asarray(json.dumps(meta, sort_keys=True)), **arrs)
os.replace(tmp, sp)
new_sha = hashlib.sha256(open(sp, "rb").read()).hexdigest()
print(f"doctored shard {g0}: recorded sha "
      f"{metas[g0]['shard_sha256'][:12]}.. != file sha {new_sha[:12]}..")

try:
    summary = A.publish(OUT, plan, [metas[g] for g in plan["games"]])
    gen_dir = os.path.join(OUT, "generations", summary["generation"])
    qs = np.load(os.path.join(gen_dir, "Q_std.npy"))
    print(f"PUBLISHED generation {summary['generation']} with assembled "
          f"Q_std[0,0]={float(qs[0, 0])!r} (doctored value 123.456 "
          f"present: {float(qs[0, 0]) == 123.456})")
except SystemExit as e:
    print(f"PUBLISH DIED: {e}")
