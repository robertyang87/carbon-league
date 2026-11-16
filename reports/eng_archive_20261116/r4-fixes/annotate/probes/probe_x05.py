"""X05 probe: publish() reuse of an already-published generation does
not verify the published meta.json against the manifest seal.

Harness (publish-level, no teacher eval): from one tiny run's shards,
publish inputs A -> gen A, pointer=A. Publish modified inputs B
(different teacher seed -> different meta -> different gen id) ->
gen B, pointer=B. Corrupt A's meta.json on disk. Publish A again:
  base:  returns reused=true and flips current.json back to bad A;
  fixed: dies naming meta.json and current.json still points at B.

Usage: python3 probe_x05.py RUN_DIR TINY_OUT
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
WORK = "/tmp/x05_probe"
shutil.rmtree(WORK, ignore_errors=True)
shutil.copytree(SRC_OUT, os.path.join(WORK, "out"))
OUT = os.path.join(WORK, "out")
shard_dir = os.path.join(OUT, "shards")

progress = json.load(open(os.path.join(OUT, "progress.json")))
games = sorted(progress["games"])
WIN = progress["identity"].get("window", [206, 208])


def shard_metas():
    metas = []
    for g in games:
        p = os.path.join(shard_dir, g + ".npz")
        with np.load(p) as D:
            m = json.loads(str(D["_meta"].item()))
        m["shard_sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        metas.append(m)
    return metas


import argparse


def make_plan(seed_teacher):
    gf = os.path.join(WORK, "games.json")
    json.dump(games, open(gf, "w"))
    args = argparse.Namespace(
        games=gf, replays="/home/hatch/workspace/gen1r_smoke/replays",
        weights="/home/hatch/workspace/s12work/heads_v2.npz",
        window_lo=WIN[0], window_hi=WIN[1], pi0=1.0, pi1=1.0, seed=12345,
        K=4, M=2, H=1, teacher_style=2, belief="sample",
        train_temp=1.0, behavior_temp=1.0, seed_teacher=seed_teacher,
        isolation_file=[])
    return A.build_plan(args)


metas = shard_metas()
plan_a = make_plan(7000)
sum_a = A.publish(OUT, plan_a, metas)
gen_a = sum_a["generation"]
ptr = json.load(open(os.path.join(OUT, "current.json")))
print(f"A published: {gen_a} reused={sum_a['reused']} ptr={ptr['generation']}")

plan_b = make_plan(7001)
sum_b = A.publish(OUT, plan_b, metas)
gen_b = sum_b["generation"]
ptr = json.load(open(os.path.join(OUT, "current.json")))
print(f"B published: {gen_b} reused={sum_b['reused']} ptr={ptr['generation']}")

# corrupt A's meta.json (still valid JSON, wrong content -> wrong seal)
meta_p = os.path.join(OUT, "generations", gen_a, "meta.json")
meta = json.load(open(meta_p))
meta["total"] = 999999
with open(meta_p, "w") as fh:
    json.dump(meta, fh, indent=1, sort_keys=True)
print("A meta.json corrupted (seal now broken)")

try:
    sum_a2 = A.publish(OUT, plan_a, shard_metas())
    ptr = json.load(open(os.path.join(OUT, "current.json")))
    print(f"republish A: reused={sum_a2['reused']} "
          f"ptr_now={ptr['generation']} (bad A: {ptr['generation'] == gen_a})")
except SystemExit as e:
    ptr = json.load(open(os.path.join(OUT, "current.json")))
    print(f"republish A DIED: {e}")
    print(f"ptr still B: {ptr['generation'] == gen_b}")
