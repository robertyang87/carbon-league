"""X03 plan-level probe: does build_plan's run identity notice a replay
whose BYTES changed under the same game id / path?

Two doctorings of selfplay-994000.json.gz in a scratch replays dir:
  (i)  regzip: same parsed JSON, re-serialized (bytes differ);
  (ii) extkey: one extra key added to one step dict (parsed content
       differs, roots unchanged).
Usage: python3 probe_x03_plan.py RUN_DIR WORK_DIR
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
WORK = os.path.abspath(sys.argv[2])
import k0_annotate as A  # noqa: E402

SRC = "/home/hatch/workspace/gen1r_smoke/replays"
GAMES = ["selfplay-994000", "selfplay-994001"]


def make_args(replays):
    games_f = os.path.join(WORK, "games.json")
    json.dump(GAMES, open(games_f, "w"))
    return argparse.Namespace(
        games=games_f, replays=replays,
        weights="/home/hatch/workspace/s12work/heads_v2.npz",
        window_lo=206, window_hi=208, pi0=1.0, pi1=1.0, seed=12345,
        K=4, M=2, H=1, teacher_style=2, belief="sample",
        train_temp=1.0, behavior_temp=1.0, seed_teacher=7000,
        isolation_file=[])


def fresh_replays(name):
    dst = os.path.join(WORK, name)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SRC, dst)
    return dst


def doctor(replays, mode):
    p = os.path.join(replays, "selfplay-994000.json.gz")
    d = json.load(gzip.open(p))
    if mode == "extkey":
        d["probe_note"] = "doctored"
    with gzip.open(p, "wt") as fh:
        json.dump(d, fh)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


base_dir = fresh_replays("rp_base")
plan0 = A.build_plan(make_args(base_dir))
print("roots base:", plan0["n_roots"],
      "per-game roots:", [len(pg["roots"]) for pg in plan0["per_game"]])
for mode in ("regzip", "extkey"):
    rd = fresh_replays("rp_" + mode)
    before = sha(os.path.join(rd, "selfplay-994000.json.gz"))
    doctor(rd, mode)
    after = sha(os.path.join(rd, "selfplay-994000.json.gz"))
    plan1 = A.build_plan(make_args(rd))
    same_id = plan1["identity"] == plan0["identity"]
    print(f"{mode}: replay bytes changed={before != after} "
          f"roots={plan1['n_roots']} identity_unchanged={same_id}")
    print(f"{mode}: identity keys: {sorted(plan1['identity'])}")
    if "replay_sha256" in plan1["identity"]:
        m = plan1["identity"]["replay_sha256"]
        print(f"{mode}: identity replay map 994000 matches doctored bytes: "
              f"{m.get('selfplay-994000') == after}")
