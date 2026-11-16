"""X04 (report item) probe: teacher weights are hashed by path at plan
time and loaded by path later -- swap the file in between and the run
trains/evaluates with teacher B while every record says A.

Flow (mirrors main()): build_plan(args) -> swap the weights file for a
doctored B -> load the teacher exactly the way the version under test
does (by path on base; from the plan's frozen bytes on the fix) ->
evaluate one root -> compare Q against a pristine-A reference.

Usage: python3 probe_x04w.py RUN_DIR
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as np  # noqa: E402
import k0_annotate as A  # noqa: E402

WORK = "/tmp/x04w_probe"
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
wpath = os.path.join(WORK, "heads.npz")
shutil.copy("/home/hatch/workspace/s12work/heads_v2.npz", wpath)
md5_a = hashlib.md5(open(wpath, "rb").read()).hexdigest()

games_f = os.path.join(WORK, "games.json")
json.dump(["selfplay-994000"], open(games_f, "w"))
args = argparse.Namespace(
    games=games_f, replays="/home/hatch/workspace/gen1r_smoke/replays",
    weights=wpath, window_lo=206, window_hi=208, pi0=1.0, pi1=1.0,
    seed=12345, K=4, M=2, H=1, teacher_style=2, belief="sample",
    train_temp=1.0, behavior_temp=1.0, seed_teacher=7000,
    isolation_file=[])
plan = A.build_plan(args)
print("plan weights_md5 == md5(A):", plan["weights_md5"] == md5_a)
pg = plan["per_game"][0]
spec = {"game": pg["game"], "path": pg["path"], "step": 206, "me": 0,
        "ft": pg["ft"]}

# reference: pristine A
A._init_worker(open(wpath, "rb").read() if "weights_bytes" in plan
               else wpath)
q_ref = float(A._eval_root(spec, plan["config"])["Q_mean"][0])
print("Q_ref (teacher A):", q_ref)

# swap the file for a doctored B (largest float array scaled x2)
w = dict(np.load(wpath))
key = max((k for k in w if np.issubdtype(w[k].dtype, np.floating)),
          key=lambda k: w[k].size)
w[key] = w[key] * 2.0
np.savez(wpath, **w)
md5_b = hashlib.md5(open(wpath, "rb").read()).hexdigest()
print(f"weights file swapped A->B (doctored key {key!r}); "
      f"md5 now {md5_b[:12]}.. != recorded {plan['weights_md5'][:12]}..")

# load the way main() does on this version, then evaluate the same root
A._W.clear()
if "weights_bytes" in plan:
    A._init_worker(plan["weights_bytes"])
    src = "plan-frozen bytes"
else:
    A._init_worker(args.weights)
    src = "by-path load"
q_got = float(A._eval_root(spec, plan["config"])["Q_mean"][0])
print(f"Q_after_swap ({src}): {q_got}; equals teacher A: {q_got == q_ref}")
