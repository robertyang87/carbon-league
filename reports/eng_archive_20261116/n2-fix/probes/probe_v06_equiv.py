#!/usr/bin/env python3
"""V06 invariance equivalence probe (reconstructed to the W04 log's
semantics): selection opens against the a4 fixture examiner
(example.py, md5 84face63...), evals a candidate, then the ORIGIN
examiner file is swapped on disk and the eval repeats -- the recorded
identity, the snapshot and both mean_u values must be invariant.

Usage: probe_v06_equiv.py <path-to-cnet2_train.py>
Prints one JSON object.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile

TARGET = os.path.abspath(sys.argv[1])
FIX = os.path.expanduser("~/workspace/a4-selection/fixture")
CODE = os.path.join(FIX, "code")
sys.path.insert(0, CODE)
os.chdir(CODE)
os.environ.setdefault("BC_DEVICE", "cpu")

spec = importlib.util.spec_from_file_location("cnet2_train", TARGET)
CT = importlib.util.module_from_spec(spec)
sys.modules["cnet2_train"] = CT
spec.loader.exec_module(CT)

tmp = tempfile.mkdtemp(prefix="n2v06_")
exam = os.path.join(tmp, "examiner.py")
shutil.copy(os.path.join(CODE, "example.py"), exam)
md5_v1 = hashlib.md5(open(exam, "rb").read()).hexdigest()

os.environ["BC_SELECTION_EXAMINER"] = exam
os.environ["BC_SELECTION_FIXTURE"] = "1"
os.environ["BC_SELECTION_PAIRS"] = "2"
os.environ["BC_SELECTION_WORKERS"] = "4"

sel = CT._selection_config()
CT._selection_open(sel, os.path.join(tmp, "out.npz"))
cand = os.path.join(CODE, "example.py")
row1 = CT._selection_eval(sel, cand, "naked-proxy")

with open(exam, "a") as fh:
    fh.write("\n# swapped after open\n")
md5_v2 = hashlib.md5(open(exam, "rb").read()).hexdigest()
row2 = CT._selection_eval(sel, cand, "naked-proxy")

snap_md5 = None
exec_path = sel.get("examiner_exec")
if exec_path and os.path.isfile(exec_path):
    snap_md5 = hashlib.md5(open(exec_path, "rb").read()).hexdigest()
print(json.dumps({"V06": {
    "md5_v1": md5_v1, "md5_v2_swapped": md5_v2,
    "recorded_examiner_md5": sel["examiner_md5"],
    "mean_u_eval1": row1.get("mean_u"), "ok1": row1.get("ok"),
    "mean_u_eval2_after_swap": row2.get("mean_u"), "ok2": row2.get("ok"),
    "examiner_exec": sel.get("examiner_exec"),
    "snapshot_md5": snap_md5,
    "snapshot_deps": sel.get("examiner_snapshot_deps"),
    "sel_snapshot_keys": {k: v for k, v in sel.items() if "snapshot" in k and isinstance(v, str)},
}}, indent=1))
