"""Real-examiner proof (strategy-group condition): selection must OPEN
against the genuine opp_master2r.py (md5 2aebfbc1...), freeze
master2_src.py (md5 404114a0...) into selection_examiner_deps with the
examiner identity unchanged, and complete a short fixture selection eval
whose games load the SNAPSHOT examiner (which in turn loads the frozen
same-dir master2_src.py copy).
Usage: proof_real_examiner.py <path-to-cnet2_train.py>
"""
import hashlib
import importlib.util
import json
import os
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

EXAM = os.path.expanduser("~/workspace/a4-examiner-proof/opp_master2r.py")
EXPECT_EXAM_MD5 = "2aebfbc1008d3475ece8e178696e292d"
EXPECT_DEP = {"file": "master2_src.py", "md5": "404114a0b5e422516322f62e9ddb6870"}

tmp = tempfile.mkdtemp(prefix="realproof_")
os.environ["BC_SELECTION_EXAMINER"] = EXAM
os.environ["BC_SELECTION_FIXTURE"] = "1"
os.environ["BC_SELECTION_PAIRS"] = "2"
os.environ["BC_SELECTION_WORKERS"] = "4"

sel = CT._selection_config()
identity_ok = sel["examiner_md5"] == EXPECT_EXAM_MD5
CT._selection_open(sel, os.path.join(tmp, "out.npz"))
deps = sel.get("examiner_snapshot_deps")
# W04: dep entries now also carry kind/import bookkeeping, so the
# record check is by file + md5 + kind, not whole-dict equality.
dep_ok = deps is not None and any(
    d.get("file") == EXPECT_DEP["file"] and d.get("md5") == EXPECT_DEP["md5"]
    and d.get("kind") == "code" for d in deps)
snap_dep = os.path.join(sel["dir"], "master2_src.py")
snap_dep_ok = False
if os.path.isfile(snap_dep):
    with open(snap_dep, "rb") as f:
        snap_dep_ok = hashlib.md5(f.read()).hexdigest() == EXPECT_DEP["md5"]
exec_is_snapshot = sel.get("examiner_exec", "").startswith(sel["dir"])

cand = os.path.join(CODE, "example.py")
row = CT._selection_eval(sel, cand, "naked-proxy")

print(json.dumps({"real_examiner_proof": {
    "examiner": EXAM,
    "examiner_md5_recorded": sel["examiner_md5"],
    "identity_ok": identity_ok,
    "deps": deps,
    "dep_record_ok": dep_ok,
    "snapshot_dep_bytes_ok": snap_dep_ok,
    "examiner_exec_is_snapshot": exec_is_snapshot,
    "examiner_exec": sel.get("examiner_exec"),
    "eval_row": {k: row.get(k) for k in ("ok", "games", "valid_games",
                                          "errors", "mean_u", "wall_s")},
}}), flush=True)
