"""W04 skip-ledger probes (binding condition on the W04 flag (a) acceptance).

Usage: probe_ledger.py <path-to-cnet2_train.py>
Prints one JSON object.

Part A: a constructed examiner carrying one skip of each reason class
plus a dependency-layer skip and two controls (an existing in-dir data
file that must FREEZE, a write-mode open that must NOT be ledgered).
The ledger is read from the selection record itself
(sel["public"]["examiner_data_skips"]) after _selection_freeze_examiner,
and cross-checked against the _selection_snapshot_deps skips_out param.

Part B: the real production examiner (~/workspace/a4-examiner-proof/)
re-derived through the standard selection open: identity unchanged,
frozen set unchanged ({master2_src.py 404114a0 code}), and the ledger
present as an EXPLICIT empty list.
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

TARGET = os.path.abspath(sys.argv[1])
CODE = os.path.expanduser("~/workspace/a4-selection/fixture/code")
sys.path.insert(0, CODE)
os.chdir(CODE)
os.environ.setdefault("BC_DEVICE", "cpu")

spec = importlib.util.spec_from_file_location("cnet2_train", TARGET)
CT = importlib.util.module_from_spec(spec)
sys.modules["cnet2_train"] = CT
spec.loader.exec_module(CT)

out = {}

# ---------------------------------------------------------------- Part A
tmp = tempfile.mkdtemp(prefix="ledger_")
origin = os.path.join(tmp, "origin")
outside = os.path.join(tmp, "outside")
snap = os.path.join(tmp, "snap")
os.makedirs(origin)
os.makedirs(outside)
os.makedirs(snap)
with open(os.path.join(outside, "secret.json"), "w") as f:
    f.write('{"secret": 1}\n')
with open(os.path.join(origin, "weights.json"), "w") as f:
    f.write('{"w": 1}\n')
with open(os.path.join(origin, "dep.py"), "w") as f:
    f.write("import os\n"
            "D = open(os.path.join(os.path.dirname(__file__), "
            "\"dep_absent.json\"))\n")
exam_src = (
    "import os\n"
    "import dep\n"
    "W = open(os.path.join(os.path.dirname(__file__), \"weights.json\"))\n"
    "A = open(os.path.join(os.path.dirname(__file__), \"absent.json\"))\n"
    "def _load(p):\n"
    "    return open(p)\n"
    f"B = open(\"{os.path.join(outside, 'secret.json')}\")\n"
    "OUT = open(os.path.join(os.path.dirname(__file__), \"out.txt\"), \"w\")\n"
)
exam = os.path.join(origin, "exam.py")
with open(exam, "w") as f:
    f.write(exam_src)
raw = open(exam, "rb").read()

expected = [
    {"file": "dep.py",
     "expr": "os.path.join(os.path.dirname(__file__), \"dep_absent.json\")",
     "reason": "missing"},
    {"file": "exam.py",
     "expr": f"\"{os.path.join(outside, 'secret.json')}\"",
     "reason": "out-of-dir-existing"},
    {"file": "exam.py",
     "expr": "os.path.join(os.path.dirname(__file__), \"absent.json\")",
     "reason": "missing"},
    {"file": "exam.py", "expr": "p", "reason": "irreducible"},
]

# A1: through the record write point (_selection_freeze_examiner)
sel = {"examiner": exam,
       "examiner_md5": hashlib.md5(raw).hexdigest(),
       "_examiner_bytes": raw, "dir": snap, "public": {}}
CT._selection_freeze_examiner(sel)
ledger_a = sel["public"].get("examiner_data_skips")
deps_a = sel["public"].get("examiner_deps")
out["A"] = {
    "ledger": ledger_a,
    "ledger_matches_expected": ledger_a == expected,
    "sel_key_matches_public": sel.get("examiner_data_skips") == ledger_a,
    "weights_frozen_as_data": any(
        d.get("file") == "weights.json" and d.get("kind") == "data"
        for d in deps_a or []),
    "dep_frozen_as_code": any(
        d.get("file") == "dep.py" and d.get("kind") == "code"
        for d in deps_a or []),
    "write_mode_not_ledgered": all(
        "out.txt" not in (e.get("expr") or "") for e in ledger_a or []),
    "write_mode_not_frozen": all(
        d.get("file") != "out.txt" for d in deps_a or []),
}

# A2: the skips_out parameter directly, no record involved
snap2 = os.path.join(tmp, "snap2")
os.makedirs(snap2)
direct = []
CT._selection_snapshot_deps(exam, raw, snap2, skips_out=direct)
out["A"]["skips_out_matches"] = direct == expected

# A3: no skips_out -> return contract unchanged (a plain deps list)
plain = CT._selection_snapshot_deps(exam, raw, snap2)
out["A"]["default_return_is_deps_list"] = (
    isinstance(plain, list)
    and all(isinstance(d, dict) and "md5" in d for d in plain))

# ---------------------------------------------------------------- Part B
EXAM = os.path.expanduser("~/workspace/a4-examiner-proof/opp_master2r.py")
tmpb = tempfile.mkdtemp(prefix="ledger_real_")
os.environ["BC_SELECTION_EXAMINER"] = EXAM
os.environ["BC_SELECTION_FIXTURE"] = "1"
os.environ["BC_SELECTION_PAIRS"] = "2"
os.environ["BC_SELECTION_WORKERS"] = "4"
selb = CT._selection_config()
CT._selection_open(selb, os.path.join(tmpb, "out.npz"))
deps_b = selb.get("examiner_snapshot_deps") or []
frozen_set = sorted(
    (d.get("file"), d.get("md5"), d.get("kind")) for d in deps_b)
out["B"] = {
    "identity_ok": selb["examiner_md5"] == "2aebfbc1008d3475ece8e178696e292d",
    "frozen_set": [list(t) for t in frozen_set],
    "frozen_set_unchanged": frozen_set == [
        ("master2_src.py", "404114a0b5e422516322f62e9ddb6870", "code")],
    "public_key_present": "examiner_data_skips" in selb["public"],
    "ledger_is_explicit_empty": selb["public"].get("examiner_data_skips") == []
                                and selb.get("examiner_data_skips") == [],
}

out["ALL_PASS"] = (
    out["A"]["ledger_matches_expected"]
    and out["A"]["sel_key_matches_public"]
    and out["A"]["weights_frozen_as_data"]
    and out["A"]["dep_frozen_as_code"]
    and out["A"]["write_mode_not_ledgered"]
    and out["A"]["write_mode_not_frozen"]
    and out["A"]["skips_out_matches"]
    and out["A"]["default_return_is_deps_list"]
    and out["B"]["identity_ok"]
    and out["B"]["frozen_set_unchanged"]
    and out["B"]["public_key_present"]
    and out["B"]["ledger_is_explicit_empty"])
print(json.dumps(out, indent=1, ensure_ascii=False))
