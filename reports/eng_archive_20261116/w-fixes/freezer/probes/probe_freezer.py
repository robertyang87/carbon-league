"""V06 freezer-fix probe matrix: production-form dynamic loads must freeze;
unresolvable ones must die by name; static behavior unchanged.
Usage: probe_freezer.py <path-to-cnet2_train.py>
Prints one JSON object: {case: {"outcome": ..., ...}}.
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

HDR = "import importlib.util\nimport os\n"
results = {}


def run_case(name, files, examiner, expect, must_mention=None):
    """files: {relname: content} written under a fresh origin dir; examiner:
    the entry filename. expect: 'freeze' or 'die'."""
    tmp = tempfile.mkdtemp(prefix="frz_")
    origin = os.path.join(tmp, "origin")
    snap = os.path.join(tmp, "snap")
    os.makedirs(origin)
    os.makedirs(snap)
    for rel, content in files.items():
        p = os.path.join(origin, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    exam = os.path.join(origin, examiner)
    with open(exam, "rb") as f:
        raw = f.read()
    try:
        deps = CT._selection_snapshot_deps(exam, raw, snap)
    except SystemExit as e:
        msg = str(e)
        ok = expect == "die" and (must_mention is None or must_mention in msg)
        results[name] = {"outcome": "die", "pass": ok, "message": msg[:160]}
        return
    if expect != "freeze":
        results[name] = {"outcome": "freeze", "pass": False, "deps": deps}
        return
    # verify every recorded dep is byte-exact in the snap dir
    bad = []
    for d in deps:
        sp = os.path.join(snap, d["file"])
        if not os.path.isfile(sp):
            bad.append(d["file"] + ":missing")
            continue
        with open(sp, "rb") as f:
            if hashlib.md5(f.read()).hexdigest() != d["md5"]:
                bad.append(d["file"] + ":md5")
    results[name] = {"outcome": "freeze", "pass": not bad,
                     "deps": sorted(d["file"] for d in deps), "bad": bad}


def loader_examiner(body, extra=""):
    return HDR + body + "\n" + extra


# ---- should freeze -------------------------------------------------------
run_case("join_via_module_name", {
    "examiner.py": loader_examiner(
        '_HERE = os.path.dirname(os.path.abspath(__file__))\n'
        '_s = importlib.util.spec_from_file_location("dep", os.path.join(_HERE, "dep.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("join_inline_dirname", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("dep", os.path.join(os.path.dirname(__file__), "dep.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("add_concat", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("dep", os.path.dirname(__file__) + "/dep.py")\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("pathlib_div", {
    "examiner.py": (
        "import importlib.util\nfrom pathlib import Path\n"
        '_s = importlib.util.spec_from_file_location("dep", Path(__file__).parent / "dep.py")\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("pathlib_with_name", {
    "examiner.py": (
        "import importlib.util\nfrom pathlib import Path\n"
        '_s = importlib.util.spec_from_file_location("dep", Path(__file__).with_name("dep.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("kwarg_location", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("dep", location=os.path.join(os.path.dirname(__file__), "dep.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("source_file_loader", {
    "examiner.py": (
        "import os\nfrom importlib.machinery import SourceFileLoader\n"
        '_m = SourceFileLoader("dep", os.path.join(os.path.dirname(__file__), "dep.py")).load_module()\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("local_function_binding", {
    "examiner.py": loader_examiner(
        'def _load():\n'
        '    here = os.path.dirname(os.path.abspath(__file__))\n'
        '    s = importlib.util.spec_from_file_location("dep", os.path.join(here, "dep.py"))\n'
        '    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'
        '    return m\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("recursive_chain", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("a", os.path.join(os.path.dirname(__file__), "a.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "a.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("b", os.path.join(os.path.dirname(__file__), "b.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "b.py": "import c\nY = c.Z\n",
    "c.py": "Z = 3\n"}, "examiner.py", "freeze")

run_case("mixed_static_and_dyn", {
    "examiner.py": loader_examiner(
        'import helper\n'
        '_s = importlib.util.spec_from_file_location("dep", os.path.join(os.path.dirname(__file__), "dep.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'
        'CFG = "table.json"\n'),
    "helper.py": "H = 1\n",
    "dep.py": "X = 1\n",
    "table.json": '{"k": 1}\n'}, "examiner.py", "freeze")

run_case("mention_only_no_die", {
    "examiner.py": (
        '# this examiner used to call spec_from_file_location near __file__\n'
        'NOTE = "exec_module"\n'
        'def make_agent():\n    return None\n'),
    }, "examiner.py", "freeze")

# ---- should die ----------------------------------------------------------
run_case("variable_path", {
    "examiner.py": loader_examiner(
        'def _load(p):\n'
        '    s = importlib.util.spec_from_file_location("dep", p)\n'
        '    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "die", "not statically reducible")

run_case("bare_relative_literal", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("dep", "dep.py")\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "die", "bare relative path")

run_case("escapes_dir", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("dep", os.path.join(os.path.dirname(__file__), "..", "dep.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    }, "examiner.py", "die", "outside the loading file's directory")

run_case("missing_target", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("dep", os.path.join(os.path.dirname(__file__), "nope.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    }, "examiner.py", "die", "does not exist")

run_case("cycle_ab", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("a", os.path.join(os.path.dirname(__file__), "a.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "a.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("b", os.path.join(os.path.dirname(__file__), "b.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "b.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("a", os.path.join(os.path.dirname(__file__), "a.py"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    }, "examiner.py", "die", "cycle")

run_case("self_load", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("me", __file__)\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    }, "examiner.py", "die", "cycle")

run_case("exec_without_spec", {
    "examiner.py": (
        'def run(loader, mod):\n    loader.exec_module(mod)\n'),
    }, "examiner.py", "die", "without creating its spec")

run_case("fstring_path", {
    "examiner.py": loader_examiner(
        '_s = importlib.util.spec_from_file_location("dep", f"{os.path.dirname(__file__)}/dep.py")\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "die", "not statically reducible")

# depth: examiner -> d1 -> ... -> d9 (9 dynamic edges > cap 8)
depth_files = {"examiner.py": loader_examiner(
    '_s = importlib.util.spec_from_file_location("d1", os.path.join(os.path.dirname(__file__), "d1.py"))\n'
    '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n')}
for i in range(1, 10):
    nxt = f"d{i + 1}.py" if i < 9 else "leaf.py"
    depth_files[f"d{i}.py"] = loader_examiner(
        f'_s = importlib.util.spec_from_file_location("d{i + 1}", os.path.join(os.path.dirname(__file__), "{nxt}"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n')
depth_files["leaf.py"] = "X = 1\n"
run_case("depth_cap", depth_files, "examiner.py", "die", "depth")

# depth boundary: exactly 8 edges must freeze
ok_files = {"examiner.py": loader_examiner(
    '_s = importlib.util.spec_from_file_location("e1", os.path.join(os.path.dirname(__file__), "e1.py"))\n'
    '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n')}
for i in range(1, 8):
    nxt = f"e{i + 1}.py" if i < 7 else "leaf.py"
    ok_files[f"e{i}.py"] = loader_examiner(
        f'_s = importlib.util.spec_from_file_location("e{i + 1}", os.path.join(os.path.dirname(__file__), "{nxt}"))\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n')
ok_files["leaf.py"] = "X = 1\n"
run_case("depth_boundary_8", ok_files, "examiner.py", "freeze")

# ---- production shape: exists-guarded for-loop over candidate paths -----
_PROD = (
    'import importlib.util, os\n'
    '_H = os.path.dirname(os.path.abspath(__file__))\n'
    'for _p in (os.path.join(_H, "dep.py"), os.path.join(_H, "..", "..", "data", "public_src", "dep.py")):\n'
    '    if os.path.exists(_p):\n'
    '        break\n'
    '_s = importlib.util.spec_from_file_location("dep_public", _p)\n'
    '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n')
run_case("prod_loop_first_exists", {
    "examiner.py": _PROD,
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("prod_loop_second_samedir", {
    "examiner.py": (
        'import importlib.util, os\n'
        '_H = os.path.dirname(os.path.abspath(__file__))\n'
        'for _p in (os.path.join(_H, "nope.py"), os.path.join(_H, "dep.py")):\n'
        '    if os.path.exists(_p):\n'
        '        break\n'
        '_s = importlib.util.spec_from_file_location("dep_public", _p)\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "freeze")

run_case("prod_loop_none_exist", {
    "examiner.py": _PROD}, "examiner.py", "die", "none of the candidate paths exists")

run_case("prod_loop_unguarded", {
    "examiner.py": (
        'import importlib.util, os\n'
        '_H = os.path.dirname(os.path.abspath(__file__))\n'
        'for _p in (os.path.join(_H, "dep.py"),):\n'
        '    pass\n'
        '_s = importlib.util.spec_from_file_location("dep_public", _p)\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'),
    "dep.py": "X = 1\n"}, "examiner.py", "die", "not statically reducible")

# absolute literal into the same dir (needs the real tmp path baked in)
_tmp = tempfile.mkdtemp(prefix="frz_abs_")
_org = os.path.join(_tmp, "origin"); _snp = os.path.join(_tmp, "snap")
os.makedirs(_org); os.makedirs(_snp)
with open(os.path.join(_org, "dep.py"), "w") as f:
    f.write("X = 1\n")
_abs_exam = os.path.join(_org, "examiner.py")
with open(_abs_exam, "w") as f:
    f.write(loader_examiner(
        f'_s = importlib.util.spec_from_file_location("dep", {os.path.join(_org, "dep.py")!r})\n'
        '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n'))
with open(_abs_exam, "rb") as f:
    _raw = f.read()
# W09 (supersedes the V06-era expectation): an absolute-literal load in the
# EXAMINER itself can neither be honored (the runtime reads the original
# path, never the copy) nor rewritten (the examiner's bytes are its
# identity), so it must now DIE by name. Dep-level absolute literals are
# covered by probe_w_repro.py (rewritten + redirected instead).
try:
    _deps = CT._selection_snapshot_deps(_abs_exam, _raw, _snp)
    results["absolute_literal"] = {"outcome": "freeze",
                                    "pass": False,
                                    "deps": [d["file"] for d in _deps]}
except SystemExit as e:
    results["absolute_literal"] = {"outcome": "die",
                                    "pass": "absolute path" in str(e),
                                    "message": str(e)[:160]}

fails = [k for k, v in results.items() if not v["pass"]]
print(json.dumps({"cases": results, "failures": fails,
                  "n_pass": len(results) - len(fails), "n_total": len(results)},
                 indent=1), flush=True)
