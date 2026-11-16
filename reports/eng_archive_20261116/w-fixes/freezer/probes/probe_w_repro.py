"""W-round freezer repro probes (W03/W04/W09/W10/W11).
Usage: probe_w_repro.py <path-to-cnet2_train.py> <bug|fixed>
bug   = expectations against the a3ad86d4 baseline (defects reproduce)
fixed = expectations against the repaired freezer
Prints one JSON object with per-case pass flags.
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types

TARGET = os.path.abspath(sys.argv[1])
MODE = sys.argv[2]
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


def rec(name, **kw):
    results[name] = kw


def mk_origin(files):
    tmp = tempfile.mkdtemp(prefix="wfix_")
    origin = os.path.join(tmp, "origin")
    snap = os.path.join(tmp, "snap")
    os.makedirs(origin)
    os.makedirs(snap)
    for rel, content in files.items():
        p = os.path.join(origin, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    return tmp, origin, snap


def load_module_from(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ================= W03: sys.modules substitution =================
def w03():
    files = {
        "examiner.py": "import helper\n\ndef make_agent():\n    return None\n",
        "helper.py": "MARKER = 'frozen'\n",
    }
    tmp, origin, snap = mk_origin(files)
    exam = os.path.join(origin, "examiner.py")
    with open(exam, "rb") as f:
        raw = f.read()
    # poison: a different helper.py loaded under the same plain name
    pdir = os.path.join(tmp, "poison")
    os.makedirs(pdir)
    ppath = os.path.join(pdir, "helper.py")
    with open(ppath, "w") as f:
        f.write("MARKER = 'poison'\n")
    poison = load_module_from(ppath, "helper")
    sys.modules["helper"] = poison
    try:
        deps = CT._selection_snapshot_deps(exam, raw, snap)
        audit_outcome, audit_msg = "froze", ""
        dep_files = sorted(d["file"] for d in deps)
    except SystemExit as e:
        audit_outcome, audit_msg = "die", str(e)
        dep_files = []
    if MODE == "bug":
        rec("w03_freeze_audit", outcome=audit_outcome,
            pass_=(audit_outcome == "froze" and dep_files == ["helper.py"]),
            note="baseline freezes with no sys.modules audit")
    else:
        rec("w03_freeze_audit", outcome=audit_outcome,
            pass_=(audit_outcome == "die" and "helper" in audit_msg
                   and ppath in audit_msg),
            message=audit_msg[:200])
    sys.modules.pop("helper", None)

    # child guard: freeze cleanly, then poison AFTER the freeze and play
    snap2 = os.path.join(tmp, "snap2")
    os.makedirs(snap2)
    sel = {"examiner": exam, "examiner_md5": md5_of(exam),
           "_examiner_bytes": raw, "dir": snap2, "public": {}}
    CT._selection_freeze_examiner(sel)
    stub = types.ModuleType("eval_v53")
    stub.play_game = lambda task: {"stub": True}
    sys.modules["eval_v53"] = stub
    task = ("selection", "cand.py", sel["examiner_exec"], 997000, 0)
    sys.modules["helper"] = poison
    try:
        CT._selection_play(task)
        play_poisoned = "played"
    except RuntimeError as e:
        play_poisoned = "raise:" + str(e)[:120]
    sys.modules.pop("helper", None)
    try:
        CT._selection_play(task)
        played_clean = True
    except BaseException as e:  # noqa: BLE001
        played_clean = f"raise:{type(e).__name__}:{str(e)[:100]}"
    m = sys.modules.get("helper")
    if MODE == "bug":
        rec("w03_child_guard", poisoned=play_poisoned, clean=played_clean,
            pass_=(play_poisoned == "played" and played_clean is True
                   and m is None),
            note="baseline _selection_play has no frozen-import guard")
    else:
        rec("w03_child_guard", poisoned=play_poisoned, clean=played_clean,
            preloaded_file=getattr(m, "__file__", None),
            preloaded_marker=getattr(m, "MARKER", None),
            pass_=(play_poisoned.startswith("raise:") and "helper" in play_poisoned
                   and played_clean is True
                   and m is not None
                   and os.path.abspath(m.__file__) == os.path.join(snap2, "helper.py")
                   and getattr(m, "MARKER", None) == "frozen"))
    sys.modules.pop("helper", None)
    sys.modules.pop("eval_v53", None)


# ================= W04: dep data files not frozen =================
def w04():
    dep_src = (
        "import os\n"
        "_H = os.path.dirname(os.path.abspath(__file__))\n"
        "FN = 'cfg.json'\n"
        "with open(os.path.join(_H, 'weights.json')) as f:\n    W = f.read()\n"
        "with open(os.path.join(_H, FN)) as f:\n    C = f.read()\n"
        "with open(os.path.join(_H, 'out.json'), 'w') as f:\n    f.write('x')\n"
    )
    files = {
        "examiner.py": HDR +
            '_s = importlib.util.spec_from_file_location("dep", os.path.join(os.path.dirname(__file__), "dep.py"))\n'
            '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n',
        "dep.py": dep_src,
        "weights.json": '{"w": 1}\n',
        "cfg.json": '{"c": 2}\n',
    }
    tmp, origin, snap = mk_origin(files)
    exam = os.path.join(origin, "examiner.py")
    with open(exam, "rb") as f:
        raw = f.read()
    try:
        deps = CT._selection_snapshot_deps(exam, raw, snap)
        outcome = "froze"
    except SystemExit as e:
        deps, outcome = [], "die:" + str(e)[:120]
    by_file = {d["file"]: d for d in deps}
    if MODE == "bug":
        rec("w04_dep_data", outcome=outcome, deps=sorted(by_file),
            pass_=(outcome == "froze" and sorted(by_file) == ["dep.py"]),
            note="baseline: dep-level data files not frozen")
    else:
        ok = (outcome == "froze"
              and by_file.get("weights.json", {}).get("kind") == "data"
              and by_file.get("cfg.json", {}).get("kind") == "data"
              and by_file.get("dep.py", {}).get("kind") == "code"
              and "out.json" not in by_file
              and md5_of(os.path.join(snap, "weights.json")) == by_file["weights.json"]["md5"]
              and md5_of(os.path.join(snap, "cfg.json")) == by_file["cfg.json"]["md5"])
        rec("w04_dep_data", outcome=outcome,
            deps={k: sorted(v) for k, v in by_file.items()}, pass_=ok)


# ================= W09: absolute-path loads bypass the snapshot =====
def w09():
    tmp = tempfile.mkdtemp(prefix="wfix9_")
    origin = os.path.join(tmp, "origin")
    snap = os.path.join(tmp, "snap")
    os.makedirs(origin)
    os.makedirs(snap)
    abs_leaf = os.path.join(origin, "leaf.py")
    mid_src = (
        "import importlib.util\n"
        f"_s = importlib.util.spec_from_file_location('leaf', {abs_leaf!r})\n"
        "_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n"
        "VALUE = _m.VALUE\n"
    )
    files = {
        "examiner.py": HDR +
            '_s = importlib.util.spec_from_file_location("mid", os.path.join(os.path.dirname(__file__), "mid.py"))\n'
            '_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n',
        "mid.py": mid_src,
        "leaf.py": "VALUE = 'original'\n",
    }
    for rel, content in files.items():
        with open(os.path.join(origin, rel), "w") as f:
            f.write(content)
    exam = os.path.join(origin, "examiner.py")
    with open(exam, "rb") as f:
        raw = f.read()
    try:
        deps = CT._selection_snapshot_deps(exam, raw, snap)
        outcome = "froze"
    except SystemExit as e:
        deps, outcome = [], "die:" + str(e)[:160]
    # tamper with the origin leaf AFTER freezing, then run the snapshot mid
    with open(abs_leaf, "w") as f:
        f.write("VALUE = 'tampered'\n")
    val, run_err = None, ""
    if outcome == "froze" and os.path.isfile(os.path.join(snap, "mid.py")):
        try:
            val = load_module_from(os.path.join(snap, "mid.py"),
                                   "mid_snap").VALUE
        except BaseException as e:  # noqa: BLE001
            run_err = f"{type(e).__name__}:{str(e)[:100]}"
    by_file = {d["file"]: d for d in deps}
    if MODE == "bug":
        rec("w09_dep_absolute", outcome=outcome, value_after_tamper=val,
            pass_=(outcome == "froze" and val == "tampered"),
            note="baseline: snapshot mid still reads the origin leaf")
    else:
        mid = by_file.get("mid.py", {})
        leaf_orig_md5 = hashlib.md5(b"VALUE = 'original'\n").hexdigest()
        ok = (outcome == "froze" and val == "original"
              and mid.get("md5") == md5_of(os.path.join(origin, "mid.py"))
              and mid.get("snapshot_md5") == md5_of(os.path.join(snap, "mid.py"))
              and bool(mid.get("path_rewrites"))
              and by_file.get("leaf.py", {}).get("md5") == leaf_orig_md5)
        rec("w09_dep_absolute", outcome=outcome, value_after_tamper=val,
            run_err=run_err, mid_entry=mid, pass_=ok)

    # root-level absolute load in the examiner itself
    files2 = {
        "examiner.py": HDR +
            f"_s = importlib.util.spec_from_file_location('leaf', {abs_leaf!r})\n"
            "_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n",
        "leaf.py": "VALUE = 'original'\n",
    }
    tmp2, origin2, snap2 = mk_origin(files2)
    # rebuild with THIS origin's absolute path
    abs_leaf2 = os.path.join(origin2, "leaf.py")
    with open(os.path.join(origin2, "examiner.py"), "w") as f:
        f.write(HDR +
                f"_s = importlib.util.spec_from_file_location('leaf', {abs_leaf2!r})\n"
                "_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n")
    exam2 = os.path.join(origin2, "examiner.py")
    with open(exam2, "rb") as f:
        raw2 = f.read()
    try:
        deps2 = CT._selection_snapshot_deps(exam2, raw2, snap2)
        outcome2 = "froze:" + ",".join(sorted(d["file"] for d in deps2))
    except SystemExit as e:
        outcome2 = "die:" + str(e)[:160]
    if MODE == "bug":
        rec("w09_root_absolute", outcome=outcome2,
            pass_=(outcome2.startswith("froze") and "leaf.py" in outcome2),
            note="baseline copies the absolute target (copy is never read)")
    else:
        rec("w09_root_absolute", outcome=outcome2,
            pass_=(outcome2.startswith("die:") and "absolute" in outcome2),
            note="examiner bytes are its identity: cannot redirect, must die")


# ================= W10: parameter shadowing =================
def _w10_case(name, examiner_src, bug_expect, fixed_expect, mention=None):
    files = {"examiner.py": examiner_src, "dep.py": "X = 1\n"}
    tmp, origin, snap = mk_origin(files)
    exam = os.path.join(origin, "examiner.py")
    with open(exam, "rb") as f:
        raw = f.read()
    try:
        deps = CT._selection_snapshot_deps(exam, raw, snap)
        outcome = "froze"
        ok_deps = sorted(d["file"] for d in deps) == ["dep.py"]
    except SystemExit as e:
        outcome, ok_deps = "die", (mention is None or mention in str(e))
    expect = bug_expect if MODE == "bug" else fixed_expect
    good = (outcome == "froze" and ok_deps) if expect == "freeze" else \
        (outcome == "die" and ok_deps)
    rec(name, outcome=outcome, expect=expect, pass_=good)


_G = 'SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dep.py")\n'
def w10():
    _w10_case("w10_param_shadow",
              HDR + _G +
              'def _load(SRC):\n'
              '    s = importlib.util.spec_from_file_location("dep", SRC)\n'
              '    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n',
              "freeze", "die", "not statically reducible")
    _w10_case("w10_local_assign_shadow",
              HDR + _G +
              'def _load():\n'
              '    SRC = os.getcwd()\n'
              '    s = importlib.util.spec_from_file_location("dep", SRC)\n'
              '    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n',
              "freeze", "die", "not statically reducible")
    _w10_case("w10_global_ok",
              HDR + _G +
              'def _load(other):\n'
              '    s = importlib.util.spec_from_file_location("dep", SRC)\n'
              '    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n',
              "freeze", "freeze")
    _w10_case("w10_local_binding_wins",
              HDR +
              'def _load(SRC):\n'
              '    p = os.path.join(os.path.dirname(__file__), "dep.py")\n'
              '    s = importlib.util.spec_from_file_location("dep", p)\n'
              '    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n',
              "freeze", "freeze")
    _w10_case("w10_global_decl",
              HDR + _G +
              'def _load():\n'
              '    global SRC\n'
              '    s = importlib.util.spec_from_file_location("dep", SRC)\n'
              '    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n',
              "freeze", "freeze")


# ================= W11: snapshot rename breaks policy_* rule =======
def _eval_loader_rule(path):
    """eval_v53.load_agent_fn's dispatch, verbatim semantics."""
    s = importlib.util.spec_from_file_location("w11_probe_mod", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if os.path.basename(path).startswith("policy_") and hasattr(m, "make_agent"):
        return m.make_agent()
    return m.agent


def w11():
    files = {
        "policy_demo.py": (
            "def make_agent():\n    return ('made', 42)\n"
            "agent = ('module', 0)\n"),
    }
    tmp, origin, snap = mk_origin(files)
    exam = os.path.join(origin, "policy_demo.py")
    with open(exam, "rb") as f:
        raw = f.read()
    sel = {"examiner": exam, "examiner_md5": md5_of(exam),
           "_examiner_bytes": raw, "dir": snap, "public": {}}
    CT._selection_freeze_examiner(sel)
    exec_path = sel.get("examiner_exec")
    got = _eval_loader_rule(exec_path)
    if MODE == "bug":
        rec("w11_policy_name", exec_base=os.path.basename(exec_path), got=got,
            pass_=(os.path.basename(exec_path) == "examiner_snapshot.py"
                   and got == ("module", 0)),
            note="baseline rename routes the loader to module-level agent")
    else:
        rec("w11_policy_name", exec_base=os.path.basename(exec_path), got=got,
            pass_=(os.path.basename(exec_path) == "policy_demo.py"
                   and got == ("made", 42)))


w03()
w04()
w09()
w10()
w11()
fails = [k for k, v in results.items() if not v.get("pass_")]
print(json.dumps({"mode": MODE, "cases": results, "failures": fails,
                  "n_pass": len(results) - len(fails),
                  "n_total": len(results)}, indent=1), flush=True)
