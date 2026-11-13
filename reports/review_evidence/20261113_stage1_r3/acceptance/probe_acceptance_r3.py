"""S1 R3 read-only source/provenance and isolated acceptance probes.

Only AST-extracted acceptance functions run with synthetic fixture stubs.
No model load, training, simulation, deployment or business-source edit.
Usage: python probe_acceptance_r3.py --repo /path/to/checkout --out /tmp/s1-r3-acceptance
"""
import argparse
import ast
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types

import numpy as np


def discover_repo():
    for p in (Path.cwd(), Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        for candidate in (p, p / "repo"):
            if (candidate / ".git").exists() and (candidate / "code/stage0_teacher.py").is_file():
                return candidate
    return None


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--repo", default=os.environ.get("CARBON_REPO"))
ap.add_argument("--out", default=os.environ.get("CARBON_REVIEW_OUT", "/tmp/s1-r3-acceptance"))
a = ap.parse_args()
repo = Path(a.repo).resolve() if a.repo else discover_repo()
if repo is None:
    ap.error("provide --repo or CARBON_REPO")
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)


def git(*args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_function(rel, name, ns):
    path = repo / rel
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def observe(fn):
    stream = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(stream):
            fn()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    return {"accepted": error is None, "error": error, "stdout": stream.getvalue()}


result = {"reviewed_sha": git("rev-parse", "HEAD"),
          "scope": "AST-isolated synthetic acceptance functions; source hashes and Git inventory"}
result["packs"] = {}
for name in ("cnet2_np.py", "cnet2_search.py"):
    content = (repo / "code" / name).read_bytes()
    pack = (repo / "code/packs" / name).read_bytes()
    old = subprocess.check_output(["git", "-C", str(repo), "show", "7b025ae5:code/" + name])
    result["packs"][name] = {
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "pack_sha256": hashlib.sha256(pack).hexdigest(),
        "source_equals_pack": content == pack,
        "source_unchanged_since_7b025ae5": content == old,
    }

inventory = git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
result["canonical_heads_named_files"] = [p for p in inventory if Path(p).name == "heads_v2.npz"]
result["acceptance_files"] = [p for p in inventory if p.startswith("automation/acceptance_s1/")]
result["acceptance_syntax"] = {}
for p in result["acceptance_files"]:
    if p.endswith(".py"):
        ast.parse((repo / p).read_text())
        result["acceptance_syntax"][p] = "PASS"

result["a3_compare"] = {}
with tempfile.TemporaryDirectory(prefix="s1-r3-acceptance-") as tmp:
    compare = load_function("automation/acceptance_s1/test_s12_a3.py", "compare",
                            {"json": json, "os": os, "HERE": tmp})
    one = {"game": "fixture", "t": 0, "acts": {}, "guard_moves": 0,
           "guard_recruit": 0, "errors": 0}
    cases = {"empty_original_counterexample": [],
             "equal_errors_original_counterexample": [dict(one, errors=4)],
             "single_position_only": [one],
             "63_duplicate_positions": [one] * 63,
             "63_unique_positive": [dict(one, game=f"g{i // 21}", t=3*(i % 21)) for i in range(63)],
             "50_unique_shrunken": [dict(one, game=f"g{i // 21}", t=3*(i % 21)) for i in range(50)]}
    for key, data in cases.items():
        for side in ("old", "new"):
            (Path(tmp) / f"guard_parity_{side}.json").write_text(json.dumps(data))
        result["a3_compare"][key] = observe(compare)


class FakeNumpy:
    def load(self, path):
        return {}

    def __getattr__(self, name):
        return getattr(np, name)


class FakeSim:
    def __init__(self, cfg):
        pass

    def load_observation(self, root):
        pass


def teacher_case(kind, count=50, duplicate=False):
    calls = {"off": 0, "on": 0}

    def evaluate(*args, **kwargs):
        on = kwargs.get("guard") == "on"
        calls["on" if on else "off"] += 1
        q = np.array([.75, .25])
        Q = np.array([.8, .7])
        if kind == "all_nan" or (kind == "guard_on_nan" and on):
            q[:] = np.nan
            Q[:] = np.nan
        elif kind == "unnormalized":
            q = np.array([.75, .75])
        elif kind == "negative_mass":
            q = np.array([1.25, -.25])
        return {"q": q, "Q_mean": Q,
                "arms": [np.array([0]), np.array([1])],
                "guard": {"guard_moves": 0, "guard_recruit": 0}}

    fixture = [({"step": 51 + (0 if duplicate else i),
                 "fixture_identity": 0 if duplicate else i},
                0, None, None, None, None, None, None) for i in range(count)]
    main = load_function("automation/acceptance_s1/test_s12_a3_teacher.py", "main", {
        "np": FakeNumpy(), "os": os, "HERE": str(out), "time": time,
        "NP": types.SimpleNamespace(Policy=lambda w: object()),
        "CachePolicy": lambda w: object(),
        "roots": lambda n: fixture,
        "FastCarbon": FakeSim, "CFG": {},
        "T": types.SimpleNamespace(evaluate_root=evaluate),
    })
    r = observe(main)
    r["calls"] = calls
    return r


result["a3_teacher"] = {key: teacher_case(key) for key in
                         ("all_nan", "unnormalized", "guard_on_nan", "negative_mass", "normal")}
result["a3_teacher"]["single_root_only"] = teacher_case("normal", 1)
result["a3_teacher"]["40_unique_shrunken"] = teacher_case("normal", 40)
result["a3_teacher"]["50_duplicate_roots"] = teacher_case("normal", 50, duplicate=True)

# Exercise the REAL drive_agent producer with synthetic replay files and an
# explicit no-op agent; this verifies the new compare keys match emitted data.
with tempfile.TemporaryDirectory(prefix="s1-r3-drive-") as tmp:
    paths = []
    for i in range(3):
        path = Path(tmp) / f"game{i:04d}.json.gz"
        steps = [[{"observation": {"step": t, "players": [], "carbon": [], "trees": {}}}]
                 for t in range(61)]
        with gzip.open(path, "wt") as fh:
            json.dump({"steps": steps}, fh)
        paths.append(str(path))

    def no_op_agent(obs):
        return {}
    no_op_agent.last_stats = {"guard_moves": 0, "guard_recruit": 0, "errors": 0}
    drive_results = {}
    for side in ("old", "new"):
        drive = load_function("automation/acceptance_s1/test_s12_a3.py", "drive_agent", {
            "np": FakeNumpy(), "os": os, "HERE": tmp, "CFG": {},
            "CS": types.SimpleNamespace(make_search_agent=lambda **kwargs: no_op_agent),
            "REPLAYS": paths, "WHICH": side, "json": json, "gzip": gzip})
        drive_results[side] = observe(drive)
    compare = load_function("automation/acceptance_s1/test_s12_a3.py", "compare",
                            {"json": json, "os": os, "HERE": tmp})
    records = json.loads((Path(tmp) / "guard_parity_new.json").read_text())
    result["actual_drive_agent_identity_positive"] = {
        "drive": drive_results, "compare": observe(compare),
        "record_fields": list(records[0]), "count": len(records),
        "unique_game_t": len({(r["game"], r["t"]) for r in records}),
        "fixture_scope": "Actual replay iterator/record producer; synthetic replay observations and no-op policy, no neural parity claim."}
assert not result["a3_compare"]["empty_original_counterexample"]["accepted"]
assert not result["a3_compare"]["equal_errors_original_counterexample"]["accepted"]
assert not result["a3_teacher"]["all_nan"]["accepted"]
assert not result["a3_teacher"]["unnormalized"]["accepted"]
assert result["a3_teacher"]["normal"]["accepted"]
assert not result["a3_teacher"]["guard_on_nan"]["accepted"]
assert not result["a3_teacher"]["negative_mass"]["accepted"]
assert not result["a3_compare"]["single_position_only"]["accepted"]
assert not result["a3_compare"]["63_duplicate_positions"]["accepted"]
assert result["a3_compare"]["63_unique_positive"]["accepted"]
assert result["actual_drive_agent_identity_positive"]["compare"]["accepted"]
assert result["a3_compare"]["50_unique_shrunken"]["accepted"]
assert result["a3_teacher"]["40_unique_shrunken"]["accepted"]
assert result["a3_teacher"]["50_duplicate_roots"]["accepted"]
result["limitations"] = [
    "Controlled outputs prove assertion coverage only; no claim that real model produced NaN or invalid probabilities.",
    "Canonical Studio heads/replays and new rerun output are not reconstructed by these stubs.",
    "Exit 0 means observations were generated and expected original-counterexample closures checked, not all review findings closed.",
]
(out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reviewed_sha": result["reviewed_sha"],
                  "a3_compare": {k: v["accepted"] for k, v in result["a3_compare"].items()},
                  "a3_teacher": {k: v["accepted"] for k, v in result["a3_teacher"].items()},
                  "output": str(out / "results.json")}, ensure_ascii=False))
