"""Read-only S1 provenance checks and isolated acceptance-assertion probes.

This does not load weights, simulate games, train, contact hosts, or edit source.
AST-extracted test functions are exercised with explicitly synthetic fixtures.

Usage: python probe_docs.py --repo /path/to/carbon-league --out /tmp/s1-docs
CARBON_REPO and CARBON_REVIEW_OUT provide equivalent defaults. Without --repo,
the script locates the surrounding repository (including the isolated review
layout with a sibling repo/ directory). Results default to this script's directory.
"""
import argparse
import ast
import contextlib
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

SCRIPT_DIR = Path(__file__).resolve().parent


def discover_repo():
    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents, Path.cwd()):
        for candidate in (parent, parent / "repo"):
            if (candidate / ".git").exists() and (candidate / "code/stage0_teacher.py").is_file():
                return candidate
    return None


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", default=os.environ.get("CARBON_REPO"), help="Repository root; defaults to CARBON_REPO or discovery")
parser.add_argument("--out", default=os.environ.get("CARBON_REVIEW_OUT", str(SCRIPT_DIR)), help="Evidence output directory")
args = parser.parse_args()
repo_path = Path(args.repo).expanduser() if args.repo else discover_repo()
if repo_path is None:
    parser.error("cannot locate repository; provide --repo or CARBON_REPO")
ROOT = repo_path.resolve()
OUT = Path(args.out).expanduser().resolve()
if not (ROOT / "automation/acceptance_s1/test_s12_a3.py").is_file():
    parser.error(f"S1 acceptance source missing from repository: {ROOT}")
OUT.mkdir(parents=True, exist_ok=True)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def function(rel, name, namespace):
    src = (ROOT / rel).read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / rel), "exec"), namespace)
    return namespace[name]


result = {"head": git("rev-parse", "HEAD"), "scope": "read-only static and AST-isolated test assertions"}
result["packs"] = {}
for name in ("cnet2_np.py", "cnet2_search.py"):
    main = hashlib.sha256((ROOT / "code" / name).read_bytes()).hexdigest()
    pack = hashlib.sha256((ROOT / "code/packs" / name).read_bytes()).hexdigest()
    result["packs"][name] = {"source_sha256": main, "pack_sha256": pack, "equal": main == pack}
result["baseline"] = {
    "sha": git("rev-parse", "a0d67414"),
    "ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "a0d67414", "HEAD"]).returncode == 0,
    "commits_since": git("rev-list", "--count", "a0d67414..HEAD"),
}
result["acceptance_files"] = git("ls-tree", "-r", "--name-only", "HEAD", "automation/acceptance_s1").splitlines()
result["syntax"] = {}
for rel in result["acceptance_files"]:
    if rel.endswith(".py"):
        ast.parse((ROOT / rel).read_text())
        result["syntax"][rel] = "PASS"

# A3 comparison accepts equal empty lists and equal error-bearing records.
result["a3_compare"] = {}
with tempfile.TemporaryDirectory(prefix="s1-docs-fixture-") as tmp:
    compare = function("automation/acceptance_s1/test_s12_a3.py", "compare", {"json": json, "os": os, "HERE": tmp})
    for label, data in (
        ("zero_positions", []),
        ("one_position_with_errors", [{"game": "synthetic", "t": 0, "acts": {}, "guard_moves": 0, "guard_recruit": 0, "errors": 4}]),
    ):
        for which in ("old", "new"):
            (Path(tmp) / f"guard_parity_{which}.json").write_text(json.dumps(data))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            compare()
        result["a3_compare"][label] = {"accepted": True, "stdout": output.getvalue()}

# A3 teacher acceptance main, with all teacher numerical outputs NaN.
# All objects are stubs; no model or simulator from the repository is imported.
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


def fake_evaluate(*args, **kwargs):
    return {"q": np.array([np.nan, np.nan]), "Q_mean": np.array([np.nan, np.nan]),
            "arms": [np.array([0]), np.array([1])], "guard": {"guard_moves": 0, "guard_recruit": 0}}


teacher_main = function("automation/acceptance_s1/test_s12_a3_teacher.py", "main", {
    "np": FakeNumpy(), "os": os, "HERE": str(OUT), "time": time,
    "NP": types.SimpleNamespace(Policy=lambda w: object()), "CachePolicy": lambda w: object(),
    "roots": lambda n: [(None, 0, None, None, None, None, None, None)] * n,
    "FastCarbon": FakeSim, "CFG": {}, "T": types.SimpleNamespace(evaluate_root=fake_evaluate),
})
output = io.StringIO()
with contextlib.redirect_stdout(output):
    teacher_main()
result["a3_teacher_nan_outputs"] = {"accepted": True, "stdout": output.getvalue()}
result["limitations"] = [
    "Synthetic stubs prove acceptance assertion gaps, not that actual Studio outputs were erroneous or NaN.",
    "Canonical heads_v2 and the original replay fixture are absent, so advertised canonical A1/A2/A3 runs were not reproduced.",
    "No training, model inference, remote calls, or production mutations.",
]
(OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
