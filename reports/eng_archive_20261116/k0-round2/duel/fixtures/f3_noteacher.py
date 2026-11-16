"""Acceptance 3: no-teacher proof, two parts.
(a) Static: AST scan of k0_duel.py -- no import of any teacher module.
(b) Dynamic: a subprocess whose import system hard-blocks
    stage0_teacher and stage1_teacher_gate still imports k0_duel and
    completes a duel run; afterwards neither module is in sys.modules.
Run from the stage dir."""
import ast
import json
import subprocess
import sys

SRC = "k0_duel.py"
tree = ast.parse(open(SRC).read(), SRC)
mods = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        mods.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        mods.add(node.module.split(".")[0])
bad = [m for m in mods if "teacher" in m or m.startswith("stage")]
assert not bad, f"teacher imports present: {bad}"
print(f"F3a PASS: k0_duel imports = {sorted(mods)} (no teacher module)")

RUNNER = r'''
import importlib.abc
import json
import sys

BLOCKED = {"stage0_teacher", "stage1_teacher_gate"}


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked for no-teacher proof: {name}")
        return None


sys.meta_path.insert(0, Blocker())
sys.path.insert(0, ".")
import k0_duel as D

assert not (BLOCKED & set(sys.modules)), "teacher module got loaded"
specs, sha = D.load_roots("../fixtures/roots_f3.json", 200, 299)
res = D.run_duel("/home/hatch/workspace/heads-v2-diff/dgx.npz",
                 "/home/hatch/workspace/heads-v2-diff/dgx.npz",
                 specs, sha)
assert not (BLOCKED & set(sys.modules)), "teacher module got loaded"
D._atomic_dump(D._sanitize(res), "../evidence/f3_result.json")
print("F3b OK: duel completed with teacher imports blocked; "
      f"delta_u={res['summary']['delta_u']}")
'''

roots = json.load(open("../fixtures/roots.json"))
json.dump([r for r in roots if r["step"] == 285],
          open("../fixtures/roots_f3.json", "w"))
open("../evidence/f3_runner.py", "w").write(RUNNER)
p = subprocess.run([sys.executable, "../evidence/f3_runner.py"],
                   capture_output=True, text=True, cwd=".")
print(p.stdout.strip())
assert p.returncode == 0 and "F3b OK" in p.stdout, p.stderr[-2000:]
print("F3 PASS")
