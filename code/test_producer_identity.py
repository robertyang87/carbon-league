"""Regression fixture for the producer run identity (selfplay_gen).

_run_identity's code_md5 map used to omit cnet2_feat.py / cnet2_np.py --
the playing agent's feature and policy modules (cnet2_search imports
them unconditionally) -- so a feat/np change left the run identity, and
the equal-identity adoption check, unchanged: corpora produced under
different features were interchangeable on paper. This fixture builds a
stub module tree around the REAL selfplay_gen and pins that touching
either module changes the identity.

Run: python test_producer_identity.py  (from code/, or an acceptance tree)
"""
import json
import os
import subprocess
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "producer", "selfplay_gen.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


DRIVER = """
import sys, os, json
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "producer"))
sys.path.insert(0, root)
import selfplay_gen as SG
ident = SG._run_identity(None, {}, "a", "b")
print("IDENT:" + json.dumps(ident["code_md5"], sort_keys=True))
"""

STUBS = {
    "eval_v53.py": "CFG = {}\n",
    "fastsim.py": "class FastCarbon:\n    pass\n",
    "cnet2_search.py": "import cnet2_feat\nimport cnet2_np\n",
    "cnet2_feat.py": "MARKER = 'feat-v1'\n",
    "cnet2_np.py": "MARKER = 'np-v1'\n",
}


def run_identity(root):
    cp = subprocess.run([sys.executable, "-c", DRIVER, root],
                        capture_output=True, text=True, timeout=300, cwd=root)
    for line in cp.stdout.splitlines():
        if line.startswith("IDENT:"):
            return json.loads(line[len("IDENT:"):])
    raise RuntimeError("driver failed:\n" + cp.stdout[-1500:] + cp.stderr[-1500:])


def main():
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "code")
        os.makedirs(os.path.join(root, "producer"))
        for name, body in STUBS.items():
            with open(os.path.join(root, name), "w") as fh:
                fh.write(body)
        src = open(os.path.join(CODE, "producer", "selfplay_gen.py")).read()
        with open(os.path.join(root, "producer", "selfplay_gen.py"), "w") as fh:
            fh.write(src)

        base = run_identity(root)
        expect = {"eval_v53.py", "fastsim.py", "cnet2_search.py",
                  "cnet2_feat.py", "cnet2_np.py", "selfplay_gen.py"}
        check("identity code_md5 covers the agent's full dependency set",
              expect <= set(base))

        for dep in ("cnet2_feat.py", "cnet2_np.py"):
            with open(os.path.join(root, dep), "a") as fh:
                fh.write("# fixture touch\n")
            after = run_identity(root)
            check(f"identity changes when {dep} changes",
                  after.get(dep) != base.get(dep)
                  and all(after.get(k) == base.get(k) for k in base if k != dep))
            base = after

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
