"""F11: producer resume with a 1-byte-changed weights file must be judged
identity-mismatch, never silently adopted. STUB mode, real CLI."""
import json, os, subprocess, sys
CODE = os.path.expanduser("~/workspace/final-accept/code")
WORK = os.path.expanduser("~/workspace/final-accept/work")
PY = os.path.expanduser("~/workspace/integ-accept/venv/bin/python")
out = os.path.join(WORK, "ident_out")
import shutil
if os.path.exists(out): shutil.rmtree(out)
w1 = os.path.join(WORK, "w1.bin"); w2 = os.path.join(WORK, "w2.bin")
open(w1, "wb").write(b"WEIGHTS" + bytes(range(256)) * 4)
b = bytearray(open(w1, "rb").read()); b[100] ^= 0xFF
open(w2, "wb").write(bytes(b))

def run(weights):
    env = dict(os.environ, SPGEN_STUB="1")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([PY, os.path.join(CODE, "producer", "selfplay_gen.py"),
                        "--weights", weights, "--seed0", "200", "--seed1", "202",
                        "--out", out, "--workers", "1", "--max-games", "2",
                        "--name", "claude-AlphaCarbon-Master"],
                       capture_output=True, text=True, env=env, timeout=300)
    line = [l for l in r.stdout.splitlines() if "SPGEN-RESULT" in l]
    res = json.loads(line[0].split("SPGEN-RESULT ", 1)[1]) if line else {}
    return r, res

r1, s1 = run(w1)
print("run1 rc", r1.returncode, "identity:", s1.get("identity", {}).get("weights_md5"), "mismatches:", s1.get("identity_mismatches"), "games:", s1.get("games"))
idx1 = json.load(open(os.path.join(out, "index.json")))
r2, s2 = run(w2)
print("run2 rc", r2.returncode, "identity:", s2.get("identity", {}).get("weights_md5"), "mismatches:", s2.get("identity_mismatches"), "adopted:", s2.get("adopted"))
print("run2 stdout flags:", [l for l in r2.stdout.splitlines() if "MISMATCH" in l or "ERROR" in l][:4])
idx2 = json.load(open(os.path.join(out, "index.json")))
same_entries = idx1 == idx2
print("index unchanged after changed-weights resume:", same_entries)
ok = (s1.get("identity", {}).get("weights_md5") != s2.get("identity", {}).get("weights_md5")
      and (s2.get("identity_mismatches", 0) >= 1 or "MISMATCH" in r2.stdout)
      and s2.get("adopted", 0) == 0)
print(("PASS " if ok else "FAIL ") + "F11 identity: 1-byte weights change -> mismatch judged, 0 adopted")
sys.exit(0 if ok else 1)
