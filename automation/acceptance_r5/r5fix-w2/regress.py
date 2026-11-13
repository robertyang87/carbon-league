"""Regression: legacy flat dataset and direct-generation directory."""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import fixture  # noqa: E402

PY = sys.executable
BASE = os.path.join(HERE, "data_regress")
shutil.rmtree(BASE, ignore_errors=True)
os.makedirs(BASE)


def run_load(ds):
    r = subprocess.run([PY, os.path.join(HERE, "child_load.py"), ds],
                       capture_output=True, text=True,
                       env={**os.environ, "BC_DEVICE": "cpu"})
    for line in r.stdout.splitlines():
        if line.startswith("VERDICT "):
            return json.loads(line[8:])
    return {"ok": False, "error": r.stdout[-300:] + r.stderr[-300:]}


# legacy: root files only, no pointer/manifest/sidecar; w present
ds = os.path.join(BASE, "legacy")
os.makedirs(ds)
A = fixture.field_arrays(with_w=True)
for k, a in A.items():
    np.save(os.path.join(ds, k + ".npy"), a)
with open(os.path.join(ds, "meta.json"), "w") as fh:
    json.dump({"games": [{"val": False, "key": "g0"}], "total": fixture.N}, fh)
v = run_load(ds)
print("legacy load:", v.get("ok"), "w=", v.get("w"), "mode=", v.get("mode"))
assert v.get("ok") and v.get("w") == [2.0, 2.0, 1.0, 1.0] and v.get("mode") == "legacy-unverified"

# torn legacy: truncate one field -> refuse by row-count cross-check
np.save(os.path.join(ds, "maps.npy"), A["maps"][:3])
v = run_load(ds)
print("torn legacy refused:", not v.get("ok"), (v.get("systemexit") or "")[:80])
assert not v.get("ok")

# direct generation dir: pass the generation directory itself
sealed = fixture.build_ds(os.path.join(BASE, "sealed"), with_w=True)
gen_dir = os.path.join(sealed, "generations", "gen-AAAA")
v = run_load(gen_dir)
print("direct gen load:", v.get("ok"), "w=", v.get("w"), "mode=", v.get("mode"))
assert v.get("ok") and v.get("w") == [2.0, 2.0, 1.0, 1.0] and v.get("mode") == "generation-owned"
print("REGRESS OK")
