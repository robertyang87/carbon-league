"""Produce one immutable Saping agent with frozen v1 and the public risk head."""

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
SURV = HERE.parent
TRIAL = SURV.parent
ROOT = TRIAL.parents[3]
RULES = TRIAL / "opportunities.py"
PARENT = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
WEIGHTS = SURV / "weights.json"
PLAN = "9933468bb5c2655d4e779955aa1210d3b192a1604a8f46cf4a522414bf54856b"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        name = tmp.name
    try:
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"changed or stale frozen artifact {path}")
    finally:
        os.remove(name)


def build(directory=HERE):
    p = json.loads((HERE / "protocol.json").read_bytes())
    if (
        sha(HERE / "protocol.json") != PLAN
        or sha(SURV / "model.py") != p["model_code_sha256"]
        or sha(PARENT) != p["parent_pack_sha256"]
        or sha(WEIGHTS) != p["risk_weights_sha256"]
        or sha(RULES) != "2c44731ccc2480f9d7b0ead116fe4d8d1e54a1c2acafce03919b84a8d5a607a5"
    ):
        raise ValueError("risk, public legality or v1 identity changed")
    names = ["norm", "neighbors", "nearest", "state", "quote", "commitments"]
    txt = RULES.read_text()
    indexed = {
        n.name: ast.get_source_segment(txt, n)
        for n in ast.parse(txt).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    }
    rules = "\n\n".join(indexed[n] for n in names)
    extra = f"""
from collections import Counter
_SURV_MODEL = types.ModuleType("public_planter_risk")
exec({(SURV/'model.py').read_text()!r},_SURV_MODEL.__dict__)
_SURV_RULES = types.ModuleType("public_planter_enrollment")
_SURV_RULES.__dict__.update(Counter=Counter, MOVES=("NONE","UP","RIGHT","DOWN","LEFT"), BANK=("RECCOLLECTOR","RECPLANTER"),
    _ROUTE_DEST=np.array([[_SURV_MODEL.destination(i,k) for k in _SURV_MODEL.MOVES] for i in range(225)]),
    _ROUTE_DIST=np.array([[_SURV_MODEL.distance(i,j) for j in range(225)] for i in range(225)]), MAXDIST=15)
exec({rules!r},_SURV_RULES.__dict__)
_SURV_WEIGHTS = {json.loads(WEIGHTS.read_bytes())!r}
"""
    candidate = PARENT.read_bytes() + extra.encode() + (HERE / "runtime.py").read_bytes()
    compile(candidate, "candidate.py", "exec")
    if len(candidate) > 10_000_000:
        raise ValueError("single-file budget exceeded")
    path = directory / "candidate.py"
    freeze(path, candidate)
    manifest = {
        "protocol_sha256": PLAN,
        "parent_pack_sha256": sha(PARENT),
        "pack_sha256": sha(path),
        "bytes": len(candidate),
        "risk_weights_sha256": sha(WEIGHTS),
        "student_npz_sha256": p["student_npz_sha256"],
        "parent_weight_manifest_sha256": p["parent_weight_manifest_sha256"],
        "rules_sha256": sha(RULES),
        "model_code_sha256": sha(SURV / "model.py"),
        "build_sha256": sha(HERE / "build.py"),
        "runtime_sha256": sha(HERE / "runtime.py"),
    }
    freeze(
        directory / "release_manifest.json", (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build()))
