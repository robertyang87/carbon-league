"""Paths and historical fixture extraction for the runtime fix review."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"


def repo_path():
    configured = os.environ.get("CARBON_REPO")
    candidates = [Path(configured).expanduser().resolve()] if configured else Path(__file__).resolve().parents
    for path in candidates:
        if (path / "code" / "fastsim.py").is_file():
            return path
    raise RuntimeError("Cannot find code/fastsim.py; set CARBON_REPO to the repository root")


def output_path():
    return Path(os.environ.get("CARBON_REVIEW_OUT", "/tmp/carbon-rereview-3/runtime")).expanduser().resolve()


def setup():
    repo, out = repo_path(), output_path()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "code"))
    return repo, out


def fixture_weights():
    """Extract literal model payload only; never execute the historical submission."""
    repo, out = setup()
    import numpy as np
    import cnet2_pack

    source = repo / "code" / "exp" / "cnet2_search_final_packed.py"
    raw_source = source.read_bytes()
    wanted = {"_Q1", "_M1", "_B1"}
    literals = {}
    for node in ast.parse(raw_source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in wanted:
                literals[key] = ast.literal_eval(node.value)
    if set(literals) != wanted:
        raise RuntimeError("Historical packed fixture no longer has the expected literal payload")
    weights = cnet2_pack.dequantize(
        zlib.decompress(base64.b64decode(literals["_B1"])), literals["_Q1"]
    )
    weights["meta"] = np.array(literals["_M1"])
    np.savez(out / "baseline.npz", **weights)
    provenance = {
        "source": str(source.relative_to(repo)),
        "source_sha256": hashlib.sha256(raw_source).hexdigest(),
        "meta": json.loads(literals["_M1"]),
        "meaning": "Historical b128 fixture, not current Echo2/n1r online weights",
    }
    (out / "fixture_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return weights


def emit(name, rows):
    _, out = setup()
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (out / name).write_text(text)
    print(text, end="")
