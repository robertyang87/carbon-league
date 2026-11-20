"""Package a source-locked original player with a frozen neural root-action proposal."""
import argparse
import ast
import math
import hashlib
import json
import os
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PARENT = STUDY.parent / "cold_start/candidate.py"
PARENT_SHA = "7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def build():
    p = PARENT.read_bytes()
    if sha(p) != PARENT_SHA:
        raise ValueError("Source is not the frozen ds-pro-v1 parent")
    model = (STUDY / "model.py").read_bytes()
    frozen = (STUDY / "weights.json").read_bytes()
    weights = json.loads(frozen, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    registration = (STUDY / "train_manifest.json").read_bytes()
    identity = json.loads((HERE / "frozen_inputs.json").read_bytes())
    if (identity["model_sha256"] != sha(model) or identity["weights_sha256"] != sha(frozen) or
            weights["feature_sha256"] != sha(model) or weights["manifest"] != sha(registration)):
        raise ValueError("Archived model, training registration or source identity changed")
    n, h = len(weights["mean"]), len(weights["b0"])
    if not (n and h and len(weights["scale"]) == len(weights["w0"]) == n and len(weights["w1"]) == h and
            all(len(row) == h for row in weights["w0"]) and all(v > 0 for v in weights["scale"])):
        raise ValueError("Invalid dimensions or scale in the frozen return model")
    scalars = (weights["mean"] + weights["scale"] + weights["b0"] + weights["w1"] +
               [v for row in weights["w0"] for v in row])
    if not all(math.isfinite(v) for v in scalars):
        raise ValueError("Nonfinite frozen model")
    src = model.decode()
    tree = ast.parse(src)
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module}
    if imports - {"numpy", "math", "collections", "functools", "__future__"}:
        raise ValueError(f"Unexpected runtime dependencies: {imports}")
    body = []
    for n in tree.body:
        if isinstance(n,ast.ImportFrom) and n.module == "__future__":continue
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant):continue
        body.append(ast.get_source_segment(src, n))
    # Place all original model constants/functions inside their own closed
    # scope: helper names such as move/dist cannot shadow the parent globals.
    runtime = ("def _return_model_bind():\n" +
               textwrap.indent("\n\n".join(body), "    ") +
               "\n    return features, score\n\n_RETURN_FEATURES, _RETURN_SCORE = _return_model_bind()\n" +
               "_RETURN_WEIGHTS = " + repr(weights) + "\n\n" +
               (HERE / "offer.py").read_text() + "\n\n")
    header = b"def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    placement = b"    if len(cands) < 2:\r\n        _S_STATS['single'] += 1"
    if p.count(header)!=1 or p.count(placement)!=1:
        raise ValueError("The parent root-search contract changed")
    code = p.replace(header, runtime.replace("\n","\r\n").encode() + header)
    insert = b"    if _RETURN_ENABLED:\r\n        _return_menu(cands, players, me, step, home, most, t0, configuration, obs['carbon'])\r\n"
    code = code.replace(placement, insert + placement)
    if len(code) > 10_000_000:raise ValueError("Pack too large")
    compile(code, "return_rank_candidate.py", "exec")
    return code, dict(parent_sha256=PARENT_SHA, model_sha256=sha(model), weights_sha256=sha(frozen),
                      integration_sha256=sha((HERE/"offer.py").read_bytes()),
                      protocol_sha256=sha((HERE/"protocol.json").read_bytes()),
                      candidate_sha256=sha(code), bytes=len(code))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args=parser.parse_args()
    code, meta=build()
    if args.verify:
        if (HERE/"candidate.py").read_bytes()!=code or json.loads((HERE/"manifest.json").read_bytes())!=meta:
            raise ValueError("Frozen bytes differ from the current inputs")
    else:
        put(HERE/"candidate.py",code)
        put(HERE/"manifest.json",(json.dumps(meta,indent=2,sort_keys=True)+"\n").encode())
    print(meta)
