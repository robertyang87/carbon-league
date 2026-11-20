"""Rebuild the research-only single-file agent from its parent and verified weight changes."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1] / "cargo_tree_action_learning_dev/candidate.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rebuild():
    record = json.loads((HERE / "result.json").read_bytes())
    src = PARENT.read_bytes()
    patch = (HERE/"int8_delta.bin").read_bytes()
    if digest(src) != record["parent_sha256"] or digest(patch) != record["patch"]["sha256"]:
        raise ValueError("wrong parent or patch")
    lines = src.decode().splitlines(keepends=True)
    assignments = {n.targets[0].id:n for n in ast.parse(src).body if isinstance(n,ast.Assign)
                   and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    for name, row in zip(("_B1","_W64"),record["patch"]["towers"],strict=True):
        node = assignments[name]
        if node.lineno != node.end_lineno:
            raise ValueError("unexpected parent source layout")
        base = zlib.decompress(base64.b64decode(ast.literal_eval(node.value)))
        delta = zlib.decompress(patch[row["start"]:row["start"]+row["length"]])
        if digest(base)!=row["original_raw_sha256"] or len(delta)!=len(base):
            raise ValueError("wrong tower delta")
        updated = bytes(a^b for a,b in zip(base,delta,strict=True))
        if digest(updated)!=row["raw_sha256"]:
            raise ValueError("reconstructed tensor digest differs")
        payload = base64.b64encode(zlib.compress(updated,9)).decode()
        lines[node.lineno-1] = f"{name} = {payload!r}\n"
    package = "".join(lines).encode()
    if len(package)>10_000_000:
        raise ValueError("single-file size exceeds platform limit")
    return package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,help="optional new file; refuses overwrite")
    args = parser.parse_args()
    data = rebuild()
    if args.out:
        with args.out.open("xb") as f:
            f.write(data)
    print(f"bytes={len(data)} sha256={digest(data)}")
