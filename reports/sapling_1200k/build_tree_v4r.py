"""Embed Carbon League's own V3 and V4 int8 payloads, each with identity pinned."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "data/weights/sapling_1200k"
SOURCES = {"cash_v3_agent.py": "4cd4c1fcf40f05217ecba2a59095ba40286c5f96017d2b15285c942a3b0f5348",
           "tree_v4_agent.py": "c656f78327f513a2eb502eae66c7baa98166633018eeae4cba648a65daa95387"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    sources = {}
    for name, expected in SOURCES.items():
        raw = (WEIGHTS / name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("changed input: " + name)
        sources[name] = raw.decode()
    v3, suffix = sources["cash_v3_agent.py"].split("\n_AG = None\n")
    if "def agent(" not in suffix:
        raise ValueError("missing old V3 entry point")
    payload = {}
    for node in ast.parse(sources["tree_v4_agent.py"]).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ("_W64", "_QMETA", "_WMETA"):
                payload[key] = ast.literal_eval(node.value)
    if set(payload) != {"_W64", "_QMETA", "_WMETA"}:
        raise ValueError("incomplete V4 payload")
    weights = "\n".join(f"_V4{k} = {v!r}" for k, v in sorted(payload.items()))
    runtime = Path(__file__).with_name("tree_v4r_runtime.py").read_text()
    result = (v3 + "\n" + weights + "\n" + runtime).encode()
    if len(result) > 10_000_000:
        raise ValueError("pack exceeds submission limit")
    path = WEIGHTS / "tree_v4r_agent.py"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(result)
    tmp.replace(path)
    print(path, len(result), sha(result))


if __name__ == "__main__":
    main()
