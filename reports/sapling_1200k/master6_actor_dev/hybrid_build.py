"""Install the updated student in the frozen shared-prefix 75/25 deployment policy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "data/weights/sapling_1200k"
REFERENCE = WEIGHTS / "cash_v3_agent.py"
REFERENCE_SHA = "4cd4c1fcf40f05217ecba2a59095ba40286c5f96017d2b15285c942a3b0f5348"
PARENT = WEIGHTS / "hybrid_it50_v3.py"
PARENT_SHA = "b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca"
PAYLOAD_NAMES = {"_W64", "_QMETA", "_WMETA"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payloads(source):
    lines = source.splitlines(keepends=True)
    blocks = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in PAYLOAD_NAMES:
            continue
        if target.id in blocks:
            raise ValueError(f"duplicate weight field {target.id}")
        ast.literal_eval(node.value)
        blocks[target.id] = (node.lineno - 1, node.end_lineno)
    if blocks.keys() != PAYLOAD_NAMES:
        raise ValueError("missing embedded student weights")
    return lines, blocks


def substitute(source, weights_source):
    original, locations = payloads(source)
    other, replacement = payloads(weights_source)
    for field in sorted(locations, key=lambda key: locations[key][0], reverse=True):
        a, b = locations[field]
        c, d = replacement[field]
        original[a:b] = other[c:d]
    return "".join(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student", type=Path)
    parser.add_argument("npz", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if digest(REFERENCE) != REFERENCE_SHA or digest(PARENT) != PARENT_SHA:
        raise ValueError("unexpected baseline policy")
    old, updated, parent = REFERENCE.read_text(), args.student.read_text(), PARENT.read_text()
    if substitute(updated, old) != old:
        raise ValueError("the student changed inference code or configuration")
    if substitute(parent, old) != parent:
        raise ValueError("baseline student and compound weights differ")
    source = substitute(parent, updated)
    if source != parent:
        old_name = '"student": "Sapling cash_v3"'
        if source.count(old_name) != 1:
            raise ValueError("unexpected embedded student descriptor")
        source = source.replace(old_name, '"student": "Sapling Master6 actor V1"')
    output = source.encode()
    if len(output) > 10_000_000:
        raise ValueError("agent size limit")
    args.output.write_bytes(output)
    paths = [REFERENCE, PARENT, args.student, args.npz, args.output, Path(__file__).resolve()]
    def path_key(path):
        absolute = path.resolve()
        return str(absolute.relative_to(ROOT) if absolute.is_relative_to(ROOT) else absolute)
    receipt = {"teacher_weight": 0.75, "student_weight": 0.25, "guard": False,
               "search": False, "development_seed": 621458920,
               "input_and_output_sha256": {path_key(p): digest(p) for p in paths}}
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
