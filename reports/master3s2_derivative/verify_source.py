"""Verify that only the opponent root readout differs from the registered Master3s2."""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = "sources/master3s2/master3s2.py"
ORIGINAL_SHA256 = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
BEGIN = "# BEGIN ORIGINAL OPPONENT READOUT"
END = "# END ORIGINAL OPPONENT READOUT"
ORIGINAL_CALL = "    lwo, lbo, _ = net(xo.astype(np.float32), _SID)"
NEW_CALL = "    lwo, lbo = _opponent_root_readout(xo, lwo, lbo, players, op)"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique_index(lines, item):
    if lines.count(item) != 1:
        raise ValueError(f"Expected one exact source line: {item}")
    return lines.index(item)


def embedded_weights(source):
    payloads = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"_WEIGHTS", "_VALUE", "_ROLL"}:
            raw = base64.b64decode(ast.literal_eval(node.value), validate=True)
            payloads[target.id] = {"bytes": len(raw), "sha256": digest(raw)}
    return payloads


def verify(original, candidate):
    if digest(original) != ORIGINAL_SHA256:
        raise ValueError("Examiner bytes do not match the registered original")
    if not 0 < len(candidate) <= 10_000_000:
        raise ValueError("Candidate exceeds the platform's single-file limit")
    source = original.decode("utf-8").replace("\r\n", "\n")
    text = candidate.decode("utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    begin = unique_index(lines, BEGIN)
    end = unique_index(lines, END)
    root = unique_index(lines, ORIGINAL_CALL)
    changed = unique_index(lines, NEW_CALL)
    if not begin < end < root or changed != root + 1:
        raise ValueError("Readout or call is outside the opponent-root insertion site")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "_opponent_root_readout"
               for n in ast.parse("\n".join(lines[begin + 1:end])).body):
        raise ValueError("Missing opponent readout definition")
    del lines[changed]
    del lines[begin:end + 1]
    # Blank lines used to separate the inserted block precede the original function.
    # Infer only that run of whitespace; all source statements must match exactly.
    while "\n".join(lines) != source and begin < len(lines) and lines[begin] == "":
        del lines[begin]
    if "\n".join(lines) != source:
        raise ValueError("A change remains outside the opponent readout and its one call")
    weights = embedded_weights(source)
    if embedded_weights(text) != weights:
        raise ValueError("An embedded examiner network changed")
    return {"original_sha256": digest(original), "candidate_sha256": digest(candidate),
            "candidate_bytes": len(candidate), "embedded_original_weights": weights,
            "single_opponent_root_call": True, "original_restored": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    original = subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{SOURCE}"])
    print(json.dumps(verify(original, args.candidate.read_bytes()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
