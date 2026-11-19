"""Build the training-only instrument from the exact frozen public entry."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "candidate.py"
PARENT_SHA = "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build():
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("starting policy changed")
    src = PARENT.read_text()
    pos = src.index("def _return_decode(")
    prefix, decoder = src[:pos], src[pos:]
    marker = "action = int(np.argmax(np.where(legal, combined, -np.inf)))"
    if decoder.count(marker) != 1:
        raise ValueError("unrecognized decoder")
    decoder = decoder.replace(marker, marker + "\n        action = _sample_slot(apply_residual, slot, action, states[1][3], combined, legal)")
    start = decoder.rindex("def _mixed_actions(")
    upstream, gate = decoder[:start], decoder[start:]
    if gate.count("return baseline") != 3 or gate.count("return candidate") != 1:
        raise ValueError("unrecognized safety gate")
    gate = gate.replace("    F = ", "    global _chosen_route\n    _chosen_route = False\n    F = ",1)
    gate = gate.replace("    return candidate", "    _chosen_route = True\n    return candidate")
    contents = prefix + upstream + gate + "\n" + (HERE / "sampler.py").read_text()
    ast.parse(contents)
    pack = HERE / "training_policy.py"
    if not pack.exists() or pack.read_text() != contents:
        pack.write_text(contents)
    return pack

if __name__ == "__main__":
    print(sha(build()))
