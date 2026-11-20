from hashlib import sha256
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "ladder/prediction_feedback/package/candidate.py"
ROTATION = HERE.parent / "symmetry_value/rotate.py"
FAMILY = HERE.parent
PARENT_SHA = EXPECTED_PARENT = "8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43"


def build():
    parent = PARENT.read_bytes()
    if sha256(parent).hexdigest() != EXPECTED_PARENT:
        raise ValueError("Unexpected V2 source")
    geometry = ROTATION.read_text()
    marker = "def rotate_features(x):"
    if geometry.count(marker) != 1:
        raise ValueError("Unable to embed the proven 54-plane permutation")
    rotation = marker + geometry.split(marker, 1)[1].rstrip()
    runtime = (HERE / "runtime.py").read_text().strip()
    insert = (rotation + "\n\n\n" + runtime + "\n\n\n").replace("\n", "\r\n").encode()
    entry = b"def _act(obs, configuration=None, t0=None):"
    old = b"    lw, lb = _POLICY.forward(x, _SID)"
    new = b"    lw, lb = _root_rotation_policy(x, _SID)"
    if parent.count(entry) != 1 or parent.count(old) != 1:
        raise ValueError("Unexpected root entry point")
    source = parent.replace(old, new).replace(entry, insert + entry)
    if source.replace(insert, b"").replace(new, old) != parent:
        raise ValueError("Unable to recover V2 byte-for-byte")
    compile(source, "candidate.py", "exec")
    (HERE / "candidate.py").write_bytes(source)
    return dict(candidate_sha256=sha256(source).hexdigest(), parent_sha256=EXPECTED_PARENT,
                bytes=len(source), geometry_sha256=sha256(ROTATION.read_bytes()).hexdigest(),
                runtime_sha256=sha256((HERE / "runtime.py").read_bytes()).hexdigest())


if __name__ == "__main__":
    record = build()
    (HERE / "identity.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record)


def sha(data):
    return sha256(data).hexdigest()
