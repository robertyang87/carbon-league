from hashlib import sha256
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "advantage_policy_update/v2_bc.py"
ROTATION = HERE.parent / "symmetry_value/rotate.py"
EXPECTED_PARENT = "cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a"


def build():
    parent = PARENT.read_bytes()
    if sha256(parent).hexdigest() != EXPECTED_PARENT:
        raise ValueError("Unexpected v4 source")
    geometry = ROTATION.read_text()
    marker = "def rotate_features(x):"
    if geometry.count(marker) != 1:
        raise ValueError("Unable to embed the proven 54-plane permutation")
    rotation = marker + geometry.split(marker, 1)[1].rstrip()
    runtime = (HERE / "runtime.py").read_text().strip()
    insert = (rotation + "\n\n\n" + runtime + "\n\n\n").replace("\n", "\r\n").encode()
    entry = b"def _act(obs, configuration=None, t0=None):"
    old = b"    lw, lb = _POLICY.forward(x, _SID, own_root=True)"
    new = b"    lw, lb = _root_rotation_policy(x, _SID)"
    if parent.count(entry) != 1 or parent.count(old) != 1:
        raise ValueError("Unexpected root entry point")
    source = parent.replace(old, new).replace(entry, insert + entry)
    if source.replace(insert, b"").replace(new, old) != parent:
        raise ValueError("Unable to recover v4 byte-for-byte")
    compile(source, "submission.py", "exec")
    (HERE / "submission.py").write_bytes(source)
    return dict(candidate_sha256=sha256(source).hexdigest(), parent_sha256=EXPECTED_PARENT,
                bytes=len(source), geometry_sha256=sha256(ROTATION.read_bytes()).hexdigest(),
                runtime_sha256=sha256((HERE / "runtime.py").read_bytes()).hexdigest())


if __name__ == "__main__":
    record = build()
    (HERE / "identity.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record)
