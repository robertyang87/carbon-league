"""Build a reproducible single-file candidate from two frozen Sapling project assets."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
STUDENT = ROOT / "data/weights/sapling_1200k/master6_actor_v1.npz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path, value):
    if path.exists() and path.read_bytes() != value:
        raise ValueError(f"frozen artifact differs: {path.name}")
    if not path.exists():
        tmp = path.with_suffix(".writing")
        tmp.write_bytes(value)
        tmp.replace(path)


def build():
    p = json.loads((HERE / "protocol.json").read_bytes())
    if digest(PARENT) != p["parent_pack_sha256"] or digest(STUDENT) != p["student_npz_sha256"]:
        raise ValueError("wrong parent or student checkpoint")
    out = HERE / "candidate.py"
    seal(out, PARENT.read_bytes() + b"\n\n" + (HERE / "runtime.py").read_bytes())
    if out.stat().st_size > 10_000_000:
        raise ValueError("package too large")
    release = {"parent_pack_sha256": digest(PARENT), "candidate_pack_sha256": digest(out),
               "parent_weight_manifest_sha256": p["parent_weight_manifest_sha256"],
               "student_npz_sha256": digest(STUDENT), "runtime_sha256": digest(HERE / "runtime.py"),
               "protocol_sha256": digest(HERE / "protocol.json"), "bytes": out.stat().st_size,
               "expert_weight": 0.75, "student_weight": 0.25, "views": [0, 1, 2, 3]}
    seal(HERE / "release.json", (json.dumps(release, indent=2) + "\n").encode())
    return release


if __name__ == "__main__":
    print(build())
