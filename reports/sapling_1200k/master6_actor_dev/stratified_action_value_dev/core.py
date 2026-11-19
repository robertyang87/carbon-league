"""Persistent state shared by the prospectively locked action-response study."""
import gzip
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent.parent
OLD = HERE.parent / "late_local_residual_dev"
KINDS = ("mine_hold", "bank_route", "evade", "ore_route")
TRAIN_SEEDS = (621476120, 621476121, 621477120, 621477121)
HOLD_SEEDS = (621461321, 621461322)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def value_hash(obj):
    return hashlib.sha256(json.dumps(obj, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)

def save(path, obj, immutable=True):
    encoded = (json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n").encode()
    if path.suffix == ".gz":
        encoded = gzip.compress(encoded, mtime=0)
    if path.exists() and immutable:
        if read(path) != obj:
            raise ValueError(f"immutable record changed: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_bytes(encoded)
    tmp.replace(path)
