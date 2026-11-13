"""Sealed-generation dataset fixtures for the weight-chain P1 reproductions.

Builds REAL sealed datasets: generation dir with complete field files,
manifest sha256 over the actual bytes, current.json pointer -- the exact
shape cnet2_build publishes and load_generation(verify=True) certifies.
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
import cnet2_feat as F  # noqa: E402

N = 4
# tok_kind first-10 pattern: exactly four 0s (collector slots) per row.
KIND10 = [0, 0, 0, 0, 1, 1, 1, 2, 2, 1]


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def field_arrays(seed=0, with_w=False, kind_torn=False):
    rng = np.random.default_rng(seed)
    n = N
    A = {}
    A["maps"] = rng.integers(0, 255, (n, F.P, F.N), dtype=np.uint8)
    A["scal"] = rng.random((n, F.NS), dtype=np.float32).astype(np.float16)
    A["tok_pos"] = rng.integers(0, F.N, (n, F.T), dtype=np.int16)
    kind = np.full((n, F.T), 2, dtype=np.int8)
    kind[:, :10] = 1 if kind_torn else KIND10
    A["tok_kind"] = kind
    A["tok_mask"] = np.ones((n, F.T), dtype=bool)
    A["tok_cargo"] = rng.random((n, F.T), dtype=np.float32).astype(np.float16)
    A["hist_cell"] = rng.integers(0, F.N, (n, F.T, F.H), dtype=np.int16)
    A["hist_cargo"] = rng.random((n, F.T, F.H), dtype=np.float32).astype(np.float16)
    A["hist_move"] = rng.integers(0, 5, (n, F.T, F.H), dtype=np.int8)
    A["uf"] = rng.random((n, F.OWN, F.UF), dtype=np.float32).astype(np.float16)
    A["mf"] = rng.random((n, F.OWN, 5, F.MF), dtype=np.float32).astype(np.float16)
    A["legal"] = np.ones((n, F.OWN, 5), dtype=bool)
    y = -np.ones((n, F.OWN), dtype=np.int8)
    y[:, :10] = 1
    A["y"] = y
    A["carbon"] = rng.integers(0, 255, (n, F.N), dtype=np.uint8)
    A["value"] = np.zeros(n, dtype=np.float16)
    A["teacher"] = np.ones(n, dtype=np.int8)
    A["gid"] = np.zeros(n, dtype=np.int32)
    A["step"] = np.full(n, 100, dtype=np.int16)
    if with_w:
        A["w"] = np.array([2, 2, 1, 1], dtype=np.float16)
    return A


def build_ds(root, gen_id="gen-AAAA", with_w=False, seed=0):
    """Create a sealed dataset at root; root flat files are faithful
    copies of the generation (the builder's published flat mirror)."""
    if os.path.exists(root):
        shutil.rmtree(root)
    gen_dir = os.path.join(root, "generations", gen_id)
    os.makedirs(gen_dir)
    A = field_arrays(seed=seed, with_w=with_w)
    fields = {}
    for k, arr in A.items():
        p = os.path.join(gen_dir, k + ".npy")
        np.save(p, arr)
        fields[k] = {"file": k + ".npy", "rows": N, "sha256": _sha(p)}
    meta = {"games": [{"val": False, "key": "g0"}], "total": N}
    with open(os.path.join(gen_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    man = {"generation": gen_id, "total": N, "weighted": bool(with_w),
           "fields": fields}
    with open(os.path.join(gen_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    with open(os.path.join(root, "current.json"), "w") as fh:
        json.dump({"generation": gen_id, "dir": f"generations/{gen_id}",
                   "total": N}, fh)
    # faithful flat mirror in the root
    for k in A:
        shutil.copy(os.path.join(gen_dir, k + ".npy"),
                    os.path.join(root, k + ".npy"))
    with open(os.path.join(root, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return root


def expected_uw_from_gen(ds, car_w=0.0, pl_w=3.0, base_w=1.0):
    """Hand-computed set_uw output from the SEALED generation's features."""
    ptr = json.load(open(os.path.join(ds, "current.json")))
    gd = os.path.join(ds, ptr["dir"])
    kind = np.load(os.path.join(gd, "tok_kind.npy"))
    y = np.load(os.path.join(gd, "y.npy"))
    n, S = y.shape
    uw = np.full((n, S), base_w, np.float32)
    uw[:, :10] = np.where(np.asarray(kind[:, :10]) == 0, car_w, pl_w)
    return uw
