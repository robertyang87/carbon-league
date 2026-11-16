"""Doctoring helper for k0_join negative/positive seam tests.

Copies the annotator fixture generation (out1) into a fresh dataset
root, applies ONE named mutation to its field arrays, and re-seals it
with the annotator's own publish math (field table rebuilt from the
mutated bytes; core/gen-id/self-seal/pointer recomputed exactly as
k0_annotate.publish does). The result is a fully seal-valid product
whose CONTENT carries the defect under test -- so a join refusal can
only be credited to the content gate, never to a broken seal.
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from k0_distill import construct_target  # noqa: E402

SRC = "/home/hatch/workspace/k0-round2/annotate/fixture/out1"
FIELDS = ("gid", "step", "seat", "arms", "n_arms", "priors", "q", "Q",
          "Q_std", "pi0", "pi1", "pi", "belief", "n_unknown")


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def mutate(name, A):
    if name == "seat_flip":
        # root #0 seat 0 -> 1: duplicates the existing (0, 206, 1) root
        # key and mismatches the root's own arms/features.
        A["seat"][0] = 1
    elif name == "real_neg2":
        # plant the padding sentinel inside a REAL arm of root #0
        A["arms"][0, 0, 0] = -2
    elif name == "pad_root":
        # make root #0 a genuinely padded root: n_arms 4 -> 3, slot 3
        # becomes annotator padding (-2 arms, 0 priors/q, NaN Q) with
        # priors/q renormalised over the 3 real arms so the record
        # stays S13-C consistent.
        A["n_arms"][0] = 3
        A["arms"][0, 3, :] = -2
        A["priors"][0, 3] = 0.0
        A["priors"][0, :3] /= A["priors"][0, :3].sum()
        A["Q"][0, 3] = np.nan
        A["Q_std"][0, 3] = np.nan
        A["q"][0, :3] = construct_target(A["Q"][0, :3], A["priors"][0, :3], 0.05)
        A["q"][0, 3] = 0.0
    else:
        raise SystemExit(f"unknown mutation {name!r}")


def main():
    name, dst = sys.argv[1], sys.argv[2]
    ptr = json.load(open(os.path.join(SRC, "current.json")))
    src_gen = os.path.join(SRC, ptr["dir"])
    old_man = json.load(open(os.path.join(src_gen, "manifest.json")))
    A = {k: np.load(os.path.join(src_gen, k + ".npy")) for k in FIELDS}
    mutate(name, A)
    staging = os.path.join(dst, ".staging")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(staging)
    for k in FIELDS:
        np.save(os.path.join(staging, k + ".npy"), A[k])
    shutil.copyfile(os.path.join(src_gen, "meta.json"),
                    os.path.join(staging, "meta.json"))
    finfo = {}
    for k in FIELDS:
        p = os.path.join(staging, k + ".npy")
        finfo[k] = {"file": k + ".npy", "bytes": os.path.getsize(p),
                    "rows": int(A[k].shape[0]), "sha256": _sha256_file(p)}
    core = {"format": old_man["format"], "total": old_man["total"],
            "producer": old_man["producer"],
            "meta_sha256": old_man["meta_sha256"], "fields": finfo}
    gen_id = "gen-" + hashlib.sha256(_canonical(core)).hexdigest()[:16]
    man = dict(core)
    man["generation"] = gen_id
    man["created"] = 0.0
    man["self_sha256"] = hashlib.sha256(_canonical(man)).hexdigest()
    with open(os.path.join(staging, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    gen_dir = os.path.join(dst, "generations", gen_id)
    os.makedirs(os.path.join(dst, "generations"))
    os.replace(staging, gen_dir)
    new_ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
               "manifest_sha256": _sha256_file(
                   os.path.join(gen_dir, "manifest.json")),
               "total": old_man["total"], "created": 0.0}
    with open(os.path.join(dst, "current.json"), "w") as fh:
        json.dump(new_ptr, fh)
    print(json.dumps({"doctored": name, "dataset": dst, "generation": gen_id}))


if __name__ == "__main__":
    main()
