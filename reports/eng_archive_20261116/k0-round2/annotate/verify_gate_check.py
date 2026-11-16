"""Acceptance #3 (product side): a generation whose seals are all honest
but whose records break the S13-C gates must be refused by --verify with
the RECORD violation named (not a seal error). Builds such a generation
by hand (same sealing formulas as k0_annotate.publish) with one root's
step moved outside the registered window, and one with pi deleted."""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/run")
import numpy as np  # noqa: E402
import k0_annotate as A  # noqa: E402


def build(root, mode):
    gen = os.path.join(root, "generations", "gen-doctored")
    os.makedirs(gen, exist_ok=True)
    n, K = 2, 4
    arrs = {"gid": np.array([0, 0], np.int32), "step": np.array([206, 207], np.int32),
            "seat": np.array([0, 1], np.int8),
            "arms": np.zeros((n, K, 12), np.int16), "n_arms": np.array([2, 2], np.int16),
            "priors": np.array([[.5, .5, 0, 0], [.5, .5, 0, 0]]),
            "q": np.array([[.5, .5, 0, 0], [.25, .75, 0, 0]]),
            "Q": np.array([[.5, .6, np.nan, np.nan], [.4, .7, np.nan, np.nan]]),
            "Q_std": np.zeros((n, K)),
            "pi0": np.array([1.0, 1.0]), "pi1": np.array([1.0, 1.0]),
            "pi": np.array([1.0, 1.0]),
            "belief": np.array([0, 0], np.int8), "n_unknown": np.array([3, 3], np.int32)}
    if mode == "window":
        arrs["step"] = np.array([150, 207], np.int32)  # outside [200, 299)
    if mode == "missing-pi":
        del arrs["pi"]
    finfo = {}
    for k, v in arrs.items():
        p = os.path.join(gen, k + ".npy")
        np.save(p, v)
        finfo[k] = {"file": k + ".npy", "bytes": os.path.getsize(p),
                    "rows": int(v.shape[0]), "sha256": A._sha256_file(p)}
    cfg = {"K": 4, "M": 2, "H": 1, "teacher_style": 2, "belief": "sample",
           "train_temp": 1.0, "behavior_temp": 1.0, "seed_teacher": 7000,
           "guard_cfg": {}, "window": [200, 299],
           "scene_contract": "prefix-feat-guard-snapshot-v1"}
    meta = {"format": A.FORMAT, "total": n, "games": [],
            "teacher": {"version": "v3", "belief": "sample", "weights": "x",
                        "weights_md5": "w" * 32, "config": cfg,
                        "config_sha256": "c" * 64},
            "sampling": {}, "record_schema": {}}
    meta["teacher"]["config_sha256"] = A._sha256_bytes(A._canonical(cfg))
    with open(os.path.join(gen, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    producer = {"k0_annotate_md5": "x", "config_sha256": meta["teacher"]["config_sha256"],
                "config": cfg, "weights_md5": "w" * 32}
    core = {"format": A.FORMAT, "total": n, "producer": producer,
            "meta_sha256": A._sha256_file(os.path.join(gen, "meta.json")),
            "fields": finfo}
    man = dict(core)
    man["generation"] = "gen-doctored"
    man["created"] = time.time()
    man["self_sha256"] = A._sha256_bytes(A._canonical(man))
    with open(os.path.join(gen, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    ptr = {"generation": "gen-doctored", "dir": "generations/gen-doctored",
           "manifest_sha256": A._sha256_file(os.path.join(gen, "manifest.json")),
           "total": n, "created": time.time()}
    with open(os.path.join(root, "current.json"), "w") as fh:
        json.dump(ptr, fh)
    return root


if __name__ == "__main__":
    mode, root = sys.argv[1], sys.argv[2]
    build(root, mode)
    try:
        s = A.resolve_out_arg(root)
        print("NO DIE -- BAD:", s)
        sys.exit(1)
    except SystemExit as e:
        print(f"{mode}: verify DIED as required -- {str(e)[:160]}")
