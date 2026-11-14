"""exit_merge -- merge data/exit1/shards/*.npz (exit_gen) into a SEALED generation dataset for
cnet2_exit_train, mirroring cnet2_build's F05 generation publishing (D1 spec, 策略组）:
  OUT_DIR/generations/gen-<id>/   one complete product: one .npy per field, meta.json and
                                  manifest.json (per-file {file, bytes, sha256, rows}, producer
                                  identity = exit_gen/exit_merge code md5 + config hash, schema
                                  format, meta_sha256 seal of meta.json, manifest self_sha256),
                                  written in a staging dir and renamed into place;
  OUT_DIR/current.json            the pointer ({generation, dir, manifest_sha256, total}),
                                  switched LAST with a single os.replace.
The generation id is content-addressed (sha256 of the manifest core), so re-merging identical
inputs republishes the identical generation. cnet2_exit_train consumes sealed generations ONLY
via the pointer (verify-or-refuse); unsealed legacy flat dirs stay loadable by direct path, named.
Extra fields: searched, changed, pi, pi_slot, y_net, q.  Validation games: seed % 10 == 0.
  python3 exit_merge.py OUT_DIR [SHARD_DIR ...]"""
import gc
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnet2_build import FIELDS  # noqa: E402
import exit_gen  # noqa: E402
from exit_gen import EXTRA  # noqa: E402


EXCL = [x for x in os.environ.get("EXIT_EXCLUDE", "cnet2_v13,opp_1cz_v5,b160").split(",") if x]   # held-out test opponents
FORMAT = "cnet2-exit-generation/1"


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def main(out, dirs):
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.npz")))
    games, ns = [], []
    for fn in files:
        try:
            with np.load(fn) as D:
                n = int(D["y"].shape[0])
        except Exception:
            continue
        b = os.path.basename(fn)[:-4]
        seed = int(b.split("_")[0])
        if any(x in b for x in EXCL):
            continue
        games.append({"shard": fn, "key": b, "seed": seed, "opp": "_".join(b.split("_")[1:-1]), "val": seed % 10 == 0, "n": n})
        ns.append(n)
    total = int(sum(ns))
    fields = dict(FIELDS); fields.update(EXTRA)
    os.makedirs(os.path.join(out, "generations"), exist_ok=True)
    staging = os.path.join(out, f".staging-exit-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        mm = {k: np.lib.format.open_memmap(os.path.join(staging, k + ".npy"), mode="w+", dtype=dt, shape=(total,) + sh)
              for k, (dt, sh) in fields.items()}
        o = 0
        for gi, g in enumerate(games):
            with np.load(g["shard"]) as D:
                n = g["n"]
                for k in fields:
                    if k == "gid":
                        mm[k][o:o + n] = gi
                    elif k in D.files:
                        mm[k][o:o + n] = D[k]
                    else:
                        mm[k][o:o + n] = np.nan if k == "q" else 0
            g["gi"] = gi
            o += n
        for k in mm:
            mm[k].flush()
        ch = np.asarray(mm["changed"]); se = np.asarray(mm["searched"])
        n_searched, n_changed = int(se.sum()), int(ch.sum())
        mm = {}
        gc.collect()  # release the memmaps so the rename cannot race a writer
        with open(os.path.join(staging, "meta.json"), "w") as fh:
            json.dump({"games": games, "total": total}, fh)
        # ---- D1 seal: manifest over the finished product, self-sealed ----
        config = {"exclude": EXCL, "shard_dirs": [str(d) for d in dirs],
                  "val_rule": "seed % 10 == 0",
                  "fields": {k: [str(dt), [int(s) for s in sh]] for k, (dt, sh) in fields.items()}}
        producer = {"exit_gen_md5": _md5_file(os.path.abspath(exit_gen.__file__)),
                    "exit_merge_md5": _md5_file(os.path.abspath(__file__)),
                    "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
                    "config": config}
        finfo = {}
        for k in fields:
            p = os.path.join(staging, k + ".npy")
            arr = np.load(p, mmap_mode="r")
            finfo[k] = {"file": k + ".npy", "bytes": os.path.getsize(p),
                        "rows": int(arr.shape[0]), "sha256": _sha256_file(p)}
            del arr
        core = {"format": FORMAT, "total": total, "producer": producer,
                "meta_sha256": _sha256_file(os.path.join(staging, "meta.json")),
                "fields": finfo}
        gen_id = "gen-" + hashlib.sha256(_canonical(core)).hexdigest()[:16]
        man = dict(core)
        man["generation"] = gen_id
        man["created"] = time.time()
        man["self_sha256"] = hashlib.sha256(_canonical(man)).hexdigest()
        with open(os.path.join(staging, "manifest.json"), "w") as fh:
            json.dump(man, fh)
        gen_dir = os.path.join(out, "generations", gen_id)
        landed = False
        if os.path.exists(os.path.join(gen_dir, "manifest.json")):
            try:
                landed = json.load(open(os.path.join(gen_dir, "manifest.json"))).get("self_sha256") == man["self_sha256"]
            except Exception:
                landed = False
        if not landed:
            shutil.rmtree(gen_dir, ignore_errors=True)
            os.replace(staging, gen_dir)
        # The pointer switch is the single atomic commit of the publish.
        ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
               "manifest_sha256": _sha256_file(os.path.join(gen_dir, "manifest.json")),
               "total": total, "created": time.time()}
        ptr_tmp = os.path.join(out, f".current-{os.getpid()}.tmp")
        with open(ptr_tmp, "w") as fh:
            json.dump(ptr, fh)
        os.replace(ptr_tmp, os.path.join(out, "current.json"))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps({"games": len(games), "samples": total, "val_games": sum(g["val"] for g in games),
                      "searched": n_searched, "changed": n_changed,
                      "generation": gen_id, "sealed": True, "pointer": "current.json"}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or [os.path.join(HERE, "..", "data", "exit1", d) for d in ("shards", "shards_mfxn", "shards_mfh")])
