"""Copy a sealed annotation dataset root and re-seal it after a
caller mutation of meta.json / manifest producer -- for the X02
negative fixtures (missing record, disagreeing records, disagreeing
hashes). Mirrors make_fixture2's sealing composition exactly."""
import hashlib
import json
import os
import shutil
import sys


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def reseal(src_root, dst_root, mutate):
    shutil.rmtree(dst_root, ignore_errors=True)
    shutil.copytree(src_root, dst_root)
    gen = json.load(open(os.path.join(dst_root, "current.json")))["generation"]
    gen_dir = os.path.join(dst_root, "generations", gen)
    man = json.load(open(os.path.join(gen_dir, "manifest.json")))
    meta = json.load(open(os.path.join(gen_dir, "meta.json")))
    mutate(man, meta)
    with open(os.path.join(gen_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    core = {"format": man["format"], "total": man["total"],
            "producer": man["producer"],
            "meta_sha256": _sha256_file(os.path.join(gen_dir, "meta.json")),
            "fields": man["fields"]}
    gen_id = "gen-" + hashlib.sha256(_canonical(core)).hexdigest()[:16]
    new_man = dict(core)
    new_man["generation"] = gen_id
    new_man["created"] = 0.0
    new_man["self_sha256"] = hashlib.sha256(_canonical(
        {k: v for k, v in new_man.items() if k != "self_sha256"})).hexdigest()
    new_dir = os.path.join(dst_root, "generations", gen_id)
    os.replace(gen_dir, new_dir)
    with open(os.path.join(new_dir, "manifest.json"), "w") as fh:
        json.dump(new_man, fh)
    ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
           "manifest_sha256": _sha256_file(os.path.join(new_dir, "manifest.json")),
           "total": man["total"], "created": 0.0}
    with open(os.path.join(dst_root, "current.json"), "w") as fh:
        json.dump(ptr, fh)
    return gen_id


if __name__ == "__main__":
    src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "missing":
        def mut(man, meta):
            meta.pop("teacher", None)
    elif mode == "disagree":
        def mut(man, meta):
            man["producer"]["config"]["train_temp"] = 3.0
    elif mode == "hashdis":
        def mut(man, meta):
            meta["teacher"]["config_sha256"] = "0" * 64
    else:
        raise SystemExit(f"unknown mode {mode}")
    print(reseal(src, dst, mut))
