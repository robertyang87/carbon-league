"""Regression fixture for the generation meta seal (cnet2_build).

A generation's manifest used to hash only the field .npy files, while
the split declaration (per-game val flags) and game counts live in the
generation's meta.json -- flipping a val flag in meta was undetectable
by load_generation(verify=True). The manifest now carries meta_sha256
and the certified loader verifies it. Generations published before the
seal (no meta_sha256 in the manifest) keep loading, for compatibility;
the Stage B split checklist requires the seal on new generations.

Run: python test_build_meta_seal.py  (from code/, or an acceptance tree)
"""
import hashlib
import json
import os
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "cnet2_build.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))
sys.path.insert(0, CODE)

import numpy as np
import cnet2_build as B

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def build_gen(root, sealed):
    gen_dir = os.path.join(root, "generations", "gen-fixture")
    os.makedirs(gen_dir)
    np.save(os.path.join(gen_dir, "gid.npy"), np.arange(6, dtype=np.int64))
    meta = {"games": [{"key": f"g{i}", "val": i == 0} for i in range(3)], "total": 6}
    with open(os.path.join(gen_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    man = {"generation": "gen-fixture", "total": 6,
           "fields": {"gid": {"file": "gid.npy", "rows": 6,
                              "sha256": sha256_file(os.path.join(gen_dir, "gid.npy"))}}}
    if sealed:
        man["meta_sha256"] = sha256_file(os.path.join(gen_dir, "meta.json"))
    with open(os.path.join(gen_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    with open(os.path.join(root, "current.json"), "w") as fh:
        json.dump({"dir": "generations/gen-fixture", "generation": "gen-fixture"}, fh)
    return gen_dir


def refuses(fn, needle):
    try:
        fn()
    except Exception as exc:
        return needle in str(exc)
    return False


def main():
    with tempfile.TemporaryDirectory() as td:
        gen_dir = build_gen(td, sealed=True)
        got_dir, man = B.load_generation(td, verify=True)
        check("sealed generation verifies", got_dir == gen_dir and "meta_sha256" in man)

        meta_path = os.path.join(gen_dir, "meta.json")
        meta = json.load(open(meta_path))
        meta["games"][1]["val"] = True  # the undetectable edit this seal exists for
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        check("val-flag flip in meta.json is refused",
              refuses(lambda: B.load_generation(td, verify=True), "meta hash mismatch"))

    with tempfile.TemporaryDirectory() as td:
        gen_dir = build_gen(td, sealed=True)
        arr = np.load(os.path.join(gen_dir, "gid.npy"))
        arr[0] = 99
        np.save(os.path.join(gen_dir, "gid.npy"), arr)
        check("field tamper still refused (pre-existing seal intact)",
              refuses(lambda: B.load_generation(td, verify=True), "hash mismatch"))

    with tempfile.TemporaryDirectory() as td:
        build_gen(td, sealed=False)
        got_dir, man = B.load_generation(td, verify=True)
        check("legacy unsealed generation still loads (compatibility)",
              man.get("meta_sha256") is None)

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
