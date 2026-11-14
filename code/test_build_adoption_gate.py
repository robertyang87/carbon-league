"""Regression fixture for the generation adoption gate (cnet2_build).

Build used to reuse an existing same-id generation when every file its
manifest names merely EXISTED (_gen_complete): a directory whose bytes
differ from its own manifest was adopted whole -- pointer switched to it
and the flat mirror refreshed from it -- and an incomplete one was
rmtree'd. Pinned here:
- a verified generation is adoptable ("verified");
- same id, different field bytes / different meta bytes / foreign
  generation name / missing meta seal -> ValueError naming the file;
- an incomplete directory is reported "incomplete" and quarantine moves
  it aside with its contents (sentinel) intact, outside the gen-* space;
- every decision lands in generations/ADOPTION_LEDGER.jsonl;
- main() no longer carries the existence-only check or an rmtree of a
  generation directory.

Run: python test_build_adoption_gate.py  (from code/, or an acceptance tree)
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
GID = "gen-fixture"


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make_gen(root, sealed=True, name=GID):
    gen_root = os.path.join(root, "generations")
    gen_dir = os.path.join(gen_root, GID)
    os.makedirs(gen_dir)
    np.save(os.path.join(gen_dir, "gid.npy"), np.arange(6, dtype=np.int64))
    np.save(os.path.join(gen_dir, "y.npy"), np.zeros(6, dtype=np.int8))
    with open(os.path.join(gen_dir, "meta.json"), "w") as fh:
        json.dump({"games": [{"key": "g0", "val": True}], "total": 6}, fh)
    man = {"generation": name, "total": 6,
           "fields": {k: {"file": k + ".npy", "rows": 6, "sha256": sha(os.path.join(gen_dir, k + ".npy"))}
                      for k in ("gid", "y")}}
    if sealed:
        man["meta_sha256"] = sha(os.path.join(gen_dir, "meta.json"))
    with open(os.path.join(gen_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    return gen_root, gen_dir


def refuses(fn, needle):
    try:
        fn()
    except ValueError as exc:
        return needle in str(exc)
    return False


def main():
    with tempfile.TemporaryDirectory() as td:
        check("absent directory", B._inspect_generation(os.path.join(td, "nope"), GID) == ("absent", None))
        _root, gen_dir = make_gen(td)
        state, man = B._inspect_generation(gen_dir, GID)
        check("intact sealed generation verifies", state == "verified" and man["generation"] == GID)

    with tempfile.TemporaryDirectory() as td:       # the defect: same id, different bytes
        _root, gen_dir = make_gen(td)
        arr = np.load(os.path.join(gen_dir, "y.npy")); arr[3] = 1
        np.save(os.path.join(gen_dir, "y.npy"), arr)
        check("same-id generation with different field bytes is refused by name",
              refuses(lambda: B._inspect_generation(gen_dir, GID), "y.npy"))
        check("refused generation is left untouched", os.path.exists(os.path.join(gen_dir, "y.npy")))

    with tempfile.TemporaryDirectory() as td:
        _root, gen_dir = make_gen(td)
        with open(os.path.join(gen_dir, "meta.json"), "w") as fh:
            json.dump({"games": [{"key": "g0", "val": False}], "total": 6}, fh)
        check("meta.json edit is refused (seal)",
              refuses(lambda: B._inspect_generation(gen_dir, GID), "meta_sha256"))

    with tempfile.TemporaryDirectory() as td:
        _root, gen_dir = make_gen(td, sealed=False)
        check("unsealed (pre-seal) generation is not adoptable",
              refuses(lambda: B._inspect_generation(gen_dir, GID), "no meta_sha256 seal"))

    with tempfile.TemporaryDirectory() as td:
        _root, gen_dir = make_gen(td, name="gen-other")
        check("manifest naming another generation is refused",
              refuses(lambda: B._inspect_generation(gen_dir, GID), "gen-other"))

    for label, breaker in (("field file missing", lambda d: os.remove(os.path.join(d, "y.npy"))),
                           ("meta.json missing", lambda d: os.remove(os.path.join(d, "meta.json"))),
                           ("manifest missing", lambda d: os.remove(os.path.join(d, "manifest.json"))),
                           ("manifest garbage", lambda d: open(os.path.join(d, "manifest.json"), "w").write("{"))):
        with tempfile.TemporaryDirectory() as td:
            gen_root, gen_dir = make_gen(td)
            breaker(gen_dir)
            with open(os.path.join(gen_dir, "SENTINEL"), "w") as fh:
                fh.write("keep me")
            state, why = B._inspect_generation(gen_dir, GID)
            check(f"{label}: incomplete, not an error", state == "incomplete" and isinstance(why, str))
            dst = B._quarantine_generation(gen_root, gen_dir, why)
            check(f"{label}: quarantined, sentinel survives",
                  not os.path.exists(gen_dir) and open(os.path.join(dst, "SENTINEL")).read() == "keep me")
            check(f"{label}: quarantine is outside the gen-* space",
                  not os.path.basename(dst).startswith("gen-") and not B._is_generation_path(
                      os.path.join(dst, "y.npy")))
            rows = [json.loads(l) for l in open(os.path.join(gen_root, "ADOPTION_LEDGER.jsonl"))]
            check(f"{label}: ledger row written",
                  len(rows) == 1 and rows[0]["event"] == "quarantined" and rows[0]["generation"] == GID
                  and rows[0]["to"] == os.path.basename(dst))

    with tempfile.TemporaryDirectory() as td:       # two quarantines of the same id never collide
        gen_root, gen_dir = make_gen(td)
        os.remove(os.path.join(gen_dir, "y.npy"))
        d1 = B._quarantine_generation(gen_root, gen_dir, "x")
        os.makedirs(gen_dir)
        d2 = B._quarantine_generation(gen_root, gen_dir, "x")
        check("repeated quarantine keeps both", d1 != d2 and os.path.isdir(d1) and os.path.isdir(d2))

    src = open(os.path.join(CODE, "cnet2_build.py")).read()
    check("existence-only check is gone", "_gen_complete" not in src.split('"""', 2)[2].replace(
        "(_gen_complete)", ""))
    check("build never rmtree's a generation directory", "rmtree(gen_dir" not in src)

    print(f"{sum(RES)}/{len(RES)} passed", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
