"""Residual 3 re-verification: torn flat mirror after mid-publish failure.
train load() must read ONLY the old complete generation; flat meta stamp;
BUILD_FAILED marker; legacy pointer-less dataset readable + torn legacy refused."""
import json, os, shutil, sys
CODE = "/home/hatch/workspace/fix4-residual/tree"
WORK = os.path.expanduser("~/workspace/final-accept/work")
sys.path.insert(0, CODE)
os.environ["BC_DEVICE"] = "cpu"
import numpy as np
import cnet2_build as B
import cnet2_train as T

fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f" | {extra}" if extra else ""))
    if not cond: fails.append(name)

corpus = os.path.join(WORK, "corpus_raw")
out = "/home/hatch/workspace/fix4-residual/ds_r3"
if os.path.exists(out): shutil.rmtree(out)
B.main(out, [corpus])
ptr1 = json.load(open(os.path.join(out, "current.json")))
gen1_dir = B.resolve_dataset(out)
D1, _, meta1 = T.load(out)
rows1 = len(np.asarray(D1["y"]))
y1 = np.asarray(D1["y"]).copy()
flat_meta = json.load(open(os.path.join(out, "meta.json")))
check("flat meta stamped with generation", flat_meta.get("flat_generation") == ptr1["generation"],
      str(flat_meta.get("flat_generation")))
check("flat meta stamped with total", flat_meta.get("flat_total") == rows1, str(flat_meta.get("flat_total")))
check("gen meta pristine (no stamp)", "flat_generation" not in json.load(open(os.path.join(gen1_dir, "meta.json"))))
check("load resolved gen1 rows", rows1 == meta1["total"], f"rows={rows1}")

# snapshot legacy (pointer-less) copy of gen1 flat view
legacy = "/home/hatch/workspace/fix4-residual/ds_legacy"
if os.path.exists(legacy): shutil.rmtree(legacy)
os.makedirs(legacy)
for f in os.listdir(out):
    p = os.path.join(out, f)
    if os.path.isfile(p) and f != "current.json":
        shutil.copyfile(p, os.path.join(legacy, f))
Dl, _, _ = T.load(legacy)
check("legacy pointer-less dataset loads", len(np.asarray(Dl["y"])) == rows1)

# torn legacy mirror: truncate one field -> must be refused by name
arr = np.load(os.path.join(legacy, "maps.npy"))
np.save(os.path.join(legacy, "maps.npy"), arr[: max(1, rows1 - 1)])
try:
    T.load(legacy)
    check("torn legacy mirror refused", False, "load silently succeeded")
except SystemExit as e:
    check("torn legacy mirror refused", "torn flat mirror" in str(e), str(e)[:110])

# second build, inject failure at 2nd flat .npy replace
corpus2 = "/home/hatch/workspace/fix4-residual/corpus_b"
if os.path.exists(corpus2): shutil.rmtree(corpus2)
shutil.copytree(corpus, corpus2)
idx = json.load(open(os.path.join(corpus2, "index.json")))
idx[0]["scores"] = {"a": 501, "b": 100}; idx[0]["margin"] = {"a": 401, "b": -401}
from cnet2_build import _u_of_margin
idx[0]["u"] = {"a": _u_of_margin(401), "b": _u_of_margin(-401)}
json.dump(idx, open(os.path.join(corpus2, "index.json"), "w"))

real_replace = os.replace
state = {"n": 0}
def bad_replace(src, dst):
    if str(dst).startswith(out) and str(dst).endswith(".npy"):
        state["n"] += 1
        if state["n"] == 2:
            raise OSError("injected second-file publication failure")
    return real_replace(src, dst)
B.os.replace = bad_replace
try:
    B.main(out, [corpus2])
    check("build raises on injected publish failure", False)
except OSError:
    check("build raises on injected publish failure", True)
finally:
    B.os.replace = real_replace

bf = os.path.join(out, "BUILD_FAILED.json")
check("BUILD_FAILED marker written", os.path.exists(bf) and json.load(open(bf)).get("failed_stage") == "publish",
      open(bf).read()[:120] if os.path.exists(bf) else "missing")
ptr2 = json.load(open(os.path.join(out, "current.json")))
check("pointer still certifies old generation", ptr2 == ptr1)
D2, _, meta2 = T.load(out)
rows2 = len(np.asarray(D2["y"]))
check("load after failure reads old gen completely", rows2 == rows1 and meta2["total"] == meta1["total"],
      f"rows={rows2} vs {rows1}")
check("load after failure: y identical to gen1 (no mixed rows)",
      np.array_equal(np.asarray(D2["y"]), y1))
allrows = {k: int(np.asarray(D2[k]).shape[0]) for k in B.FIELDS}
check("load after failure: all field rows consistent", len(set(allrows.values())) == 1, str(allrows))

# clean rebuild -> new generation readable, stamp updated
# (subset corpus: 3 of 4 games -> different gen_key, hence a new generation;
#  a scores-only change intentionally keeps the same generation id -- the
#  builder's content identity covers cfg + game keys/counts, see report note)
corpus3 = "/home/hatch/workspace/fix4-residual/corpus_sub"
if os.path.exists(corpus3): shutil.rmtree(corpus3)
shutil.copytree(corpus, corpus3)
_idx = json.load(open(os.path.join(corpus3, "index.json")))[:3]
json.dump(_idx, open(os.path.join(corpus3, "index.json"), "w"))
B.main(out, [corpus3])
ptr3 = json.load(open(os.path.join(out, "current.json")))
D3, _, meta3 = T.load(out)
fm3 = json.load(open(os.path.join(out, "meta.json")))
check("clean rebuild switches pointer", ptr3["generation"] != ptr1["generation"])
check("clean rebuild load rows == new total", len(np.asarray(D3["y"])) == meta3["total"] == ptr3["total"])
check("clean rebuild flat stamp updated", fm3.get("flat_generation") == ptr3["generation"] and fm3.get("flat_total") == ptr3["total"])
check("BUILD_FAILED cleared after success", not os.path.exists(bf))

print("RESULT:", "ALL PASS" if not fails else f"FAILURES {fails}")
sys.exit(1 if fails else 0)
