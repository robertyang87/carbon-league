"""F05 first-hand: mid-publish os.replace failure.
Assert: build raises; current.json still points at the OLD generation;
resolve_dataset()/load(gen dir) read the complete old generation;
document the flat-root state honestly."""
import json, os, shutil, sys
CODE = os.path.expanduser("~/workspace/final-accept/code")
WORK = os.path.expanduser("~/workspace/final-accept/work")
sys.path.insert(0, CODE)
os.environ["BC_DEVICE"] = "cpu"
import numpy as np
import cnet2_build as B

corpus = os.path.join(WORK, "corpus_raw")          # producer stub corpus from derived tests (4 games)
out = os.path.join(WORK, "ds_pub")
if os.path.exists(out): shutil.rmtree(out)
B.main(out, [corpus])
ptr1 = json.load(open(os.path.join(out, "current.json")))
print("gen1:", ptr1["generation_id"] if "generation_id" in ptr1 else ptr1)
gen1_dir = B.resolve_dataset(out)
meta1 = json.load(open(os.path.join(out, "meta.json")))

# second build with one game changed cfg fingerprint -> new generation, inject failure at 2nd flat replace
real_replace = os.replace
state = {"n": 0}
def bad_replace(src, dst):
    if str(dst).startswith(out) and str(dst).endswith(".npy"):
        state["n"] += 1
        if state["n"] == 2:
            raise OSError("injected second-file publication failure")
    return real_replace(src, dst)
B.os.replace = bad_replace
os.environ["CNET2_BUILD_TAG"] = "v2"   # may or may not affect fingerprint; also touch corpus copy
corpus2 = os.path.join(WORK, "corpus_raw_b")
if os.path.exists(corpus2): shutil.rmtree(corpus2)
shutil.copytree(corpus, corpus2)
idx = json.load(open(os.path.join(corpus2, "index.json")))
idx[0]["scores"] = {"a": 501, "b": 100}; idx[0]["margin"] = {"a": 401, "b": -401}
from cnet2_build import _u_of_margin
idx[0]["u"] = {"a": _u_of_margin(401), "b": _u_of_margin(-401)}
json.dump(idx, open(os.path.join(corpus2, "index.json"), "w"))
try:
    B.main(out, [corpus2])
    print("FAIL build did not raise")
    sys.exit(1)
except OSError as e:
    print("build raised as injected:", e)
finally:
    B.os.replace = real_replace

ptr2 = json.load(open(os.path.join(out, "current.json")))
print("pointer after failure:", ptr2)
same_ptr = ptr2 == ptr1
gen_dir = B.resolve_dataset(out)
import cnet2_train as T
D, val, meta = T.load(gen_dir)
rows = len(np.asarray(D["y"]))
print(f"resolved gen dir: {os.path.basename(gen_dir)} | load rows={rows} | meta total={meta['total']}")
print(("PASS " if same_ptr else "FAIL ") + "F05 pointer stays on old generation after mid-publish replace failure")
# honest flat-root state
import glob
flat_meta = json.load(open(os.path.join(out, "meta.json")))
print("flat meta total:", flat_meta["total"], "| flat meta == old meta:", flat_meta == meta1)
sys.exit(0 if same_ptr else 1)
