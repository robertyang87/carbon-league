"""X04 gap demonstration against the CURRENT (X07-fixed, pre-binding)
driver: weight identity and consumption are two separate reads of
the path. Step 1 records md5_of(path) -- exactly what run_duel puts
in the header/rows. Step 2 replaces the file at the path with
DIFFERENT weights (the perturbed student fixture). Step 3 loads via
_load_policy(path) -- exactly what a worker consumes. The consumed
tensors are the replacement's, while the recorded identity is the
original's: record A, measure B. Run from the stage dir."""
import hashlib
import shutil
import sys

import numpy as np

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

CANON = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
OTHER = "../fixtures/student_perturbed.npz"
SWAP = "../fixtures/swap_x04.npz"

shutil.copy(CANON, SWAP)
recorded = D.md5_of(SWAP)
print(f"step1 recorded md5 (run_duel header value): {recorded}")

shutil.copy(OTHER, SWAP)
print(f"step2 file at path replaced; md5_of(path) now: {D.md5_of(SWAP)}")

cd_ = dict(np.load(CANON))
od = dict(np.load(OTHER))
diff = [k for k in cd_ if cd_[k].tobytes() != od[k].tobytes()]
print(f"tensors differing between the two files: {len(diff)} "
      f"(e.g. {diff[:3]})")
key = diff[0]
pol = D._load_policy(SWAP)
consumed = pol.w[key].tobytes()
print(f"step3 consumed {key} == recorded file's: "
      f"{consumed == cd_[key].tobytes()}")
print(f"step3 consumed {key} == replacement's:  "
      f"{consumed == od[key].tobytes()}")
assert consumed == od[key].tobytes() and consumed != cd_[key].tobytes()
print("RESULT: GAP CONFIRMED -- identity (md5) and consumption "
      "(loaded tensors) come from two reads; a swap between them "
      "records A while measuring B")
