"""ds_subset SRC OUT NTRAIN NVAL [SEED] -- random row subset of a cnet2 dataset (all FIELDS + w + uw), gids and
meta kept as they are (the val split stays the same games)."""
import json, os, shutil, sys
import numpy as np
from cnet2_build import FIELDS
src, out, ntr, nva = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rs = np.random.RandomState(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
meta = json.load(open(os.path.join(src, "meta.json"))); gid = np.load(os.path.join(src, "gid.npy"), mmap_mode="r")
isv = np.array([g["val"] for g in meta["games"]])[np.asarray(gid)]
tr = np.nonzero(~isv)[0]; va = np.nonzero(isv)[0]
sel = np.sort(np.concatenate([rs.choice(tr, min(ntr, len(tr)), replace=False), rs.choice(va, min(nva, len(va)), replace=False)]))
os.makedirs(out, exist_ok=True)
for k in list(FIELDS) + ["w", "uw"]:
    f = os.path.join(src, k + ".npy")
    if os.path.exists(f):
        np.save(os.path.join(out, k + ".npy"), np.asarray(np.load(f, mmap_mode="r")[sel]))
shutil.copy(os.path.join(src, "meta.json"), os.path.join(out, "meta.json"))
print(json.dumps({"rows": int(len(sel)), "train": int(min(ntr, len(tr))), "val": int(min(nva, len(va)))}))
