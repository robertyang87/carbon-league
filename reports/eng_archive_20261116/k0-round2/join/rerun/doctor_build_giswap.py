"""Cross-generation gid trap test for k0_join (r4 item 3).

Copies the sealed build fixture generation with the two games' gi
values SWAPPED consistently (gid.npy values 0<->1 and the meta games
entries' gi fields exchanged), resealing the field table entry and
meta hash it affects. Game names/keys, row order and all other bytes
are unchanged. If k0_join paired generations by bare gid equality it
would now attach game 0's roots to game 1's rows; if it pairs by the
games-table NAME mapping (as designed), the joined feature/bundle
bytes must come out identical to the unswapped run."""
import hashlib
import json
import os
import shutil
import sys

import numpy as np

SRC = "/home/hatch/workspace/k0-round2/join/fixture/build_out"
DST = sys.argv[1]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ptr = json.load(open(os.path.join(SRC, "current.json")))
sgd = os.path.join(SRC, ptr["dir"])
man = json.load(open(os.path.join(sgd, "manifest.json")))
meta = json.load(open(os.path.join(sgd, "meta.json")))

shutil.rmtree(DST, ignore_errors=True)
gen_dir = os.path.join(DST, "generations", ptr["generation"])
os.makedirs(gen_dir)
for name, info in man["fields"].items():
    src_p = os.path.join(sgd, info["file"])
    dst_p = os.path.join(gen_dir, info["file"])
    if name == "gid":
        a = np.load(src_p)
        assert set(np.unique(a).tolist()) == {0, 1}
        np.save(dst_p, (1 - a).astype(a.dtype))
    else:
        shutil.copyfile(src_p, dst_p)

games = meta["games"]
assert len(games) == 2 and [g["gi"] for g in games] == [0, 1]
games[0]["gi"], games[1]["gi"] = 1, 0
with open(os.path.join(gen_dir, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1, sort_keys=True)

new_fields = {}
for name, info in man["fields"].items():
    p = os.path.join(gen_dir, info["file"])
    e = dict(info)
    e["sha256"] = sha256_file(p)
    e["bytes"] = os.path.getsize(p)
    new_fields[name] = e
man["fields"] = new_fields
man["meta_sha256"] = sha256_file(os.path.join(gen_dir, "meta.json"))
with open(os.path.join(gen_dir, "manifest.json"), "w") as fh:
    json.dump(man, fh, indent=1, sort_keys=True)
ptr2 = dict(ptr)
ptr2["manifest_sha256"] = sha256_file(os.path.join(gen_dir, "manifest.json"))
with open(os.path.join(DST, "current.json"), "w") as fh:
    json.dump(ptr2, fh)
print(json.dumps({"giswapped_build": DST,
                  "games": [(g["key"], g["gi"]) for g in games]}))
