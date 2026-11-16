"""Build-side negative fixture for k0_join: drop exactly ONE feature
row from the sealed build fixture generation -- the row annotation
root #6 needs -- and reseal the build manifest's field table/total
to match. meta.json is carried over byte-for-byte (its sha stays the
manifest's meta_sha256). The doctored generation passes every check
k0_join.load_build_source performs (pointer/manifest generation
match, per-field sha256, meta seal); the join must then die at the
missing feature row, naming the root."""
import json
import os
import shutil
import sys

import numpy as np

SRC = "/home/hatch/workspace/k0-round2/join/fixture/build_out"
ANNOT = "/home/hatch/workspace/k0-round2/annotate/fixture/out1"
DST = sys.argv[1]
DROP_ROOT = 6
FEATURE_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
                  "tok_cargo", "hist_cell", "hist_cargo", "hist_move",
                  "uf", "mf", "legal", "carbon", "teacher", "gid", "step")


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- locate the build row for annotation root #DROP_ROOT ------------
aptr = json.load(open(os.path.join(ANNOT, "current.json")))
agd = os.path.join(ANNOT, aptr["dir"])
ameta = json.load(open(os.path.join(agd, "meta.json")))
A = {k: np.load(os.path.join(agd, k + ".npy"))
     for k in ("gid", "step", "seat")}
r = DROP_ROOT
tgid, tstep, tseat = int(A["gid"][r]), int(A["step"][r]), int(A["seat"][r])
game = [g for g in ameta["games"] if int(g["gid"]) == tgid][0]["game"]

bptr = json.load(open(os.path.join(SRC, "current.json")))
bgd = os.path.join(SRC, bptr["dir"])
bman = json.load(open(os.path.join(bgd, "manifest.json")))
bmeta = json.load(open(os.path.join(bgd, "meta.json")))
bg = [g for g in bmeta["games"]
      if g["key"].rsplit(".json", 1)[0].rsplit(".gz", 1)[0] == game
      or g["key"].startswith(game)][0]
feats = {k: np.load(os.path.join(bgd, bman["fields"][k]["file"]))
         for k in FEATURE_FIELDS}
gid = np.asarray(feats["gid"]).astype(np.int64)
step = np.asarray(feats["step"]).astype(np.int64)
rows = np.nonzero(gid == int(bg["gi"]))[0]
st = step[rows]
bounds = [0]
for i in range(1, len(rows)):
    if st[i] <= st[i - 1]:
        bounds.append(i)
bounds.append(len(rows))
seat_pos = list(bg["sides"]).index(tseat)
seg = rows[bounds[seat_pos]:bounds[seat_pos + 1]]
hit = [int(x) for x in seg if int(step[x]) == tstep]
assert len(hit) == 1, f"expected exactly one build row, got {hit}"
drop = hit[0]
print(f"target root #{r}: annot gid {tgid} game {game!r} step {tstep} "
      f"seat {tseat} -> build row {drop} (gi {bg['gi']})")

# --- write the doctored generation -----------------------------------
shutil.rmtree(DST, ignore_errors=True)
gen_dir = os.path.join(DST, "generations", bptr["generation"])
os.makedirs(gen_dir)
keep = np.ones(int(bman["total"]), dtype=bool)
keep[drop] = False
new_fields = {}
for name, info in bman["fields"].items():
    src_p = os.path.join(bgd, info["file"])
    dst_p = os.path.join(gen_dir, info["file"])
    if name in FEATURE_FIELDS:
        arr = np.asarray(feats[name])[keep]
        np.save(dst_p, arr)
        new_fields[name] = {"file": info["file"],
                            "bytes": os.path.getsize(dst_p),
                            "rows": int(arr.shape[0]),
                            "sha256": sha256_file(dst_p)}
    else:
        shutil.copyfile(src_p, dst_p)
        new_fields[name] = dict(info)
shutil.copyfile(os.path.join(bgd, "meta.json"),
                os.path.join(gen_dir, "meta.json"))
man = dict(bman)
man["fields"] = new_fields
man["total"] = int(bman["total"]) - 1
with open(os.path.join(gen_dir, "manifest.json"), "w") as fh:
    json.dump(man, fh, indent=1, sort_keys=True)
ptr = dict(bptr)
ptr["manifest_sha256"] = sha256_file(os.path.join(gen_dir, "manifest.json"))
ptr["total"] = man["total"]
with open(os.path.join(DST, "current.json"), "w") as fh:
    json.dump(ptr, fh)
print(json.dumps({"doctored_build": DST, "generation": bptr["generation"],
                  "dropped_row": drop, "new_total": man["total"]}))
