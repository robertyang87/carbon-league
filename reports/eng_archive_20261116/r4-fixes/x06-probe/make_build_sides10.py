"""X06 probe Case B: derive a meta-consistent sides=[1,0] build generation
from the accepted fixture generation gen-75de8ffbdd421e42 by swapping, per
game, the two 299-row side blocks in EVERY field file and recording
sides=[1,0] in the meta games entries, then resealing (per-field sha256 +
meta_sha256). This is the generation a producer running sides=[1,0] would
emit (build_game concatenates blocks in `sides` order); join's seat
attribution must follow the meta `sides`, not a hard-coded [0,1]."""
import hashlib, json, os, shutil, time
import numpy as np

SRC = "/home/hatch/workspace/k0-round2/join/fixture/build_out/generations/gen-75de8ffbdd421e42"
OUT = "/home/hatch/workspace/r4-fixes/x06-probe/build_sides10_out"
GEN = "gen-x06sides10"
BLOCK = 299

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

man = json.load(open(os.path.join(SRC, "manifest.json")))
meta = json.load(open(os.path.join(SRC, "meta.json")))
total = int(man["total"])
per_game = total // len(meta["games"])
assert per_game == 2 * BLOCK, per_game

gen_dir = os.path.join(OUT, "generations", GEN)
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(gen_dir)
for name, info in man["fields"].items():
    a = np.load(os.path.join(SRC, info["file"]))
    out = a.copy()
    for gi in range(len(meta["games"])):
        o = gi * per_game
        out[o:o + BLOCK] = a[o + BLOCK:o + per_game]
        out[o + BLOCK:o + per_game] = a[o:o + BLOCK]
    np.save(os.path.join(gen_dir, info["file"]), out)
for g in meta["games"]:
    assert g["sides"] == [0, 1], g["sides"]
    g["sides"] = [1, 0]
meta["fixture_note"] = ("X06 probe Case B (doctored from gen-75de8ffbdd421e42): "
                        "per-game side blocks swapped in all field files and "
                        "meta sides recorded as [1, 0]; resealed.")
with open(os.path.join(gen_dir, "meta.json"), "w") as fh:
    json.dump(meta, fh)
man2 = dict(man)
man2["generation"] = GEN
man2["created"] = time.time()
man2["meta_sha256"] = sha(os.path.join(gen_dir, "meta.json"))
man2["fields"] = {k: {"file": v["file"], "rows": v["rows"],
                      "sha256": sha(os.path.join(gen_dir, v["file"]))}
                  for k, v in man["fields"].items()}
with open(os.path.join(gen_dir, "manifest.json"), "w") as fh:
    json.dump(man2, fh)
ptr = {"generation": GEN, "dir": f"generations/{GEN}",
       "cfg_fp": man.get("cfg_fp"), "total": total, "created": time.time()}
with open(os.path.join(OUT, "current.json"), "w") as fh:
    json.dump(ptr, fh)
print(json.dumps({"caseB_generation": GEN, "games": [(g["key"], g["sides"]) for g in meta["games"]]}))
