#!/usr/bin/env python3
"""Part 2 (PREREG_SEARCH §2): build the HELD-OUT root set for the
value-head dimensional fit.

Caliber mirror of Stage 1 build_roots.py (DGX archive ppo_attrib),
changing ONLY the game slice: Stage 1 took ds_m3s2 games [0:32] by
filename order; this takes [32:64] of the same sorted list (asserted
in-script against the registered rootset32 manifest). Steps [200,299),
BOTH seats, featurized via the repo's own cnet2_build.build_game with
the dataset's recorded teacher tids [2,2] and per-side margins from
the ds meta scores -- identical conventions to Stage 1.

Runs on DGX. Read-only on the corpus and repo code.
"""
import hashlib
import json
import os
import sys

import numpy as np

CODE = "/home/arm/junyang/carbon/code"
sys.path.insert(0, CODE)
import cnet2_build as B  # noqa: E402

WORK = "/home/arm/junyang/katago_probe_scratch/part2"
DS_META = "/home/arm/junyang/carbon/data/ds_m3s2/meta.json"
REPLAYS = "/home/arm/junyang/carbon/data/m3s2_corpus/replays"
ROOTSET32 = "/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/rootset32.json"
OUT = os.path.join(WORK, "heldout.npz")
MANIFEST = os.path.join(WORK, "heldout_manifest.json")
DS_META_MD5 = "1d7d41febe1813467e1e83b312b8cae8"

FEAT = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
        "hist_cell", "hist_cargo", "hist_move", "uf", "mf", "legal",
        "y", "teacher")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert md5_file(DS_META) == DS_META_MD5, "ds meta md5 drift"
    assert md5_file(os.path.join(CODE, "cnet2_build.py")) == \
        "e005152499b2ba30a56f92c7096e2707", "cnet2_build md5 drift"
    meta = json.load(open(DS_META))
    games = sorted(meta["games"], key=lambda g: g["key"])
    rs = json.load(open(ROOTSET32))
    assert [g["key"] for g in games[:32]] == [g["key"] for g in rs], \
        "filename-order first 32 != registered rootset32"
    held = games[32:64]
    assert len(held) == 32
    manifest = [{"key": g["key"], "md5": md5_file(os.path.join(REPLAYS, g["key"])),
                 "names": g["names"], "gi": g["gi"],
                 "scores": g["scores"]} for g in held]
    json.dump(manifest, open(MANIFEST, "w"), indent=1)

    stack = {k: [] for k in FEAT}
    meta_rows = []
    counts = {}
    for gi, g in enumerate(held):
        path = os.path.join(REPLAYS, g["key"])
        scores = g["scores"]
        margins = [scores[0] - scores[1], scores[1] - scores[0]]
        tids = [2, 2]   # dataset of record's recorded teacher value
        for seat in (0, 1):
            out = os.path.join(WORK, f"hshard_{gi}_{seat}.npz")
            task = (path, [seat], gi, [tids[seat]], [margins[seat]], out)
            rgi, n, err = B.build_game(task)
            assert err is None, (g["key"], seat, err)
            z = np.load(out)
            step = z["step"]
            sel = (step >= 200) & (step < 299)
            counts[f"{g['key']}|seat{seat}"] = int(sel.sum())
            for k in FEAT:
                stack[k].append(z[k][sel])
            for s in step[sel]:
                meta_rows.append((g["key"], seat, int(s)))
            os.remove(out)

    arr = {k: np.concatenate(v) for k, v in stack.items()}
    arr["seat"] = np.array([r[1] for r in meta_rows], dtype=np.int8)
    arr["step_arr"] = np.array([r[2] for r in meta_rows], dtype=np.int32)
    arr["game_key"] = np.array([r[0] for r in meta_rows])
    np.savez(OUT, **arr)
    print("heldout rows:", len(arr["y"]))
    print("per (game,seat) counts min/max:",
          min(counts.values()), max(counts.values()))
    print("games with <99 window rows:",
          {k: v for k, v in counts.items() if v != 99})
    print("teacher ids present:", sorted(set(arr["teacher"].tolist())))
    json.dump(counts, open(os.path.join(WORK, "heldout_counts.json"), "w"),
              indent=1)
    print("heldout.npz md5:", md5_file(OUT))
    print("manifest md5:", md5_file(MANIFEST))


if __name__ == "__main__":
    main()
