#!/usr/bin/env python3
"""Stage 1 deliverable C, part 1: build the KL root set on L20.

Root set (PREREG): the run's training dataset of record = ds_m3s2
(DGX /home/arm/junyang/carbon/data/ds_m3s2, meta.json md5
1d7d41febe1813467e1e83b312b8cae8; source corpus
/home/arm/junyang/carbon/data/m3s2_corpus/replays). First 32 games by
filename order; steps [200,299); BOTH seats featurized from the replay
files via the repo's own path (cnet2_build.build_game, one call per
seat so seat attribution is by construction).

Runs on L20 inside /root/ppo_attrib (independent work dir; the
training directory is never written). Imports repo code read-only
from /root/carbon/code.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root/carbon/code")
import cnet2_build as B  # noqa: E402

WORK = "/root/ppo_attrib"
ROOTS = os.path.join(WORK, "roots")
OUT = os.path.join(WORK, "roots.npz")
MANIFEST = os.path.join(WORK, "rootset32.json")

FEAT = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
        "hist_cell", "hist_cargo", "hist_move", "uf", "mf", "legal",
        "y", "teacher")

games = json.load(open(MANIFEST))
stack = {k: [] for k in FEAT}
meta_rows = []  # (game_key, seat, step)
counts = {}
for gi, g in enumerate(games):
    path = os.path.join(ROOTS, g["key"])
    scores = g["scores"]
    margins = [scores[0] - scores[1], scores[1] - scores[0]]
    # teacher field: the dataset of record's own rows (ds_m3s2 flat
    # product) carry teacher=2 for every row (verified by field-digest
    # cross-check against DGX). B.teacher_id() name lookup falls back
    # to TEACHERS.index("other")=6 for these ladder display names and
    # must NOT be used here; tids are the dataset's recorded value.
    tids = [2, 2]
    for seat in (0, 1):
        out = os.path.join(WORK, f"shard_{gi}_{seat}.npz")
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
print("roots rows:", len(arr["y"]))
print("per (game,seat) counts min/max:",
      min(counts.values()), max(counts.values()))
print("games with <99 window rows:",
      {k: v for k, v in counts.items() if v != 99})
# validity cross-check: own-slot tok_mask (first OWN=12 of 24 tokens)
# should agree with y>=0
y = arr["y"]
tm = arr["tok_mask"][:, :12]
print("rows:", y.shape, "valid slots (y>=0):", int((y >= 0).sum()),
      "own tok_mask slots:", int(tm.sum()),
      "mismatch slots (tok_mask != (y>=0)):",
      int((tm != (y >= 0)).sum()))
print("teacher ids present:", sorted(set(arr["teacher"].tolist())))
json.dump(counts, open(os.path.join(WORK, "root_counts.json"), "w"), indent=1)
