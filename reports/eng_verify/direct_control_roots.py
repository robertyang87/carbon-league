#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT (not approved, not committed) — same-machine direct control.

Purpose: if the control gate in the approved v2 proposal (bit-identical
q/Q_mean vs L20's teacher shards) cannot be met on a different machine
because of cross-machine float noise, this script provides the
alternative control: run production evaluate_root DIRECTLY (teacher
trunk, no harness code at all) for the 8 control roots on the SAME
seat, and compare the harness control output against it bit-for-bit.
That isolates harness fidelity from machine noise.

Evidence that motivated this draft (local 1-root control smoke,
2026-11-18, this VM vs L20 teacher shards):
  - harness run1 == run2 bit-identical on q/Q/Q_std/priors/arms
    (harness deterministic on one machine);
  - vs L20 shards: arms identical, priors pass-through identical,
    diagnostic gen_candidates arms match, priors max diff 1.357e-06
    (identical across both runs — a stable machine difference),
    q max diff 2.20e-07, Q max diff 7.03e-08, argmax same;
  - act-call counts exact: 648/root (harness) + 32 (diagnostic
    gen_candidates) = 680 = production.
  Conclusion: harness structure is faithful; the residual ~1e-7 is
  cross-machine float noise. The approved gate's (b) clause bar
  (uniform deviation <= 1e-12) does not cover 1e-7, so on spot-64
  the gate as written would most likely stop under (b).

Usage (after the seat identity verification, before the harness
control gate):
  python3 direct_control_roots.py --code-dir <pinned code dir>
      --replays <appendix dir> --weights <teacher weights npz>
      --out <out dir> [--workers 2]
Writes one shard npz per game (same schema as k0_annotate shards,
caliber "direct-control") plus diagnostics.jsonl. Compare with the
harness control output by np.array_equal on q/Q/Q_std/priors/arms.

This file is a DRAFT. Do not run it as part of the approved
procedure unless the strategy group approves the gate amendment.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = HERE
for _i, _a in enumerate(sys.argv):
    if _a == "--code-dir" and _i + 1 < len(sys.argv):
        _CODE_DIR = sys.argv[_i + 1]
    elif _a.startswith("--code-dir="):
        _CODE_DIR = _a.split("=", 1)[1]
sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, HERE)

import cnet2_feat as F
import cnet2_np as NP
import k0_annotate as K0
import stage0_teacher as T
import stage1_teacher_gate as G

CONTROL_ROOTS = {(200, 0), (200, 1), (201, 0), (201, 1)}
GAMES = ("selfplay-994400", "selfplay-994401")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _direct_root(task):
    g, gid, step, seat, replay, weights = task
    spec = {"game": gid, "path": replay, "step": step, "me": seat, "ft": 299}
    scene = G.extract_scene(spec)
    feats = [F.Feat.observe(scene, 2, 7000)]
    scene["fm"] = feats[0]
    sim = G._fresh_sim(scene)
    me_snap = G._teacher_guard_snaps(scene, "sample")
    res = T.evaluate_root(
        scene, sim, feats, me_snap, K0.TEACHER_VERSION, "sample",
        _W, 7000, feats[0])
    row = K0._root_row(spec, res, K0.TEACHER_VERSION, "sample")
    return g, step, seat, row


_W = {}


def _init(weights_path):
    global _W
    _W = {"pol": NP.Policy(dict(np.load(weights_path)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", default=HERE)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    for fname in ("stage0_teacher.py", "k0_annotate.py", "cnet2_np.py",
                  "cnet2_search.py", "cnet2_feat.py", "fastsim.py",
                  "stage1_teacher_gate.py"):
        p = os.path.join(args.code_dir, fname)
        print(f"[direct] {fname} {_md5(p)}")
    print(f"[direct] weights {_md5(args.weights)}")

    _init(args.weights)
    tasks = []
    for g in GAMES:
        gid = int(g.split("-")[1])
        for step, seat in sorted(CONTROL_ROOTS):
            tasks.append((g, gid, step, seat,
                          os.path.join(args.replays, g + ".json.gz"),
                          args.weights))
    rows = {g: {k: [] for k in K0.FIELDS} for g in GAMES}
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init,
                                 initargs=(args.weights,)) as ex:
            results = list(ex.map(_direct_root, tasks))
    else:
        results = [_direct_root(t) for t in tasks]
    for g, step, seat, row in results:
        for k in K0.FIELDS:
            rows[g][k].append(row[k])
    os.makedirs(os.path.join(args.out, "shards"), exist_ok=True)
    for g in GAMES:
        out = os.path.join(args.out, "shards", g + ".npz")
        np.savez(out, **{k: np.asarray(v) for k, v in rows[g].items()})
        print(f"[direct] wrote {out} ({len(rows[g]['step'])} roots)")
    print("[direct] done")


if __name__ == "__main__":
    main()
