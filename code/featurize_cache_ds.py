#!/usr/bin/env python3
"""featurize_cache_ds -- featurize a BC npz ONCE into a cache file in
exactly train_bc.build_dataset's output format (Xw, yw, tw, gw, Xb,
yb, gb), plus per-row strategy arrays (sw, sb).

Why: build_dataset featurizes inside train/calibrate (3x per model)
and is single-threaded pure Python; on this shared 4-core box the
66k-sample DeepSeek family corpus starved. Featurization is per-game
independent (seq state is intra-game only -- it resets at every game
boundary in build_dataset), so games are featurized in parallel with
a 2-process pool (mirroring the 2-thread training cap) and cached.

Equivalence: verified bitwise against train_bc_ds.build_dataset on
bc_data/deepseek_targets.npz (see bc_shadow_deepseek_report.md).

Usage: venv/bin/python featurize_cache_ds.py NPZ CACHE_OUT
"""

import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from bc_features import featurize  # noqa: E402
from train_bc_ds import VAL_FILES  # noqa: E402

VAL_SET = set(VAL_FILES)
_D = None  # per-process data dict


def _init(d):
    global _D
    _D = d


def featurize_game(idxs):
    """idxs: sample row indices of ONE game, in npz order. Replicates
    build_dataset's per-sample loop (seq state fresh at game start,
    which matches build_dataset: its seq branch requires step ==
    prev_step + 1 AND same game, and the first sample of a game in
    the full pass always takes the reset branch because the previous
    sample belongs to another game or a step gap)."""
    d = _D
    Xw, yw, tw, gw, sw = [], [], [], [], []
    Xb, yb, gb, sb = [], [], [], []
    seq_game, seq_prev_step, seq_prev_n = None, None, 0
    seq_prev_rec, seq_last_rec_step = 0, None
    for i in idxs:
        p = int(d["player"][i])
        step_i = int(d["step"][i])
        n_now = int(np.sum(d["workers_mask"][i][p]))
        if seq_game is not None and step_i == seq_prev_step + 1:
            lost = max(0, seq_prev_n + seq_prev_rec - n_now)
            since = 60 if seq_last_rec_step is None \
                else step_i - seq_last_rec_step
            seq = {"prev_n": float(seq_prev_n), "lost": float(lost),
                   "since": float(since)}
        else:
            seq = {"prev_n": float(n_now), "lost": 0.0, "since": 60.0}
            seq_last_rec_step = None
        W, B = featurize(d["carbon"][i], d["cash"][i], d["bases"][i],
                         d["workers"][i], d["workers_mask"][i],
                         d["trees"][i], d["trees_mask"][i],
                         p, step_i, seq=seq)
        seq_game = d["game_file"][i]
        seq_prev_step = step_i
        seq_prev_n = n_now
        seq_prev_rec = int(np.sum(d["base_actions"][i] > 0))
        if seq_prev_rec > 0:
            seq_last_rec_step = step_i
        is_val = d["game_file"][i] in VAL_SET
        mask = d["workers_mask"][i][p]
        for j in range(10):
            if mask[j]:
                Xw.append(W[j])
                yw.append(int(d["worker_actions"][i][j]))
                tw.append(int(d["workers"][i][p][j][2]))
                gw.append(bool(is_val))
                sw.append(str(d["strategy"][i]))
        for bi in range(2):
            if int(d["bases"][i][p][bi]) >= 0:
                Xb.append(B[bi])
                yb.append(int(d["base_actions"][i][bi]))
                gb.append(bool(is_val))
                sb.append(str(d["strategy"][i]))
    return (Xw, yw, tw, gw, sw, Xb, yb, gb, sb)


def main():
    npz_path, out_path = sys.argv[1], sys.argv[2]
    _z = np.load(npz_path, allow_pickle=True)
    d = {k: _z[k] for k in _z.files}
    games = {}
    for i in range(len(d["step"])):
        games.setdefault(str(d["game_file"][i]), []).append(i)
    print(f"{len(d['step'])} samples in {len(games)} games")
    keys = list(games.keys())
    with Pool(2, initializer=_init, initargs=(d,)) as pool:
        parts = pool.map(featurize_game, [games[k] for k in keys])
    Xw, yw, tw, gw, sw, Xb, yb, gb, sb = [], [], [], [], [], [], [], [], []
    for p in parts:
        Xw += p[0]; yw += p[1]; tw += p[2]; gw += p[3]; sw += p[4]
        Xb += p[5]; yb += p[6]; gb += p[7]; sb += p[8]
    np.savez_compressed(
        out_path,
        Xw=np.array(Xw, dtype=np.float32), yw=np.array(yw),
        tw=np.array(tw), gw=np.array(gw), sw=np.array(sw),
        Xb=np.array(Xb, dtype=np.float32), yb=np.array(yb),
        gb=np.array(gb), sb=np.array(sb))
    print(f"cached {len(yw)} worker rows, {len(yb)} base rows -> "
          f"{out_path}")


if __name__ == "__main__":
    main()
