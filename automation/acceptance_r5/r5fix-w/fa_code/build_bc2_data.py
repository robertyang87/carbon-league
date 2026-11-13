"""build_bc2_data -- precompute BC features for the stage-2 A/B:
control arm (bc_features.featurize) and belief arm (belief.featurize2)
over bc_data/claude_family_v2.npz, same samples, same val split
(train_bc.VAL_FILES), same seq-state semantics as train_bc.

The belief tracker runs along each game's step sequence from the
extracted player's perspective and is reset on game change / step
gaps. Output: bc_data/bc2_dataset.npz with Xw/Xb (control),
Xw2/Xb2 (belief), labels, types, val masks.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bc_features import featurize  # noqa: E402
from belief import BeliefTracker, featurize2  # noqa: E402
from train_bc import VAL_FILES  # noqa: E402


def main():
    z = np.load(os.path.join(HERE, "bc_data", "claude_family_v2.npz"),
                allow_pickle=True)
    d = {k: z[k] for k in z.files}
    n = len(d["step"])
    # order samples by (game_file, step) -- extraction already emits
    # them so, but do not rely on it.
    order = sorted(range(n), key=lambda i: (str(d["game_file"][i]),
                                            int(d["step"][i])))
    is_val = np.array([str(f) in VAL_FILES for f in d["game_file"]])

    Xw, yw, tw, gw, Xb, yb, gb = [], [], [], [], [], [], []
    Xw2, Xb2 = [], []
    tracker = BeliefTracker()
    cur_game = None
    seq_game, seq_prev_step, seq_prev_n = None, None, 0
    seq_prev_rec, seq_last_rec_step = 0, None
    gaps = 0
    t0 = time.time()
    for i in order:
        game = str(d["game_file"][i])
        p = int(d["player"][i])
        step_i = int(d["step"][i])
        if game != cur_game:
            tracker.reset()
            cur_game = game
        elif step_i != tracker.last_step + 1 and tracker.last_step >= 0:
            gaps += 1
        carbon = d["carbon"][i]
        workers = d["workers"][i]
        wmask = d["workers_mask"][i]
        trees = d["trees"][i]
        tmask = d["trees_mask"][i]
        belief = tracker.update(carbon, workers, wmask, trees, tmask,
                                p, step_i)
        n_now = int(np.sum(wmask[p]))
        if game == seq_game and step_i == seq_prev_step + 1:
            lost = max(0, seq_prev_n + seq_prev_rec - n_now)
            since = 60 if seq_last_rec_step is None \
                else step_i - seq_last_rec_step
            seq = {"prev_n": float(seq_prev_n), "lost": float(lost),
                   "since": float(since)}
        else:
            seq = {"prev_n": float(n_now), "lost": 0.0, "since": 60.0}
            seq_last_rec_step = None
        args = (carbon, d["cash"][i], d["bases"][i], workers, wmask,
                trees, tmask, p, step_i)
        W, B = featurize(*args, seq=seq)
        W2, B2 = featurize2(*args, belief, seq=seq)
        seq_game, seq_prev_step, seq_prev_n = game, step_i, n_now
        seq_prev_rec = int(np.sum(d["base_actions"][i] > 0))
        if seq_prev_rec > 0:
            seq_last_rec_step = step_i
        wmask_p = wmask[p]
        for j in range(10):
            if wmask_p[j]:
                Xw.append(W[j])
                Xw2.append(W2[j])
                yw.append(int(d["worker_actions"][i][j]))
                tw.append(int(workers[p][j][2]))
                gw.append(bool(is_val[i]))
        for bi in range(2):
            if int(d["bases"][i][p][bi]) >= 0:
                Xb.append(B[bi])
                Xb2.append(B2[bi])
                yb.append(int(d["base_actions"][i][bi]))
                gb.append(bool(is_val[i]))
    print(f"featurized {n} points ({gaps} step gaps) "
          f"in {time.time() - t0:.1f}s")
    out = dict(
        Xw=np.array(Xw, dtype=np.float32), yw=np.array(yw),
        tw=np.array(tw), gw=np.array(gw, dtype=bool),
        Xb=np.array(Xb, dtype=np.float32), yb=np.array(yb),
        gb=np.array(gb, dtype=bool),
        Xw2=np.array(Xw2, dtype=np.float32),
        Xb2=np.array(Xb2, dtype=np.float32))
    np.savez(os.path.join(HERE, "bc_data", "bc2_dataset.npz"), **out)
    print({k: v.shape for k, v in out.items()})
    print("val games:", sorted({str(f) for f, v in
                                 zip(d['game_file'], is_val) if v}))
    with open(os.path.join(HERE, "bc_data", "bc2_dataset_meta.json"),
              "w") as f:
        json.dump({"step_gaps": gaps, "n_points": n}, f)


if __name__ == "__main__":
    main()
