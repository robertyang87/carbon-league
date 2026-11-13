#!/usr/bin/env python3
"""eval_bc_ds_offline -- per-group held-out top-1 for DeepSeek BC
weight files.

Featurizes ONLY the val games (deepseek_val_files.json) -- legitimate
because bc_features' seq state is purely intra-game (it resets at
every game boundary / step discontinuity in train_bc.build_dataset),
so val rows' features are identical whether or not train games are
featurized alongside. Groups: RL-policy-1 / RL-light-1 / NN-light
siblings / other NaMnSi.

Usage: venv/bin/python eval_bc_ds_offline.py WEIGHTS.npz [NPZ]
NPZ defaults to bc_data/deepseek_family.npz. Prints a JSON blob.
"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from bc_features import featurize  # noqa: E402
from train_bc_ds import VAL_FILES  # noqa: E402
from calibrate_base_ds import forward  # noqa: E402

VAL_SET = set(VAL_FILES)


def group_of(name):
    if name == "NaMnSi-DeepSeek-RL-policy-1":
        return "RL-policy-1"
    if name == "NaMnSi-DeepSeek-RL-light-1":
        return "RL-light-1"
    if "NN-light" in name:
        return "NN-light"
    return "other-NaMnSi"


def build_grouped_val(d):
    idx = [i for i in range(len(d["step"]))
           if d["game_file"][i] in VAL_SET]
    Xw, yw, tw, sw = [], [], [], []
    Xb, yb, sb = [], [], []
    seq_game, seq_prev_step, seq_prev_n = None, None, 0
    seq_prev_rec, seq_last_rec_step = 0, None
    for i in idx:
        p = int(d["player"][i])
        step_i = int(d["step"][i])
        n_now = int(np.sum(d["workers_mask"][i][p]))
        if d["game_file"][i] == seq_game and step_i == seq_prev_step + 1:
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
        gname = group_of(str(d["strategy"][i]))
        mask = d["workers_mask"][i][p]
        for j in range(10):
            if mask[j]:
                Xw.append(W[j])
                yw.append(int(d["worker_actions"][i][j]))
                tw.append(int(d["workers"][i][p][j][2]))
                sw.append(gname)
        for bi in range(2):
            if int(d["bases"][i][p][bi]) >= 0:
                Xb.append(B[bi])
                yb.append(int(d["base_actions"][i][bi]))
                sb.append(gname)
    return (np.array(Xw, dtype=np.float32), np.array(yw), np.array(tw),
            np.array(sw), np.array(Xb, dtype=np.float32), np.array(yb),
            np.array(sb))


def main():
    wpath = sys.argv[1]
    npz = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        HERE, "bc_data", "deepseek_family.npz")
    _z = np.load(npz, allow_pickle=True)
    d = {k: _z[k] for k in _z.files}
    Xw, yw, tw, sw, Xb, yb, sb = build_grouped_val(d)
    W = dict(np.load(os.path.join(HERE, wpath)))
    wpred = forward(W, "w_", Xw).argmax(1)
    bpred = forward(W, "b_", Xb).argmax(1)
    out = {"weights": wpath, "npz": os.path.basename(npz), "groups": {}}
    for g in ("RL-policy-1", "RL-light-1", "NN-light", "other-NaMnSi",
              "TARGETS", "ALL"):
        if g == "TARGETS":
            m = np.isin(sw, ["RL-policy-1", "RL-light-1"])
            mb = np.isin(sb, ["RL-policy-1", "RL-light-1"])
        elif g == "ALL":
            m = np.ones(len(sw), dtype=bool)
            mb = np.ones(len(sb), dtype=bool)
        else:
            m = sw == g
            mb = sb == g
        if not m.sum():
            continue
        e = {"worker_n": int(m.sum()),
             "worker_acc": float(np.mean(wpred[m] == yw[m])),
             "worker_stay_baseline": float(np.mean(yw[m] == 0))}
        for tname, tval in (("collector", 0), ("planter", 1)):
            mt = m & (tw == tval)
            if mt.sum():
                e[f"{tname}_acc"] = float(np.mean(wpred[mt] == yw[mt]))
                e[f"{tname}_stay_baseline"] = float(
                    np.mean(yw[mt] == 0))
                e[f"{tname}_n"] = int(mt.sum())
        if mb.sum():
            e["base_n"] = int(mb.sum())
            e["base_acc"] = float(np.mean(bpred[mb] == yb[mb]))
            for cname, cval in (("RECC", 1), ("RECP", 2)):
                mc = mb & (yb == cval)
                if mc.sum():
                    e[f"base_{cname}_recall"] = float(
                        np.mean(bpred[mc] == cval))
                    e[f"base_{cname}_n"] = int(mc.sum())
        out["groups"][g] = e
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
