#!/usr/bin/env python3
"""build_distill_dataset -- join bc_v53_corpus.npz (labels + canonical
state) with distill_capture.npz (teacher belief + tracker) on
(file, player, step), featurize with distill_features.featurize, and
split by seed (whole games; seed % 5 == 4 -> validation, so a seed's
side-swapped mirror games never straddle the split).

Outputs bc_data/distill_dataset.npz:
  W_x float16 [Nw,247], W_y int64, W_split/W_phase/W_state/W_type
  B_x float16 [Nb,108], B_y int64, B_split
  plus per-head mean/std computed on the TRAIN split (float32).
Phase: 0 opening (step<50), 1 treewar (50-199), 2 endgame (>=200).
State: value diff = (cash_me + my cargo) - (cash_opp + opp cargo);
  >= +300 -> 1 (favourable), <= -300 -> -1 (adverse), else 0.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from distill_features import featurize  # noqa: E402

CORPUS = HERE / "bc_data" / "bc_v53_corpus.npz"
CAPTURE = HERE / "bc_data" / "distill_capture.npz"
OUT = HERE / "bc_data" / "distill_dataset.npz"
SEED_RE = re.compile(r"seed(\d+)")


def merge_shards(nshards=2):
    keys = ["W_x", "W_y", "W_split", "W_phase", "W_state", "W_type",
            "W_step", "W_dest_tree", "W_dest5", "B_x", "B_y", "B_split"]
    parts = {k: [] for k in keys}
    for s in range(nshards):
        d = np.load(HERE / "bc_data" / f"distill_dataset_shard{s}.npz")
        for k in keys:
            parts[k].append(d[k])
    A = {k: np.concatenate(v, axis=0) for k, v in parts.items()}

    def stats(X, sp):
        tr = X[sp == 0].astype(np.float32)
        mean = tr.mean(axis=0)
        std = tr.std(axis=0)
        std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    W_mean, W_std = stats(A["W_x"], A["W_split"])
    B_mean, B_std = stats(A["B_x"], A["B_split"])
    np.savez_compressed(OUT, **A, W_mean=W_mean, W_std=W_std,
                        B_mean=B_mean, B_std=B_std)
    W_y, B_y = A["W_y"], A["B_y"]
    W_sp, B_sp = A["W_split"], A["B_split"]
    meta = {
        "worker_samples": int(len(W_y)),
        "worker_val": int((W_sp == 1).sum()),
        "base_samples": int(len(B_y)),
        "base_val": int((B_sp == 1).sum()),
        "worker_label_dist_train": {
            str(k): int(v) for k, v in
            zip(*np.unique(W_y[W_sp == 0], return_counts=True))},
        "base_label_dist_train": {
            str(k): int(v) for k, v in
            zip(*np.unique(B_y[B_sp == 0], return_counts=True))},
    }
    print(json.dumps(meta, indent=1))
    (HERE / "distill_dataset_meta.json").write_text(
        json.dumps(meta, indent=1))
    print("wrote", OUT)


def main():
    shard = None
    nshards = 1
    if "--shard" in sys.argv:
        shard = int(sys.argv[sys.argv.index("--shard") + 1])
        nshards = int(sys.argv[sys.argv.index("--nshards") + 1])
    if "--merge" in sys.argv:
        merge_shards(nshards=int(sys.argv[sys.argv.index("--merge") + 1])
                     if sys.argv[sys.argv.index("--merge") + 1].isdigit()
                     else 2)
        return
    cor = np.load(CORPUS, allow_pickle=False)
    cap = np.load(CAPTURE, allow_pickle=False)
    if "--extra" in sys.argv:
        # Merge a second (corpus, capture) pair -- e.g. DAgger
        # correction data -- with the extra corpus repeated --repeat
        # times (upweighting by duplication; capture rows are shared
        # through the (file, player, step) index, not repeated).
        rep = int(sys.argv[sys.argv.index("--repeat") + 1]) \
            if "--repeat" in sys.argv else 1
        cor_keys = list(cor.files)
        cap_keys = list(cap.files)
        for tag2 in sys.argv[sys.argv.index("--extra") + 1].split(","):
            cor2 = np.load(HERE / "bc_data" / f"{tag2}_corpus.npz",
                           allow_pickle=False)
            cap2 = np.load(HERE / "bc_data" / f"{tag2}_capture.npz",
                           allow_pickle=False)
            cor = {k: np.concatenate([cor[k]] + [cor2[k]] * rep)
                   for k in cor_keys}
            off = len(cap["files"])
            cap = {
                k: (np.concatenate([cap[k], cap2[k]]) if k == "files"
                    else np.concatenate([cap[k], cap2[k] + off])
                    if k == "file_idx"
                    else np.concatenate([cap[k], cap2[k]]))
                for k in cap_keys}
            print(f"merged extra={tag2} x{rep}: "
                  f"corpus={len(cor['step'])} "
                  f"capture={len(cap['step'])}", flush=True)
    n = len(cor["step"])
    print(f"corpus samples={n} shard={shard}/{nshards}", flush=True)

    cap_files = [str(f) for f in cap["files"]]
    _fi = cap["file_idx"]
    _pl = cap["player"]
    _st = cap["step"]
    cap_index = {}
    for i in range(len(_st)):
        cap_index[(cap_files[int(_fi[i])], int(_pl[i]),
                   int(_st[i]))] = i
    print(f"capture rows={len(cap_index)}", flush=True)

    # Hoist ALL npz arrays out of the loop: NpzFile[key] re-inflates
    # the whole array on every access -- indexing it per sample made
    # the first build attempt ~100x slower (560 CPU-s, no output).
    C_file = cor["game_file"]
    C_player = cor["player"]
    C_step = cor["step"]
    C_cash = cor["cash"]
    C_workers = cor["workers"]
    C_wmask = cor["workers_mask"]
    C_bases = cor["bases"]
    C_trees = cor["trees"]
    C_tmask = cor["trees_mask"]
    C_wa = cor["worker_actions"]
    C_ba = cor["base_actions"]
    K_belief = cap["belief"]
    K_known = cap["known"]
    K_prev = cap["prev_act"]
    K_seq = cap["seq"]
    K_since = cap["since_dep"]

    W_x, W_y, W_sp, W_ph, W_st, W_ty = [], [], [], [], [], []
    W_step, W_dt, W_d5 = [], [], []
    B_x, B_y, B_sp = [], [], []
    DELTA = {0: (0, 0), 1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}
    missing = 0
    for i in range(n):
        if shard is not None and i % nshards != shard:
            continue
        fname = str(C_file[i])
        p = int(C_player[i])
        step = int(C_step[i])
        ci = cap_index.get((fname, p, step))
        if ci is None:
            missing += 1
            continue
        seed = int(SEED_RE.search(fname).group(1))
        split = 1 if seed % 5 == 4 else 0
        cash = C_cash[i].astype(np.float64)
        workers = C_workers[i]
        wmask = C_wmask[i]
        wx, bx = featurize(
            p, K_belief[ci], K_known[ci], cash,
            C_bases[i].astype(np.int64), workers, wmask,
            C_trees[i], C_tmask[i], step,
            K_prev[ci], K_seq[ci],
            float(K_since[ci]))
        q = 1 - p
        myw = workers[p][wmask[p]]
        enw = workers[q][wmask[q]]
        vdiff = (cash[p] + float(myw[:, 1].sum())) - \
            (cash[q] + float(enw[:, 1].sum()))
        state = 1 if vdiff >= 300 else (-1 if vdiff <= -300 else 0)
        phase = 0 if step < 50 else (1 if step < 200 else 2)
        wa = C_wa[i]
        tree_owner = {}
        tarr = C_trees[i][C_tmask[i]]
        for trow in tarr:
            tree_owner[int(trow[0])] = int(trow[2])
        for j in range(len(wx)):
            W_x.append(wx[j])
            W_y.append(int(wa[j]))
            W_sp.append(split)
            W_ph.append(phase)
            W_st.append(state)
            W_ty.append(int(myw[j, 2]))
            W_step.append(step)
            wr, wc = divmod(int(myw[j, 0]), 15)
            d5 = []
            for a in range(5):
                dr, dc = DELTA[a]
                dest = ((wr + dr) % 15) * 15 + ((wc + dc) % 15)
                owner = tree_owner.get(dest)
                d5.append(0 if owner is None
                          else (1 if owner == p else 2))
            W_dt.append(d5[int(wa[j])])
            W_d5.append(d5)
        ba = C_ba[i]
        nb = int((C_bases[i][p] >= 0).sum())
        for j in range(len(bx)):
            B_x.append(bx[j])
            B_y.append(int(ba[j]))
            B_sp.append(split)
        if i % 10000 == 0:
            print(f"  {i}/{n}", flush=True)
    print(f"joined missing={missing}")
    assert missing == 0, "capture/corpus join incomplete"

    W_x = np.stack(W_x).astype(np.float16)
    B_x = np.stack(B_x).astype(np.float16)
    W_y = np.asarray(W_y, dtype=np.int64)
    B_y = np.asarray(B_y, dtype=np.int64)
    W_sp = np.asarray(W_sp, dtype=np.int8)
    B_sp = np.asarray(B_sp, dtype=np.int8)
    arrays = dict(
        W_x=W_x, W_y=W_y, W_split=W_sp,
        W_phase=np.asarray(W_ph, dtype=np.int8),
        W_state=np.asarray(W_st, dtype=np.int8),
        W_type=np.asarray(W_ty, dtype=np.int8),
        W_step=np.asarray(W_step, dtype=np.int16),
        W_dest_tree=np.asarray(W_dt, dtype=np.int8),
        W_dest5=np.asarray(W_d5, dtype=np.int8),
        B_x=B_x, B_y=B_y, B_split=B_sp)
    if shard is not None:
        sp_out = HERE / "bc_data" / f"distill_dataset_shard{shard}.npz"
        np.savez_compressed(sp_out, **arrays)
        print(f"wrote shard {sp_out}: W={len(W_y)} B={len(B_y)}",
              flush=True)
        return

    def stats(X, sp):
        tr = X[sp == 0].astype(np.float32)
        mean = tr.mean(axis=0)
        std = tr.std(axis=0)
        std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    W_mean, W_std = stats(W_x, W_sp)
    B_mean, B_std = stats(B_x, B_sp)
    if "--extra" in sys.argv:
        out_path = HERE / "bc_data" / "distill_dataset_dagger.npz"
        meta_path = HERE / "distill_dataset_dagger_meta.json"
    else:
        out_path = OUT
        meta_path = HERE / "distill_dataset_meta.json"
    np.savez_compressed(
        out_path, **arrays,
        W_mean=W_mean, W_std=W_std, B_mean=B_mean, B_std=B_std)
    meta = {
        "worker_samples": int(len(W_y)),
        "worker_val": int((W_sp == 1).sum()),
        "base_samples": int(len(B_y)),
        "base_val": int((B_sp == 1).sum()),
        "worker_label_dist_train": {
            str(k): int(v) for k, v in
            zip(*np.unique(W_y[W_sp == 0], return_counts=True))},
        "base_label_dist_train": {
            str(k): int(v) for k, v in
            zip(*np.unique(B_y[B_sp == 0], return_counts=True))},
    }
    print(json.dumps(meta, indent=1))
    meta_path.write_text(json.dumps(meta, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
