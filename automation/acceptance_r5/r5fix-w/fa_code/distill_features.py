"""distill_features -- THE featurizer for Case-B distillation.

One implementation, two callers:
  * build_distill_dataset.py (offline, from corpus npz + belief capture)
  * distilled_v1.py (deployment; this file's source is inlined verbatim
    by gen_distilled.py, so the deployed policy featurizes with exactly
    the code the training data was built with -- zero feature drift).

Canonical state (all deployment-obtainable; no opponent-private data):
  p        acting player (0/1)
  belief   float32[225]  v53 Belief.estimate() grid for player p
  known    bool[225]     v53 Belief.known mask
  cash     float64[2]
  bases    int64[2,2]    base cell per player/slot, -1 = absent
  workers  float32[2,10,3] (pos, cargo, type 0=C 1=P) in corpus slot
           order (wid numeric suffix ascending), wmask bool[2,10]
  trees    float32[30,3] (pos, age, owner), tmask bool[30]
  step     int
  prev_act int8[10]  acting player's per-slot previous action
           (0 stay, 1 UP, 2 RIGHT, 3 DOWN, 4 LEFT, 5 unknown)
  seq      float32[3] (prev_n, lost, since) tracker triple
  since_dep float  steps since last observed own deposit (<=60)

Outputs: (W_x float32[n_my_workers, 247], B_x float32[n_my_bases, 108])
"""

import numpy as np

SIZE = 15
N = SIZE * SIZE

# toroidal patch index tables: PATCH7[i] = 49 cell indices of the 7x7
# neighbourhood centred on cell i, row-major in (dr, dc), dr,dc in -3..3
def _patch_idx(rad):
    out = np.zeros((N, (2 * rad + 1) ** 2), dtype=np.int64)
    for i in range(N):
        r, c = divmod(i, SIZE)
        k = 0
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                out[i, k] = ((r + dr) % SIZE) * SIZE + ((c + dc) % SIZE)
                k += 1
    return out


PATCH7 = _patch_idx(3)
PATCH5 = _patch_idx(2)
_ROWS = np.arange(N) // SIZE
_COLS = np.arange(N) % SIZE


def _tdist(a, b):
    """Toroidal Manhattan distance between cell index arrays/scalars."""
    dr = np.abs(_ROWS[a] - _ROWS[b])
    dc = np.abs(_COLS[a] - _COLS[b])
    return np.minimum(dr, SIZE - dr) + np.minimum(dc, SIZE - dc)


def _globals(p, belief, known, cash, bases, workers, wmask, trees,
             tmask, step, seq, since_dep):
    q = 1 - p
    myw = workers[p][wmask[p]]
    enw = workers[q][wmask[q]]
    my_nC = float(np.sum(myw[:, 2] == 0)) if len(myw) else 0.0
    my_nP = float(np.sum(myw[:, 2] == 1)) if len(myw) else 0.0
    en_nC = float(np.sum(enw[:, 2] == 0)) if len(enw) else 0.0
    en_nP = float(np.sum(enw[:, 2] == 1)) if len(enw) else 0.0
    alive = trees[tmask & (trees[:, 1] < 50)] if tmask.any() else trees[:0]
    my_trees = float(np.sum(alive[:, 2] == p)) if len(alive) else 0.0
    en_trees = float(np.sum(alive[:, 2] == q)) if len(alive) else 0.0
    tot_trees = float(len(alive))
    if my_trees <= 0:
        price = 20.0
    else:
        price = 20.0 + 5.0 * (1.235 ** tot_trees)
    price = min(price, 2000.0)
    bsum = float(belief.sum())
    bmax = float(belief.max())
    top5 = float(np.sort(belief)[-5:].mean())
    my_cargo = float(myw[:, 1].sum()) if len(myw) else 0.0
    en_cargo = float(enw[:, 1].sum()) if len(enw) else 0.0
    g = np.array([
        step / 300.0,
        (299 - step) / 300.0,
        1.0 if step < 50 else 0.0,
        1.0 if step >= 200 else 0.0,
        cash[p] / 10000.0,
        cash[q] / 10000.0,
        (cash[p] - cash[q]) / 10000.0,
        my_nC / 10.0, my_nP / 10.0, en_nC / 10.0, en_nP / 10.0,
        my_trees / 20.0, en_trees / 20.0, tot_trees / 60.0,
        price / 400.0,
        bsum / 22500.0, bmax / 100.0, float(known.mean()), top5 / 100.0,
        seq[0] / 10.0, min(float(seq[1]), 10.0) / 10.0,
        min(float(seq[2]), 60.0) / 60.0,
        min(float(since_dep), 60.0) / 60.0,
        my_cargo / 2000.0, en_cargo / 2000.0,
        ((cash[p] + my_cargo) - (cash[q] + en_cargo)) / 10000.0,
    ], dtype=np.float32)
    return g


def _grids(p, bases, workers, wmask, trees, tmask):
    q = 1 - p
    tree_grid = np.zeros(N, dtype=np.float32)
    if tmask.any():
        t = trees[tmask]
        mag = 0.3 + 0.7 * np.minimum(t[:, 1], 50.0) / 50.0
        sgn = np.where(t[:, 2] == p, 1.0, -1.0)
        tree_grid[t[:, 0].astype(np.int64)] = (sgn * mag).astype(np.float32)
    worker_grid = np.zeros(N, dtype=np.float32)
    for pl, s in ((p, 1.0), (q, -1.0)):
        w = workers[pl][wmask[pl]]
        if len(w):
            mag = 0.4 + 0.6 * np.minimum(w[:, 1], 500.0) / 500.0
            worker_grid[w[:, 0].astype(np.int64)] = \
                (s * mag).astype(np.float32)
    base_grid = np.zeros(N, dtype=np.float32)
    for pl, s in ((p, 1.0), (q, -1.0)):
        for b in bases[pl]:
            if int(b) >= 0:
                base_grid[int(b)] = s
    return tree_grid, worker_grid, base_grid


def featurize(p, belief, known, cash, bases, workers, wmask, trees,
              tmask, step, prev_act, seq, since_dep):
    q = 1 - p
    belief = np.asarray(belief, dtype=np.float32)
    g = _globals(p, belief, known, cash, bases, workers, wmask, trees,
                 tmask, step, seq, since_dep)
    tree_grid, worker_grid, base_grid = _grids(p, bases, workers, wmask,
                                               trees, tmask)
    bel_n = belief / 100.0

    my_bases = [int(b) for b in bases[p] if int(b) >= 0]
    en_bases = [int(b) for b in bases[q] if int(b) >= 0]
    myw = workers[p][wmask[p]]
    enw = workers[q][wmask[q]]
    enc = enw[enw[:, 2] == 0] if len(enw) else enw[:0]
    tr = trees[tmask] if tmask.any() else trees[:0]
    my_trees_pos = tr[tr[:, 2] == p][:, 0].astype(np.int64) \
        if len(tr) else np.zeros(0, dtype=np.int64)
    en_trees_pos = tr[tr[:, 2] == q][:, 0].astype(np.int64) \
        if len(tr) else np.zeros(0, dtype=np.int64)
    all_trees_pos = tr[:, 0].astype(np.int64) if len(tr) \
        else np.zeros(0, dtype=np.int64)
    tree_cells = set(int(x) for x in all_trees_pos)
    my_tree_cells = set(int(x) for x in my_trees_pos)

    def _mindist(pos, targets):
        if len(targets) == 0:
            return 15.0
        return float(_tdist(pos, np.asarray(targets)).min())

    # ---- worker features ----
    W = []
    slots = np.where(wmask[p])[0]
    for si, j in enumerate(slots):
        pos = int(workers[p, j, 0])
        cargo = float(workers[p, j, 1])
        wtype = int(workers[p, j, 2])
        idx7 = PATCH7[pos]
        patch = np.concatenate([
            bel_n[idx7],
            tree_grid[idx7],
            worker_grid[idx7],
            base_grid[idx7],
        ])
        pbel = bel_n[idx7]
        am = int(np.argmax(pbel))
        adr, adc = am // 7 - 3, am % 7 - 3
        if len(enc):
            d_enc_all = _tdist(pos, enc[:, 0].astype(np.int64))
            k = int(np.argmin(d_enc_all))
            d_enc = float(d_enc_all[k])
            rel_enc = float(np.clip(
                (enc[k, 1] - cargo) / 500.0, -1.0, 1.0))
        else:
            d_enc, rel_enc = 15.0, 0.0
        others = [int(workers[p, jj, 0]) for jj in slots if jj != j]
        d_own = _mindist(pos, others) if others else 15.0
        pa = np.zeros(6, dtype=np.float32)
        pa[int(prev_act[si]) if si < len(prev_act) else 5] = 1.0
        sc = np.array([
            float(wtype),
            min(cargo, 1000.0) / 500.0,
            1.0 if cargo >= 150 else 0.0,
            1.0 if cargo >= 300 else 0.0,
            1.0 if pos in my_bases else 0.0,
            1.0 if pos in tree_cells else 0.0,
            1.0 if pos in my_tree_cells else 0.0,
            _mindist(pos, my_bases) / 14.0,
            _mindist(pos, en_bases) / 14.0,
            _mindist(pos, my_trees_pos) / 14.0,
            _mindist(pos, en_trees_pos) / 14.0,
            float(bel_n[pos]),
            float(pbel.max()),
            adr / 3.0, adc / 3.0,
            *pa.tolist(),
            d_enc / 14.0,
            rel_enc,
            1.0 if d_enc <= 3 else 0.0,
            d_own / 14.0,
        ], dtype=np.float32)
        W.append(np.concatenate([patch, sc, g]))
    W_x = np.stack(W).astype(np.float32) if W else \
        np.zeros((0, 247), dtype=np.float32)

    # ---- base features ----
    B = []
    for bi, bpos in enumerate(my_bases):
        idx5 = PATCH5[bpos]
        patch = np.concatenate([
            bel_n[idx5], tree_grid[idx5], worker_grid[idx5]])
        d_en = _tdist(bpos, enw[:, 0].astype(np.int64)) if len(enw) \
            else np.zeros(0)
        d_my = _tdist(bpos, myw[:, 0].astype(np.int64)) if len(myw) \
            else np.zeros(0)
        if len(enc):
            d_enc_all = _tdist(bpos, enc[:, 0].astype(np.int64))
            k = int(np.argmin(d_enc_all))
            b_denc, b_cenc = float(d_enc_all[k]), float(enc[k, 1])
        else:
            b_denc, b_cenc = 15.0, 0.0
        other_base = my_bases[1 - bi] if len(my_bases) == 2 else bpos
        sc = np.array([
            float(np.sum(d_en <= 2)) / 5.0 if len(enw) else 0.0,
            float(np.sum(d_my <= 2)) / 10.0 if len(myw) else 0.0,
            b_denc / 14.0,
            min(b_cenc, 1000.0) / 500.0,
            float(np.sum((myw[:, 2] == 0) & (myw[:, 1] > 0))) / 10.0
            if len(myw) else 0.0,
            1.0 if bi == 1 else 0.0,
            float(_tdist(bpos, other_base)) / 14.0,
        ], dtype=np.float32)
        B.append(np.concatenate([g, patch, sc]))
    B_x = np.stack(B).astype(np.float32) if B else \
        np.zeros((0, 108), dtype=np.float32)
    return W_x, B_x


W_IN = 247
B_IN = 108
