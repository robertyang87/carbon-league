"""bc_features -- featurization for the Carbon League BC shadow policy.

LEGALITY CONTRACT (read before touching): every quantity computed here
is derivable from the observation a submitted agent actually receives
in a live game:
  * carbon  = the player's OWN visibility-filtered carbon map
              (collectors see their cell, planters their 4-neighbours,
              everything else 0). The full-board truth (full_carbon in
              replay files) is NEVER used.
  * players = shared block: both sides' cash, bases, workers
              (pos/cargo/type) and trees (pos/age/owner) are fully
              visible in the real observation.
No hidden state, no opponent-private information.

Canonical state arrays (identical layout to bc_data/*.npz):
  carbon  (225,) float      visible carbon, row-major idx = row*15+col
  cash    (2,)   float
  bases   (2,2)  int        base cell idx per player, -1 if missing
  workers (2,10,3) float    (pos, cargo, type) type 0=COLLECTOR 1=PLANTER
  wmask   (2,10) bool
  trees   (30,3) float      (pos, age, owner)
  tmask   (30,)  bool
  player  int               whose decision point this is
  step    int

Geometry matches fastsim.py / helpers.py: idx = (14-y)*15 + x,
UP = +y. Patches are read in (x,y) space with toroidal wrap.

This module is numpy-only on purpose: train_bc.py embeds its source
verbatim into shadow_alpha.py so training and live inference can never
drift apart.
"""

import numpy as np

SIZE = 15
N = SIZE * SIZE

_xs = np.arange(N) % SIZE
_ys = SIZE - 1 - np.arange(N) // SIZE


def _idx(x, y):
    return (SIZE - 1 - (y % SIZE)) * SIZE + (x % SIZE)


_dxa = np.abs(_xs[:, None] - _xs[None, :])
_dya = np.abs(_ys[:, None] - _ys[None, :])
DIST = np.minimum(_dxa, SIZE - _dxa) + np.minimum(_dya, SIZE - _dya)

# 5x5 patch (dy +2..-2, dx -2..+2) and 3x3 patch (dy +1..-1, dx -1..+1)
PATCH = np.zeros((N, 25), dtype=np.int64)
PATCH3 = np.zeros((N, 9), dtype=np.int64)
for _i in range(N):
    _x, _y = int(_xs[_i]), int(_ys[_i])
    _k = 0
    for _dy in (2, 1, 0, -1, -2):
        for _dx in (-2, -1, 0, 1, 2):
            PATCH[_i, _k] = _idx(_x + _dx, _y + _dy)
            _k += 1
    _k = 0
    for _dy in (1, 0, -1):
        for _dx in (-1, 0, 1):
            PATCH3[_i, _k] = _idx(_x + _dx, _y + _dy)
            _k += 1

GLOBAL_DIM = 13
WORKER_LOCAL_DIM = 111
WORKER_DIM = GLOBAL_DIM + WORKER_LOCAL_DIM
BASE_LOCAL_DIM = 12
BASE_DIM = GLOBAL_DIM + BASE_LOCAL_DIM


def _global_feats(carbon, cash, bases, workers, wmask, trees, tmask,
                  player, step):
    s, o = player, 1 - player
    n_coll = np.zeros(2)
    n_plan = np.zeros(2)
    carried = np.zeros(2)
    for q in (0, 1):
        m = wmask[q]
        t = workers[q, :, 2]
        n_coll[q] = np.sum(m & (t == 0))
        n_plan[q] = np.sum(m & (t == 1))
        carried[q] = np.sum(workers[q, :, 1] * m)
    n_tree = np.zeros(2)
    if np.any(tmask):
        own = trees[:, 2]
        n_tree[0] = np.sum(tmask & (own == 0))
        n_tree[1] = np.sum(tmask & (own == 1))
    g = np.array([
        step / 300.0,
        np.clip(cash[s] / 1000.0, 0, 5),
        np.clip(cash[o] / 1000.0, 0, 5),
        np.clip((cash[s] - cash[o]) / 2000.0, -2, 2),
        n_coll[s] / 10.0,
        n_plan[s] / 10.0,
        n_coll[o] / 10.0,
        n_plan[o] / 10.0,
        n_tree[s] / 15.0,
        n_tree[o] / 15.0,
        np.clip(carried[s] / 1000.0, 0, 3),
        np.clip(np.sum(carbon) / 3000.0, 0, 3),
        np.sum(wmask[s]) / 10.0,
    ], dtype=np.float64)
    return g


def _board_maps(workers, wmask, trees, tmask, player):
    """Per-cell owner/presence maps used by the local patches."""
    s, o = player, 1 - player
    tree_map = np.zeros(N)          # +1 own tree, -1 enemy tree
    tree_age = np.zeros(N)
    if np.any(tmask):
        for pos, age, owner in trees[tmask]:
            tree_map[int(pos)] = 1.0 if int(owner) == s else -1.0
            tree_age[int(pos)] = age
    enemy_pres = np.zeros(N)
    friend_pres = np.zeros(N)
    for q, mp in ((s, friend_pres), (o, enemy_pres)):
        for j in range(workers.shape[1]):
            if wmask[q, j]:
                mp[int(workers[q, j, 0])] += 1.0
    return tree_map, tree_age, enemy_pres, friend_pres


def _nearest(pos, targets):
    """Toroidal manhattan distance from pos to nearest of targets."""
    if len(targets) == 0:
        return 14.0, -1
    d = DIST[pos, np.asarray(targets, dtype=np.int64)]
    k = int(np.argmin(d))
    return float(d[k]), int(targets[k])


def _dirvec(pos, target):
    """Signed toroidal offset (dx, dy) from pos to target, /7.
    Returns (0, 0) when target is -1 (absent)."""
    if target < 0:
        return 0.0, 0.0
    dx = (int(_xs[target]) - int(_xs[pos]) + 7) % SIZE - 7
    dy = (int(_ys[target]) - int(_ys[pos]) + 7) % SIZE - 7
    return dx / 7.0, dy / 7.0


def featurize(carbon, cash, bases, workers, wmask, trees, tmask,
              player, step, seq=None):
    """Return (W, B): worker features (10, WORKER_DIM) for the player's
    own workers (rows where wmask[player] is False are zeroed) and base
    features (2, BASE_DIM) for the player's own bases.

    seq: optional dict of trajectory-state features for the base head,
    all computable live by an agent that remembers its own previous
    observation and its own previous actions (no hidden information):
      prev_n   -- own worker count at the previous step
      lost     -- own workers lost during the previous step
                  (prev_n + own recruits ordered last step - n now)
      since    -- steps since the player last ordered any recruit
    Defaults to zeros (used for step 0 / stateless calls)."""
    carbon = np.asarray(carbon, dtype=np.float64).ravel()
    cash = np.asarray(cash, dtype=np.float64)
    workers = np.asarray(workers, dtype=np.float64)
    trees = np.asarray(trees, dtype=np.float64)
    s, o = player, 1 - player
    g = _global_feats(carbon, cash, bases, workers, wmask, trees, tmask,
                      player, step)
    tree_map, tree_age, enemy_pres, friend_pres = _board_maps(
        workers, wmask, trees, tmask, player)

    own_bases = [int(b) for b in bases[s] if int(b) >= 0]
    opp_bases = [int(b) for b in bases[o] if int(b) >= 0]
    own_trees = [int(t[0]) for t in trees[tmask] if int(t[2]) == s]
    ene_trees = [int(t[0]) for t in trees[tmask] if int(t[2]) == o]
    ene_pos = [int(workers[o, j, 0]) for j in range(workers.shape[1])
               if wmask[o, j]]

    W = np.zeros((workers.shape[1], WORKER_DIM))
    for j in range(workers.shape[1]):
        if not wmask[s, j]:
            continue
        pos = int(workers[s, j, 0])
        cargo = float(workers[s, j, 1])
        wtype = int(workers[s, j, 2])
        d_ob, ob_pos = _nearest(pos, own_bases)
        d_eb, _ = _nearest(pos, opp_bases)
        d_ot, ot_pos = _nearest(pos, own_trees)
        d_et, et_pos = _nearest(pos, ene_trees)
        dir_ob = _dirvec(pos, ob_pos)
        dir_ot = _dirvec(pos, ot_pos)
        dir_et = _dirvec(pos, et_pos)
        et_age = tree_age[et_pos] / 50.0 if et_pos >= 0 else 0.0
        d_ew, _ = _nearest(pos, ene_pos)
        fri_pos = [int(workers[s, k, 0]) for k in range(workers.shape[1])
                   if wmask[s, k] and k != j]
        d_fw, _ = _nearest(pos, fri_pos)
        # enemy cargo mass near this worker (collision stakes)
        near_enemy_cargo = 0.0
        for k in range(workers.shape[1]):
            if wmask[o, k] and DIST[pos, int(workers[o, k, 0])] <= 2:
                near_enemy_cargo += float(workers[o, k, 1])
        own_trees_near = sum(1 for tp in own_trees if DIST[pos, tp] <= 3)
        local = np.concatenate([
            [1.0 if wtype == 0 else 0.0,
             1.0 if wtype == 1 else 0.0,
             np.clip(cargo / 500.0, 0, 2) / 2.0,
             1.0 if cargo > 0 else 0.0,
             1.0 if cargo >= 150 else 0.0],
            np.clip(carbon[PATCH[pos]] / 100.0, 0, 1),
            tree_map[PATCH[pos]],
            enemy_pres[PATCH[pos]],
            friend_pres[PATCH3[pos]],
            [min(d_ob, 14) / 14.0, min(d_eb, 14) / 14.0,
             min(d_ot, 14) / 14.0, min(d_et, 14) / 14.0, et_age,
             1.0 if pos in own_bases else 0.0,
             1.0 if pos in opp_bases else 0.0,
             1.0 if tree_map[pos] > 0 else 0.0,
             1.0 if tree_map[pos] < 0 else 0.0,
             tree_age[pos] / 50.0,
             min(d_ew, 14) / 14.0, min(d_fw, 14) / 14.0,
             np.clip(near_enemy_cargo / 500.0, 0, 1),
             min(own_trees_near, 10) / 10.0,
             _xs[pos] / 14.0, _ys[pos] / 14.0,
             dir_ob[0], dir_ob[1], dir_ot[0], dir_ot[1],
             dir_et[0], dir_et[1]],
        ])
        W[j] = np.concatenate([g, local])
    if seq is None:
        seq = {"prev_n": 0.0, "lost": 0.0, "since": 0.0}
    own_pos = [int(workers[s, j, 0]) for j in range(workers.shape[1])
               if wmask[s, j]]
    B = np.zeros((2, BASE_DIM))
    for bi in range(2):
        bpos = int(bases[s, bi])
        if bpos < 0:
            continue
        d_ew, _ = _nearest(bpos, ene_pos)
        d_ot, _ = _nearest(bpos, own_trees)
        own_near = sum(1 for wp in own_pos if DIST[bpos, wp] <= 2)
        ene_near = sum(1 for ep in ene_pos if DIST[bpos, ep] <= 3)
        local = np.array([
            1.0 if bi == 1 else 0.0,
            1.0 if cash[s] >= 30 else 0.0,
            1.0 if cash[s] >= 60 else 0.0,
            (10 - int(np.sum(wmask[s]))) / 10.0,
            min(d_ew, 14) / 14.0,
            1.0 if step < 60 else 0.0,
            min(d_ot, 14) / 14.0,
            min(float(seq["prev_n"]), 10) / 10.0,
            min(float(seq["lost"]), 5) / 5.0,
            min(float(seq["since"]), 60) / 60.0,
            min(own_near, 5) / 5.0,
            min(ene_near, 5) / 5.0,
        ])
        B[bi] = np.concatenate([g, local])
    return W, B
