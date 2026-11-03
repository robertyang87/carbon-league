"""belief -- four-fold-symmetric belief carbon map for the NN line
(stage 2), plus belief-augmented featurization (featurize2).

The belief reconstruction is juzi's ObservationParser idea (the same
re-implementation policy_v0_5.BeliefState uses), re-expressed over the
canonical arrays of rl_common / shadow_alpha._canonical so that the
SAME code path serves BC data (stored canonical arrays), PPO rollouts
(canonical_from_sim) and the live policy (obs -> canonical):

  * belief starts at -1 (unknown); the four base cells start at 0.
  * own collectors reveal their cell, own planters their 4-neighbours;
    a revealed value is mirrored into the other three symmetric cells
    (map generation is four-fold symmetric) only where still unknown.
  * enemy collectors that stayed in place: their cell's belief decays
    x0.7 (they collected 30%), and a positive cargo increment reveals
    the cell value exactly: belief = inc / 0.3 * 0.7.
  * tree cells: belief = 0 while the tree lives (age < 50), 30/1.03
    for a freshly withered tree; 4-neighbours of a tree with age > 1
    decay x(1 - 0.0375) per step (tree absorption) when no collector
    stands on the neighbour cell.
  * every cell not refreshed this step regrows x1.03, capped at 100.

ONE documented deviation from juzi / v0.5: enemy collectors are
tracked by POSITION (pos -> cargo between consecutive steps of the
same game) instead of by worker id, because canonical arrays do not
carry opponent ids. Two enemy collectors swapping cells between steps
would be misread as both having stayed; the affected belief cells are
few and the error washes out on the next direct observation.

featurize2 = bc_features.featurize + belief channels:
  worker rows append (32,):
    belief 5x5 patch (unknown cells -> -0.25, known -> belief/100)
    belief at the worker's own cell (same encoding)
    distance/direction to the richest known belief cell (argmax)
    belief globals (3,): known fraction, sum(known belief)/3000,
      7th-largest belief value / 100 (juzi's theta)
  base rows append (3,): the same belief globals.
WORKER_DIM2 = 156, BASE_DIM2 = 28.
"""

import numpy as np

from bc_features import (BASE_DIM, DIST, PATCH, WORKER_DIM, _dirvec,
                         _idx, _xs, _ys, featurize)

SIZE = 15
N = SIZE * SIZE

# 4-neighbours in flat index space (UP, RIGHT, DOWN, LEFT as (dx, dy))
_NBR4 = np.zeros((N, 4), dtype=np.int64)
_MIRROR = np.zeros((N, 4), dtype=np.int64)
for _i in range(N):
    _x, _y = int(_xs[_i]), int(_ys[_i])
    for _k, (_dx, _dy) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        _NBR4[_i, _k] = _idx(_x + _dx, _y + _dy)
    _MIRROR[_i] = [_idx(_x, _y), _idx(14 - _x, _y), _idx(_x, 14 - _y),
                   _idx(14 - _x, 14 - _y)]

BASE_CELLS = frozenset(int(_idx(x, y)) for x, y in
                       ((4, 4), (4, 10), (10, 4), (10, 10)))

UNKNOWN_ENC = -0.25

WORKER_DIM2 = WORKER_DIM + 32
BASE_DIM2 = BASE_DIM + 3


class BeliefTracker:
    """Stateful belief map for ONE player in ONE game."""

    def __init__(self):
        self.belief = np.full(N, -1.0)
        for b in BASE_CELLS:
            self.belief[b] = 0.0
        self.prev_enemy_collectors = {}   # pos -> cargo
        self.last_step = None

    def reset(self):
        self.__init__()

    def update(self, carbon, workers, wmask, trees, tmask, player,
               step):
        """Advance the belief with one canonical observation (the acting
        player's view at `step`). Must be called once per step, in
        order, before featurize2 for that step."""
        if self.last_step is not None and step != self.last_step + 1:
            # sequence break (new game sliced in / missing step): the
            # stay-inference would be garbage; restart conservatively.
            self.prev_enemy_collectors = {}
        belief = self.belief
        updated = np.zeros(N, dtype=bool)
        carbon = np.asarray(carbon, dtype=np.float64)

        # --- own observations (+ symmetric mirrors into unknown) ---
        for j in range(workers.shape[1]):
            if not wmask[player, j]:
                continue
            pos = int(workers[player, j, 0])
            if int(workers[player, j, 2]) == 0:      # collector
                cells = (pos,)
            else:                                    # planter: 4-nbrs
                cells = tuple(int(c) for c in _NBR4[pos])
            for cell in cells:
                v = float(carbon[cell])
                belief[cell] = v
                updated[cell] = True
                for m in _MIRROR[cell]:
                    m = int(m)
                    if belief[m] < 0:
                        belief[m] = v
                        updated[m] = True

        # --- enemy collectors that stayed put ---
        opp = 1 - player
        new_enemy = {}
        for j in range(workers.shape[1]):
            if not wmask[opp, j] or int(workers[opp, j, 2]) != 0:
                continue
            pos = int(workers[opp, j, 0])
            cargo = float(workers[opp, j, 1])
            new_enemy[pos] = cargo
            prev = self.prev_enemy_collectors.get(pos)
            if prev is not None:
                if belief[pos] > 0:
                    belief[pos] *= 0.7
                inc = cargo - prev
                if inc > 0:
                    belief[pos] = inc / 0.3 * 0.7
                updated[pos] = True
        self.prev_enemy_collectors = new_enemy

        # --- trees: cell value + absorption decay on neighbours ---
        if np.any(tmask):
            collector_cells = set()
            for q in range(2):
                for j in range(workers.shape[1]):
                    if wmask[q, j] and int(workers[q, j, 2]) == 0:
                        collector_cells.add(int(workers[q, j, 0]))
            for pos, age, _owner in trees[tmask]:
                pos = int(pos)
                age = float(age)
                belief[pos] = 0.0 if age < 50 else 30.0 / 1.03
                updated[pos] = True
                if age > 1:
                    for nb in _NBR4[pos]:
                        nb = int(nb)
                        if nb not in collector_cells and belief[nb] > 0:
                            belief[nb] *= (1.0 - 0.0375)
                        updated[nb] = True

        # --- regrowth on everything not refreshed ---
        mask = (belief > 0) & (~updated)
        belief[mask] = np.minimum(belief[mask] * 1.03, 100.0)
        self.last_step = step
        return belief


def belief_globals(belief):
    known = belief >= 0
    frac = float(np.mean(known))
    total = float(np.sum(belief[known])) if np.any(known) else 0.0
    if np.any(known):
        vals = np.sort(belief[known])[::-1]
        theta = float(vals[min(6, len(vals) - 1)])
    else:
        theta = 0.0
    return frac, total / 3000.0, theta / 100.0


def featurize2(carbon, cash, bases, workers, wmask, trees, tmask,
               player, step, belief, seq=None):
    """bc_features.featurize + belief channels. `belief` is the (225,)
    array AFTER BeliefTracker.update for this step."""
    W, B = featurize(carbon, cash, bases, workers, wmask, trees, tmask,
                     player, step, seq=seq)
    belief = np.asarray(belief, dtype=np.float64)
    known = belief >= 0
    enc = np.where(known, np.clip(belief, 0, 100) / 100.0, UNKNOWN_ENC)
    gfrac, gsum, gtheta = belief_globals(belief)
    globals3 = np.array([gfrac, gsum, gtheta])
    if np.any(known):
        rich = int(np.argmax(np.where(known, belief, -1.0)))
    else:
        rich = -1
    W2 = np.zeros((W.shape[0], WORKER_DIM2))
    W2[:, :WORKER_DIM] = W
    for j in range(W.shape[0]):
        if not wmask[player, j]:
            continue
        pos = int(workers[player, j, 0])
        d_rich = float(DIST[pos, rich]) if rich >= 0 else 14.0
        dir_rich = _dirvec(pos, rich)
        W2[j, WORKER_DIM:] = np.concatenate([
            enc[PATCH[pos]],
            [enc[pos], min(d_rich, 14) / 14.0, dir_rich[0], dir_rich[1]],
            globals3,
        ])
    B2 = np.zeros((B.shape[0], BASE_DIM2))
    B2[:, :BASE_DIM] = B
    for bi in range(B.shape[0]):
        if int(bases[player, bi]) >= 0:
            B2[bi, BASE_DIM:] = globals3
    return W2, B2
