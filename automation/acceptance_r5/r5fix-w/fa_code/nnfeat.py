"""nnfeat -- shared NumPy feature extractor for the egocentric-CNN policy.

Used verbatim by the BC dataset builder (from replays) and by the
submission agent at play time, so train/serve features are identical.

Coordinates: index = row*15 + col, origin top-left, edges wrap.
Actions (verified on replays): UP row-1, DOWN row+1, LEFT col-1, RIGHT col+1.
Visibility (carbon.py): a collector sees its own cell; a planter sees its
4 neighbours (not its own cell). Everything else in `carbon` reads 0, so a
belief map is kept: last seen value, mirrored to never-seen symmetric cells,
and propagated for unseen cells with tree absorption and 3% regrowth.
"""

import numpy as np

S = 15
N = S * S
ROW = np.arange(N) // S
COL = np.arange(N) % S
MOVES = ["NONE", "UP", "RIGHT", "DOWN", "LEFT"]
BASE_ACTS = ["NONE", "RECCOLLECTOR", "RECPLANTER"]
DR = {"UP": -1, "DOWN": 1, "LEFT": 0, "RIGHT": 0, "NONE": 0}
DC = {"UP": 0, "DOWN": 0, "LEFT": -1, "RIGHT": 1, "NONE": 0}
C_PLANES = 14
N_SCAL = 18
PRIOR = 11.0
REGEN = 1.03
ABS = 0.0375
CAP = 100.0


def nb(i, a):
    r, c = divmod(i, S)
    return ((r + DR[a]) % S) * S + (c + DC[a]) % S


NB4 = np.array([[nb(i, a) for a in ("UP", "RIGHT", "DOWN", "LEFT")] for i in range(N)])
MIRROR = np.array([[r * S + c, r * S + (S - 1 - c), (S - 1 - r) * S + c, (S - 1 - r) * S + (S - 1 - c)]
                   for r in range(S) for c in range(S)])


def tdist(i, j):
    dr = abs(ROW[i] - ROW[j]); dc = abs(COL[i] - COL[j])
    return np.minimum(dr, S - dr) + np.minimum(dc, S - dc)


class Tracker:
    """Per-player belief + bookkeeping. Call update(obs) once per step."""

    def __init__(self, me):
        self.me = me
        self.bel = np.full(N, PRIOR, dtype=np.float64)
        self.seen = np.full(N, -1, dtype=np.int64)
        self.prev_enemy = {}
        self.step = 0

    def update(self, obs):
        me = self.me
        step = obs["step"]
        players = obs["players"]
        carbon = np.asarray(obs["carbon"], dtype=np.float64)
        trees = obs.get("trees") or {}
        # propagate unseen cells one step: absorption by adjacent trees, enemy collectors staying, regen
        if step > 0:
            adj = np.zeros(N)
            for tid, tv in self._tree_items(players):
                for k in NB4[tv[0]]:
                    adj[k] += 1
            self.bel *= np.clip(1.0 - ABS * adj, 0.0, 1.0)
            for wid, w in players[1 - me][2].items():
                if w[2] == "COLLECTOR" and self.prev_enemy.get(wid) == w[0]:
                    self.bel[w[0]] *= 0.7
            self.bel = np.minimum(self.bel * REGEN, CAP)
        vis = []
        for wid, w in players[me][2].items():
            if w[2] == "COLLECTOR":
                vis.append(w[0])
            else:
                vis.extend(NB4[w[0]].tolist())
        for v in vis:
            val = carbon[v]
            self.bel[v] = val
            self.seen[v] = step
            for m in MIRROR[v]:
                if self.seen[m] < 0:
                    self.bel[m] = val
        # cells under trees / bases hold no collectable carbon
        for tid, tv in self._tree_items(players):
            self.bel[tv[0]] = 0.0
        self.prev_enemy = {wid: w[0] for wid, w in players[1 - me][2].items()}
        self.step = step

    @staticmethod
    def _tree_items(players):
        for p in (0, 1):
            for tid, tv in players[p][3].items():
                yield tid, tv

    # ------------------------------------------------------------------
    def global_planes(self, obs):
        """Returns (G [C_PLANES, N] float32, scal [N_SCAL] float32, info dict)."""
        me = self.me
        en = 1 - me
        players = obs["players"]
        step = obs["step"]
        G = np.zeros((C_PLANES, N), dtype=np.float32)
        G[0] = self.bel / 100.0
        age = np.where(self.seen < 0, 50, np.minimum(step - self.seen, 50))
        G[1] = age / 50.0
        nC = [0, 0]; nP = [0, 0]
        for p, (pc, pcarry, pp) in ((me, (2, 3, 4)), (en, (5, 6, 7))):
            for wid, w in players[p][2].items():
                if w[2] == "COLLECTOR":
                    G[pc, w[0]] = 1.0
                    G[pcarry, w[0]] = min(w[1], 400.0) / 200.0
                    nC[p] += 1
                else:
                    G[pp, w[0]] = 1.0
                    nP[p] += 1
        my_bases = list(players[me][1].values())
        en_bases = list(players[en][1].values())
        for b in my_bases:
            G[8, b] = 1.0
        for b in en_bases:
            G[9, b] = 1.0
        nT = [0, 0]
        for p, ch in ((me, 10), (en, 11)):
            for tid, tv in players[p][3].items():
                G[ch, tv[0]] = min(tv[1], 50) / 50.0
                nT[p] += 1
        if my_bases:
            d = np.min(np.stack([tdist(np.arange(N), b) for b in my_bases]), axis=0)
            G[12] = d / 14.0
        if en_bases:
            d = np.min(np.stack([tdist(np.arange(N), b) for b in en_bases]), axis=0)
            G[13] = d / 14.0
        cash = players[me][0]; ecash = players[en][0]
        tree_price = 20.0 + 5.0 * (1.235 ** (nT[0] + nT[1]))
        scal = np.zeros(N_SCAL, dtype=np.float32)
        scal[0] = step / 300.0
        scal[1] = (300 - step) / 300.0
        scal[2] = min(cash, 3000.0) / 1000.0
        scal[3] = min(ecash, 3000.0) / 1000.0
        scal[4] = np.clip((cash - ecash) / 3000.0, -3, 3)
        scal[5] = nC[me] / 10.0
        scal[6] = nP[me] / 10.0
        scal[7] = nC[en] / 10.0
        scal[8] = nP[en] / 10.0
        scal[9] = nT[me] / 10.0
        scal[10] = nT[en] / 10.0
        scal[11] = min(tree_price, 200.0) / 100.0
        scal[12] = 1.0 if cash >= 30 else 0.0
        scal[13] = 1.0 if cash >= tree_price else 0.0
        scal[14] = 1.0 if (300 - step) <= 20 else 0.0
        # 15: unit type planter, 16: own cargo, 17: is base  (filled per unit)
        return G, scal, {"cash": cash, "tree_price": tree_price}


def ego(G, center):
    """Roll [C, N] planes so `center` sits at (7,7). Returns [C, 15, 15]."""
    r, c = divmod(center, S)
    g = G.reshape(G.shape[0], S, S)
    return np.roll(np.roll(g, 7 - r, axis=1), 7 - c, axis=2)


def unit_inputs(G, scal, center, kind, cargo):
    """kind: 0 collector, 1 planter, 2 base."""
    x = ego(G, center)
    s = scal.copy()
    s[15] = 1.0 if kind == 1 else 0.0
    s[16] = min(cargo, 400.0) / 200.0
    s[17] = 1.0 if kind == 2 else 0.0
    return x, s
