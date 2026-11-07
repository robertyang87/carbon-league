"""cnet2_feat -- full-board, seat-canonical features for the cnet2 policy.

Shared verbatim by the dataset builder (replays) and the NumPy agent.

Canonical frame: seat 1 is mirrored left-right (col -> 14-col), so own bases
always sit on the main diagonal (4,4),(10,10) and enemy bases on (4,10),(10,4).
In that frame LEFT/RIGHT are swapped for seat 1; UP/DOWN unchanged.
Moves: 0 NONE, 1 UP (row-1), 2 RIGHT (col+1), 3 DOWN (row+1), 4 LEFT (col-1).
Base actions: 0 NONE, 1 RECCOLLECTOR, 2 RECPLANTER.

Belief over carbon (raw frame):
  * est is advanced one transition with the known rules: 30% collection under
    a staying collector (exact from the collector's cargo gain when no toll
    tree is near), 3.75% absorption per adjacent old tree, no regen next to
    trees / under staying workers, withered trees leave 30, 3% regen
    elsewhere (cap 100).
  * visible cells overwrite est;
  * never-seen cells take est from the most recently seen of their 4 mirror
    cells (maps are 4-way mirror symmetric at generation).

All float outputs are quantised (planes to uint8/255, the rest via float16)
so that train and serve inputs are bit-identical.
"""

import math

import numpy as np

S = 15
N = S * S
P = 33            # map planes
NS = 20           # scalars
T = 24            # tokens: own workers 0-9, own bases 10-11, enemy workers 12-21, enemy bases 22-23
OWN = 12
H = 8             # history rows per token
MF = 15           # per-move features
UF = 9            # per-unit decoder features
MOVES = ["NONE", "UP", "RIGHT", "DOWN", "LEFT"]
BASE_ACTS = ["NONE", "RECCOLLECTOR", "RECPLANTER"]
TEACHERS = ["claude-AlphaCarbon-Master", "1C-Zero-v9", "1C-Zero-v8-preview", "1C-Zero-v7",
            "RyanX-AIPowercore-V11 Ultra", "RyanX-AIPowercore-Ultra v13-test", "other"]
N_TEACH = 8

GRID = np.arange(N)
ROW, COL = GRID // S, GRID % S
DEST = np.stack([GRID, ((ROW - 1) % S) * S + COL, ROW * S + (COL + 1) % S,
                 ((ROW + 1) % S) * S + COL, ROW * S + (COL - 1) % S], 1)        # [N,5]
NB8 = np.stack([((ROW + dr) % S) * S + (COL + dc) % S
                for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc], 1)     # [N,8]
CANON = ROW * S + (S - 1 - COL)               # seat-1 mapping (involution)
MIRROR = np.stack([GRID, ROW * S + (S - 1 - COL), (S - 1 - ROW) * S + COL,
                   (S - 1 - ROW) * S + (S - 1 - COL)], 1)
_dr = np.abs(ROW[:, None] - ROW[None, :]); _dc = np.abs(COL[:, None] - COL[None, :])
DIST = np.minimum(_dr, S - _dr) + np.minimum(_dc, S - _dc)                    # [N,N]
del _dr, _dc
CANON_MOVE = np.array([0, 1, 4, 3, 2])
REGEN, ABS, COLLECT, CAP = 1.03, 0.0375, 0.3, 100.0
PRIOR = 11.0
LOGC = 6.5


def q8(x):
    return np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def f16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def lc(x):
    return min(math.log1p(max(float(x), 0.0)) / LOGC, 1.5)


def move_between(a, b):
    """move index taking cell a to cell b (canonical or raw, same geometry); 5 if not adjacent."""
    hit = np.nonzero(DEST[a] == b)[0]
    return int(hit[0]) if len(hit) else 5


class Feat:
    def __init__(self, seat):
        self.seat = seat
        self.step = -1
        self.est = np.full(N, PRIOR)
        self.last = np.zeros(N)
        self.last_step = np.full(N, -1, np.int64)
        self.prev_w = None
        self.prev_trees = None
        self.hist = {}          # uid -> list of (canon cell, cargo, move) newest last

    def cpos(self, i):
        return int(CANON[i]) if self.seat else int(i)

    # ------------------------------------------------------------ belief
    def _advance(self, workers, trees):
        est = self.est
        pw, pt = self.prev_w, self.prev_trees
        stay_coll = {}
        occupied_stay = set()
        for uid, (side, pos, cargo, isc) in workers.items():
            o = pw.get(uid)
            if o is not None and o[1] == pos:
                occupied_stay.add(pos)
                if isc:
                    stay_coll[pos] = (side, o[2], cargo)
        tree_cells = set(trees)
        withered = [p for p in pt if p not in trees]
        old_trees = [p for p in trees if p in pt]
        before = est.copy()
        adj_any = set()
        for p in trees:
            adj_any.update(int(n) for n in DEST[p, 1:])
        # absorption by old trees on 4-neighbours not holding a staying collector / tree
        red = np.zeros(N)
        for p in old_trees:
            for n in DEST[p, 1:]:
                n = int(n)
                if n in tree_cells or n in stay_coll:
                    continue
                red[n] += before[n] * ABS
        est -= red
        # collection
        for pos, (side, c0, c1) in stay_coll.items():
            if pos in tree_cells:
                continue
            toll = any(int(n) in trees and trees[int(n)][0] != side for n in NB8[pos]) or \
                (pos in trees and trees[pos][0] != side)
            d = c1 - c0
            if not toll and d >= 0:
                est[pos] = d / COLLECT * (1 - COLLECT)
            else:
                est[pos] = before[pos] * (1 - COLLECT)
        # regen
        mask = np.ones(N, bool)
        for p in tree_cells:
            mask[p] = False
        for p in withered:
            mask[p] = False
        for p in adj_any:
            mask[p] = False
        for p in occupied_stay:
            mask[p] = False
        est[mask] = np.minimum(est[mask] * REGEN, CAP)
        for p in withered:
            est[p] = 30.0
        for p in tree_cells:
            est[p] = 0.0
        np.clip(est, 0, CAP, out=est)

    # ------------------------------------------------------------ main
    def observe(self, obs):
        step = int(obs["step"])
        if step == 0 or step <= self.step:
            self.__init__(self.seat)
        me = self.seat
        en = 1 - me
        players = obs["players"]
        carbon = np.asarray(obs["carbon"], np.float64)
        workers = {}
        for p in (0, 1):
            for uid, w in players[p][2].items():
                workers[uid] = (p, int(w[0]), float(w[1]), w[2] == "COLLECTOR")
        trees = {}
        for p in (0, 1):
            for tid, tv in players[p][3].items():
                trees[int(tv[0])] = (p, int(tv[1]))
        bases = {p: [int(b) for b in players[p][1].values()] for p in (0, 1)}
        if self.prev_w is not None:
            self._advance(workers, trees)
        else:
            self.est[:] = min(PRIOR * REGEN ** step, CAP)
        vis = np.zeros(N, bool)
        for uid, (p, pos, cargo, isc) in workers.items():
            if p == me:
                if isc:
                    vis[pos] = True
                else:
                    vis[DEST[pos, 1:]] = True
        self.est[vis] = carbon[vis]
        self.last[vis] = carbon[vis]
        self.last_step[vis] = step
        # mirror fill
        mt = self.last_step[MIRROR]                      # [N,4]
        mt_other = mt.copy(); mt_other[:, 0] = -2
        same = MIRROR == GRID[:, None]
        mt_other[same] = -2
        bo = mt_other.argmax(1)
        msrc = MIRROR[GRID, bo]
        mtime = self.last_step[msrc]
        has_m = (mt_other.max(1) >= 0)
        never = self.last_step < 0
        fill = never & has_m
        self.est[fill] = self.est[msrc[fill]]
        unk = never & ~has_m
        self.est[unk] = min(PRIOR * REGEN ** step, CAP)
        for p in (0, 1):
            for b in bases[p]:
                self.est[b] = 0.0
        for p in trees:
            self.est[p] = 0.0
        mirror_est = np.where(has_m, self.est[msrc], 0.0)

        left = 299 - step
        G = np.zeros((P, N), np.float32)
        G[0] = self.est / 100.0
        G[1] = self.last / 100.0
        G[2] = ~never
        G[3] = vis
        G[4] = np.where(never, 1.0, np.minimum(step - self.last_step, 100) / 100.0)
        G[5] = mirror_est / 100.0
        G[6] = np.where(has_m, np.minimum(step - mtime, 100) / 100.0, 1.0)
        G[7] = fill
        en_min = np.full(N, np.inf)      # min enemy collector cargo that can be on cell next step
        en_max = np.full(N, -1.0)        # max enemy collector cargo reaching cell
        en_pl = np.zeros(N)
        thr = np.zeros(N)
        own_reach = np.zeros(N)
        nC = [0, 0]; nP = [0, 0]; tot = [0.0, 0.0]
        for uid, (p, pos, cargo, isc) in workers.items():
            if p == me:
                G[8 if isc else 9, pos] += 1
                G[10, pos] = lc(cargo)
                own_reach[DEST[pos]] += 1
            else:
                G[11 if isc else 12, pos] += 1
                G[13, pos] = lc(cargo)
                d = DEST[pos]
                thr[d] += 1
                if isc:
                    en_min[d] = np.minimum(en_min[d], cargo)
                    en_max[d] = np.maximum(en_max[d], cargo)
                else:
                    en_pl[d] = 1
            if isc:
                nC[p] += 1
            else:
                nP[p] += 1
            tot[p] += cargo
        for b in bases[me]:
            G[14, b] = 1
        for b in bases[en]:
            G[15, b] = 1
        own_adj = np.zeros(N); en_adj = np.zeros(N); en_toll = np.zeros(N); own_toll = np.zeros(N)
        nT = [0, 0]
        for pos, (p, age) in trees.items():
            G[16 if p == me else 17, pos] = 1
            G[18, pos] = min(age, 50) / 50.0
            G[19, pos] = age >= 45
            nT[p] += 1
            if p == me:
                own_adj[DEST[pos, 1:]] += 1; own_toll[NB8[pos]] += 1
            else:
                en_adj[DEST[pos, 1:]] += 1; en_toll[NB8[pos]] += 1
        dob = DIST[:, bases[me]].min(1) if bases[me] else np.full(N, 14)
        deb = DIST[:, bases[en]].min(1) if bases[en] else np.full(N, 14)
        G[20] = dob / 14.0
        G[21] = deb / 14.0
        G[22] = np.clip((left - dob) / 300.0, 0, 1)
        G[23] = np.clip((left - dob) / 20.0, 0, 1)
        G[24] = np.minimum(thr, 3) / 3.0
        G[25] = np.isfinite(en_min)
        G[26] = np.where(np.isfinite(en_min), np.minimum(np.log1p(np.where(np.isfinite(en_min), en_min, 0)) / LOGC, 1), 0)
        G[27] = en_pl
        G[28] = np.minimum(own_reach, 3) / 3.0
        G[29] = own_adj / 4.0
        G[30] = en_adj / 4.0
        G[31] = np.minimum(en_toll, 4) / 4.0
        G[32] = np.minimum(own_toll, 4) / 4.0
        order = CANON if me else GRID
        maps = q8(G[:, order])

        cash = float(players[me][0]); ecash = float(players[en][0])
        alive = sum(1 for (p, a) in trees.values() if a < 50)
        price = 20.0 + 5.0 * 1.235 ** alive if nT[me] > 0 else 20.0
        slots = 10 - nC[me] - nP[me]
        sc = np.zeros(NS, np.float32)
        sc[0] = left / 300.0; sc[1] = step / 300.0
        sc[2] = math.log1p(max(cash, 0)) / 9; sc[3] = math.log1p(max(ecash, 0)) / 9
        sc[4] = math.tanh((cash - ecash) / 2000.0)
        sc[5] = nC[me] / 10; sc[6] = nP[me] / 10; sc[7] = nC[en] / 10; sc[8] = nP[en] / 10
        sc[9] = nT[me] / 20; sc[10] = nT[en] / 20
        sc[11] = min(price, 200.0) / 100.0
        sc[12] = cash >= 30; sc[13] = cash >= price; sc[14] = slots / 10
        sc[15] = left <= 30
        sc[16] = math.log1p(tot[me]) / 8; sc[17] = math.log1p(tot[en]) / 8
        sc[18] = alive / 30.0; sc[19] = min(cash / 2000.0, 3.0)

        # ---- history (canonical cells)
        for uid, (p, pos, cargo, isc) in workers.items():
            cp = self.cpos(pos)
            h = self.hist.get(uid)
            if h is None:
                h = []
                mv = 5
            else:
                mv = move_between(h[-1][0], cp)
            h.append((cp, cargo, mv))
            self.hist[uid] = h[-H:]
        for uid in [u for u in self.hist if u not in workers]:
            del self.hist[uid]

        # ---- tokens
        tok_pos = np.zeros(T, np.int64); tok_kind = np.zeros(T, np.int64)
        tok_mask = np.zeros(T, bool); tok_cargo = np.zeros(T, np.float32)
        hc = np.full((T, H), -1, np.int64); hg = np.zeros((T, H), np.float32); hm = np.full((T, H), 5, np.int64)
        own_uid = [None] * OWN

        def own_key(uid):
            p, pos, cargo, isc = workers[uid]
            return (0 if isc else 1, -cargo if isc else 0, self.cpos(pos), uid)
        mine = sorted([u for u in players[me][2]], key=own_key)
        theirs = sorted([u for u in players[en][2]], key=lambda u: (self.cpos(workers[u][1]), u))
        for start, lst, kc, kp in ((0, mine, 0, 1), (12, theirs, 3, 4)):
            for j, uid in enumerate(lst[:10]):
                s = start + j
                p, pos, cargo, isc = workers[uid]
                tok_pos[s] = self.cpos(pos); tok_kind[s] = kc if isc else kp
                tok_mask[s] = True; tok_cargo[s] = cargo
                h = self.hist[uid]
                for r, (c_, g_, m_) in enumerate(h):
                    rr = H - len(h) + r
                    hc[s, rr] = c_; hg[s, rr] = g_; hm[s, rr] = m_
                if start == 0:
                    own_uid[s] = uid
        for start, p, kb in ((10, me, 2), (22, en, 5)):
            for j, (bid, b) in enumerate(list(players[p][1].items())[:2]):
                s = start + j
                tok_pos[s] = self.cpos(int(b)); tok_kind[s] = kb; tok_mask[s] = True
                hc[s, -1] = tok_pos[s]
                if start == 10:
                    own_uid[s] = bid

        # ---- canonical helper arrays for the decoder prefix features
        aux = {
            "est": f16(G[0][order]), "en_min": en_min[order], "en_max": en_max[order],
            "en_pl": en_pl[order], "thr": thr[order], "dob": dob[order],
            "own_base": G[14][order] > 0, "en_base": G[15][order] > 0,
            "own_tree": G[16][order] > 0, "en_tree": G[17][order] > 0,
            "toll": en_toll[order], "own_adj": own_adj[order],
        }
        x = dict(maps=maps, scal=f16(sc), tok_pos=tok_pos, tok_kind=tok_kind, tok_mask=tok_mask,
                 tok_cargo=f16(tok_cargo), hist_cell=hc, hist_cargo=f16(hg), hist_move=hm,
                 cash=cash, slots=slots, price=price, left=left)
        self.prev_w = workers
        self.prev_trees = trees
        self.step = step
        self.own_uid = own_uid
        self.aux = aux
        return x

    # ------------------------------------------------------------ actions
    def labels(self, act):
        """teacher action dict -> canonical labels [OWN] (-1 = no unit)."""
        y = np.full(OWN, -1, np.int64)
        for s, uid in enumerate(self.own_uid):
            if uid is None:
                continue
            a = act.get(uid, "NONE") if act else "NONE"
            if s < 10:
                m = MOVES.index(a) if a in MOVES else 0
                y[s] = CANON_MOVE[m] if self.seat else m
            else:
                y[s] = BASE_ACTS.index(a) if a in BASE_ACTS else 0
        return y

    def to_actions(self, y):
        out = {}
        for s, uid in enumerate(self.own_uid):
            if uid is None or y[s] <= 0:
                continue
            a = int(y[s])
            if s < 10:
                out[uid] = MOVES[CANON_MOVE[a] if self.seat else a]
            else:
                out[uid] = BASE_ACTS[a]
        return out


class Prefix:
    """Autoregressive per-unit features. Units are decoded in slot order 0..11."""

    def __init__(self, x, aux):
        self.x, self.a = x, aux
        self.reserved = np.zeros(N)
        self.cash = x["cash"]
        self.slots = x["slots"]
        self.plants = 0
        self.rec_pl = 0
        self.undecided = np.zeros(N)
        for s in range(10):
            if x["tok_mask"][s]:
                self.undecided[x["tok_pos"][s]] += 1

    def _dest(self, cell, cargo, isc, stay):
        a = self.a
        f = np.zeros(MF, np.float32)
        f[0] = min(self.reserved[cell], 2) / 2
        if isc:
            f[1] = a["en_min"][cell] <= cargo
            mx = a["en_max"][cell]
            f[2] = min(math.log1p(mx) / LOGC, 1.5) if mx > cargo else 0.0
            f[3] = a["en_pl"][cell]
        else:
            f[1] = np.isfinite(a["en_min"][cell]) or a["en_pl"][cell] > 0
        f[4] = min(a["thr"][cell], 4) / 4
        f[5] = a["own_base"][cell]; f[6] = a["en_base"][cell]
        f[7] = a["est"][cell]
        f[8] = a["own_tree"][cell]; f[9] = a["en_tree"][cell]
        f[10] = a["dob"][cell] / 14.0
        f[11] = stay
        f[12] = min(self.undecided[cell], 2) / 2
        f[13] = min(a["toll"][cell], 4) / 4
        f[14] = a["own_adj"][cell] / 4
        return f

    def feats(self, s):
        """-> uf [UF], mf [5, MF], legal [5] for slot s (assumes tok_mask[s])."""
        x = self.x
        pos = int(x["tok_pos"][s])
        uf = np.zeros(UF, np.float32)
        mf = np.zeros((5, MF), np.float32)
        legal = np.ones(5, bool)
        uf[6] = math.log1p(max(self.cash, 0)) / 9
        uf[7] = self.slots / 10
        uf[8] = min(self.plants, 3) / 3
        if s < 10:
            isc = x["tok_kind"][s] == 0
            cargo = float(x["tok_cargo"][s])
            self.undecided[pos] -= 1
            uf[0] = isc; uf[1] = not isc
            uf[3] = lc(cargo)
            uf[4] = self.a["dob"][pos] / 14.0
            uf[5] = (x["left"] - self.a["dob"][pos]) <= 1
            for m in range(5):
                mf[m] = self._dest(int(DEST[pos, m]), cargo, isc, m == 0)
        else:
            uf[2] = 1
            ok = self.cash >= 30 and self.slots > 0
            legal[1] = legal[2] = ok
            legal[3] = legal[4] = False
            mf[1] = self._dest(pos, 0.0, True, True)
            mf[2] = self._dest(pos, 0.0, False, True)
        return f16(uf), f16(mf), legal

    def advance(self, s, a):
        x = self.x
        pos = int(x["tok_pos"][s])
        if s < 10:
            d = int(DEST[pos, a])
            self.reserved[d] += 1
            if a == 0 and x["tok_kind"][s] == 1:
                if not self.a["own_tree"][pos] and not self.a["own_base"][pos] and not self.a["en_base"][pos]:
                    self.plants += 1
        elif a > 0:
            self.reserved[pos] += 1
            self.cash -= 30
            self.slots -= 1


# --------------------------------------------------------------- symmetries
def _sym_tables():
    """Cell permutations of the canonical frame that keep own and enemy base sets:
    identity, rot180, transpose, anti-transpose. Returns (cellmap [4,N], movemap [4,5])."""
    r, c = ROW, COL
    maps = [GRID, (S - 1 - r) * S + (S - 1 - c), c * S + r, (S - 1 - c) * S + (S - 1 - r)]
    cm = np.stack(maps)
    mm = np.zeros((4, 5), np.int64)
    for g in range(4):
        for m in range(5):
            src = 7 * S + 7
            dst = DEST[src, m]
            mm[g, m] = move_between(cm[g][src], cm[g][dst])
    return cm, mm


SYM_CELL, SYM_MOVE = _sym_tables()
