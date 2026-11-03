"""Carbon League strategy (league: https://132.232.176.153/).

Lineage / attribution (declared per league rules, not original):
  * Base architecture and mechanisms are adapted from the PUBLIC strategy
    claude-opus-v18 by author "Claude", published at
    https://132.232.176.153/source/ce2614d195152e25e998a7b072554290
    (mirror-symmetric carbon belief tracking with class inference; whole-map
    stay/move dynamic programme for collectors; planter plant/seize/guard
    tasks; adaptive enemy-base camping; joint own-unit conflict resolution).
  * This file is a retuned derivative: parameters re-optimised in local
    self-play against the unmodified public source (banking cadence rho/rho_early,
    planting thresholds plant_min/plant_ratio/cell_loss_w, earlier blockade
    camp_from, extra early planter early_planters, deposit bonus mu, kill
    bonuses), plus a defensive fallback agent for unexpected states.
  * Studied for background only (no code taken): juzi final reference
    (/source/a938df4254388b8df8ee04f6e6e0aae5) and the environment sources of
    https://github.com/moliqingwa/carbon_challenge.
"""

import os
import numpy as np


SIZE = 15
N = SIZE * SIZE
ACT = ["UP", "RIGHT", "DOWN", "LEFT"]
INF = float("inf")

PARAMS = dict(
    H=12, S=5, gamma=0.92,
    rho=0.035, rho_early=0.09, mu=0.05, mu_early=0.3, term_k=1.0,
    tie_cost=12.0, soft_w=1.0,
    kill_c=40.0, kill_p=45.0,
    info_c=2.0, info_p=3.0,
    open="CP", planters=1, planters_late=1, planter_from=4,
    plant_ratio=1.2, plant_min=6, hold_p=0.8, steal_w=0.6, cell_loss_w=0.3,
    seize_min=8.0, pdisc=0.85, pp_cost=10.0, pp_stand=28.0, lunge=1.0,
    guard_w=0.5, guard_min=20.0,
    recruit_until=40, endgame=14,
    campers=4, camp_from=6, camp_min=0, camp_div=99, camp_decay=80, eval_T=30, eff_hi=0.6, eff_lo=0.35, eff_rate=2.3, reprobe=0, kill_win=30, kill_n=3, lost_max=6, probe_n=0, probe_T=12, probe_kill=2, probe_win=20, camp_orbit=2, orbit_kill=3, orbit_win=12, r2_hi=0.75, r2_lo=0.45, r2_kill=0.6, siege_hold=10, carry_ok=100.0, def_always=0, gate=1, gate_min=60.0, gate_w=0.5, gate_cap=150.0, no_p_T=60, predict=1, siege_spawn=1, siege_T=3, siege_carry=150.0, camper_max=20.0, retire_stock=2400.0, max_units=10, retire_from=85, retire_keep=0, retire_min_left=50, early_planters=2, camp_lunge=0.0, lunge_floor=0.0, camp_end=4, camp_end_from=28, camp_planters=1, displace=1, avoid1=0.5, avoid2=0.15, sweep_w=0.3, sweep_base=10.0, sweep_max=80.0, camp_tie=3.0, camp_w=30.0, camp_keep=25.0, camp_until=0, base_def=20.0,
)


def _rc(i):
    return i // SIZE, i % SIZE


NB = []
for _i in range(N):
    _r, _c = _rc(_i)
    NB.append([((_r - 1) % SIZE) * SIZE + _c, _r * SIZE + (_c + 1) % SIZE,
               ((_r + 1) % SIZE) * SIZE + _c, _r * SIZE + (_c - 1) % SIZE])
NB8 = []
for _i in range(N):
    _r, _c = _rc(_i)
    NB8.append([((_r + a) % SIZE) * SIZE + (_c + b) % SIZE for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b])
DIST = []
for _i in range(N):
    _r, _c = _rc(_i)
    _row = []
    for _j in range(N):
        _r2, _c2 = _rc(_j)
        _dr = abs(_r - _r2); _dc = abs(_c - _c2)
        _row.append(min(_dr, SIZE - _dr) + min(_dc, SIZE - _dc))
    DIST.append(_row)
N1 = [[i] + NB[i] for i in range(N)]
N2 = [[j for j in range(N) if DIST[i][j] <= 2] for i in range(N)]
CLS = []
for _i in range(N):
    _r, _c = _rc(_i)
    CLS.append(min(_r, SIZE - 1 - _r) * 8 + min(_c, SIZE - 1 - _c))
CLS_MEMBERS = {}
for _i in range(N):
    CLS_MEMBERS.setdefault(CLS[_i], []).append(_i)
NBR = np.array(NB, dtype=np.int64)

REGEN = 1.03
CAP = 100.0
ABS = 0.0375
COLLECT = 0.3


def apply_ops(v, ops):
    for op in ops:
        k = op[0]
        if k == 0:
            if v > 0:
                for _ in range(op[1]):
                    v = round(min(v * REGEN, CAP), 3)
                    if v >= CAP:
                        break
        elif k == 1:
            v = round(max(v - v * COLLECT, 0.0), 3)
        elif k == 2:
            v = round(max(v - v * ABS * op[1], 0.0), 3)
        elif k == 3:
            v = min(v + op[1], CAP)
        elif k == 4:
            v = op[1]
    return v


class Belief:
    def __init__(self, me, bases_all):
        self.me = me
        self.val = [0.0] * N
        self.known = [False] * N
        self.hist = [[] for _ in range(N)]
        self.pure = [True] * N
        self.cls_v0 = {}
        self.base_cells = set(bases_all)
        for b in bases_all:
            self.known[b] = True
        self.cls_v0[CLS[bases_all[0]]] = 0
        self.prev_workers = {}
        self.prev_trees = {}
        self.nz_found = 0
        self.cls_seen = 0

    def _op(self, i, op):
        if self.known[i]:
            self.val[i] = apply_ops(self.val[i], (op,))
            if op[0] == 4:
                self.pure[i] = False
        else:
            h = self.hist[i]
            if op[0] == 4:
                self.known[i] = True
                self.val[i] = op[1]
                self.pure[i] = False
                self.hist[i] = []
            elif op[0] == 0 and h and h[-1][0] == 0:
                h[-1] = (0, h[-1][1] + op[1])
            else:
                h.append(op)

    def _learn_class(self, cls, v0):
        if cls in self.cls_v0:
            return
        self.cls_v0[cls] = v0
        self.cls_seen += 1
        if v0 > 0:
            self.nz_found += 1
        for m in CLS_MEMBERS[cls]:
            if not self.known[m]:
                self.val[m] = apply_ops(float(v0), self.hist[m])
                self.known[m] = True
                self.hist[m] = []

    def _infer_v0(self, i, v_obs, ops):
        if not self.pure[i]:
            return None
        if v_obs <= 0:
            return 0
        best = None; bd = 1e9
        for v0 in range(0, 51):
            d = abs(apply_ops(float(v0), ops) - v_obs)
            if d < bd - 1e-9:
                bd = d; best = v0
        if bd > 0.05 * max(1.0, v_obs):
            return None
        return best

    def observe(self, i, v):
        v = float(v)
        if self.known[i]:
            self.val[i] = v
            return
        cls = CLS[i]
        v0 = self._infer_v0(i, v, self.hist[i]) if cls not in self.cls_v0 else None
        self.known[i] = True
        self.val[i] = v
        self.hist[i] = []
        if v0 is not None:
            self._learn_class(cls, v0)

    def prior(self):
        rem = 63 - self.cls_seen
        if rem <= 0:
            return 0.0
        p = (28.5 - self.nz_found) / rem
        return max(0.0, min(1.0, p)) * 22.0

    def estimate(self):
        """Per-cell expected carbon (exact where known)."""
        pr = self.prior()
        out = [0.0] * N
        for i in range(N):
            if self.known[i]:
                out[i] = self.val[i]
            else:
                out[i] = apply_ops(pr, self.hist[i])
        return out

    def update(self, workers, trees, visible, carbon, first, my_dest):
        if not first:
            pw = self.prev_workers; pt = self.prev_trees
            me = self.me
            stay_cells = set(); stay_coll = {}
            for wid, w in workers.items():
                o = pw.get(wid)
                if o is None or o[0] == w[0]:
                    stay_cells.add(w[0])
                    if w[2] == 'C':
                        stay_coll[w[0]] = wid
            tree_cells = {t[0] for t in trees.values()}
            noregen = set()
            for tid, t in pt.items():
                if tid not in trees:
                    self._op(t[0], (4, 30.0))
                    noregen.add(t[0])
            dead = [(wid, w) for wid, w in pw.items() if wid not in workers]
            death_near = set()
            if dead:
                groups = {}
                mine_dead_cells = set()
                for wid, w in dead:
                    for z in N1[w[0]]:
                        death_near.add(z)
                    if w[3] == me:
                        z = my_dest.get(wid, w[0])
                        groups.setdefault(z, []).append(w)
                        mine_dead_cells.add(z)
                for wid, w in dead:
                    if w[3] == me:
                        continue
                    cands = N1[w[0]]
                    z = None
                    for q in cands:
                        if q in mine_dead_cells:
                            z = q; break
                    if z is None:
                        bestk = -1
                        for q in cands:
                            for wid2, w2 in workers.items():
                                if w2[0] == q and w2[2] == 'C' and w2[3] == me:
                                    o2 = pw.get(wid2)
                                    k = 1
                                    if o2 is not None and w[1] > 0 and w2[1] - o2[1] >= w[1] - 1e-6:
                                        k = 2
                                    if k > bestk:
                                        bestk = k; z = q
                    if z is None:
                        for q in cands:
                            if q in self.base_cells:
                                z = q; break
                    if z is None:
                        z = w[0]
                    groups.setdefault(z, []).append(w)
                for z, lst in groups.items():
                    noregen.add(z)
                    winner = False
                    for w2 in workers.values():
                        if w2[0] == z and w2[2] == 'C':
                            winner = True; break
                    if not winner and z not in tree_cells and z not in self.base_cells:
                        tot = sum(w[1] for w in lst)
                        if tot > 0:
                            self._op(z, (3, tot))
            for z, wid in stay_coll.items():
                if z in tree_cells or z in self.base_cells:
                    continue
                o = pw.get(wid)
                if o is None:
                    continue
                w = workers[wid]
                if not self.known[z] and w[3] != me and z not in death_near:
                    ntax = 0
                    for tid, t in trees.items():
                        if t[2] != w[3] and tid in pt and (t[0] == z or z in NB8[t[0]]):
                            ntax += 1
                    if ntax < 10:
                        h = w[1] / (1.0 - 0.1 * ntax) - o[1]
                        before = h / COLLECT
                        if before < 0.02:
                            before = 0.0
                        cls = CLS[z]
                        if cls not in self.cls_v0:
                            v0 = self._infer_v0(z, before, self.hist[z])
                            if v0 is not None:
                                self._learn_class(cls, v0)
                        if not self.known[z]:
                            self.known[z] = True
                            self.val[z] = before
                            self.hist[z] = []
                self._op(z, (1,))
            newborn = set()
            for tid, t in trees.items():
                if tid not in pt:
                    newborn.add(tid)
                    self._op(t[0], (4, 0.0))
            absn = {}
            adj_tree = set()
            for tid, t in trees.items():
                for n in NB[t[0]]:
                    adj_tree.add(n)
                    if tid in newborn or n in tree_cells or n in stay_coll:
                        continue
                    absn[n] = absn.get(n, 0) + 1
            for n, k in absn.items():
                self._op(n, (2, k))
            for i in range(N):
                if i in stay_cells or i in noregen or i in tree_cells or i in adj_tree:
                    continue
                self._op(i, (0, 1))
        for i in visible:
            self.observe(i, carbon[i])
        self.prev_workers = dict(workers)
        self.prev_trees = dict(trees)


class Bot:
    def __init__(self):
        self.P = dict(PARAMS)
        self.belief = None
        self.last_dest = {}
        self.stat = {}          # adaptive opponent statistics: key -> [stay, total]
        self.watch = []         # (enemy wid, pos, key) observed under threat last turn
        self.watch2 = []
        self.prev_pos = {}
        self.camp = {}
        self.camp_lost = 0
        self.camp_loss_t = 0
        self.ncamp_mid = 0
        self.en_deliv = 0.0
        self.ehist = {}
        self.myh_hist = []
        self.myn_hist = []
        self.orbit_on = False
        self.ecar = 0.0
        self.no_p_until = -1
        self.siege_cnt = 0
        self.probing = False
        self.probe_on = 0
        self.block_ok = False
        self.eval_bad = 0
        self.deliv_hist = []
        self.ec_hist = []
        self.camp_deaths = []
        self.camp_level = None
        self.eval_on = 0
        self.eval_d0 = 0.0
        self.eval_lost0 = 0
        self.eval_ec = 0.0

    # ------------------------------------------------------------------
    def flow(self, wnd):
        """Opponent deliveries per collector-turn relative to my own harvest per harvester-turn."""
        e = sum(self.deliv_hist[-wnd:]) / max(1.0, sum(self.ec_hist[-wnd:]))
        m = sum(self.myh_hist[-wnd:]) / max(1.0, sum(self.myn_hist[-wnd:]))
        return e / max(0.5, m)

    def pstay(self, key, prior_s, prior_n):
        s, n = self.stat.get(key, (0, 0))
        return (s + prior_s) / (n + prior_n)

    def act(self, obs, conf):
        P = self.P
        step = obs['step']
        me = obs['player']
        players = obs['players']
        episode = conf.get('episodeSteps', 300) if hasattr(conf, 'get') else 300
        R = episode - 1 - step          # moves left including this one
        workers = {}; trees = {}
        bases = [[], []]
        for pid in range(2):
            pl = players[pid]
            for bid, pos in pl[1].items():
                bases[pid].append((bid, pos))
            for wid, w in pl[2].items():
                workers[wid] = (w[0], w[1], 'C' if w[2] == 'COLLECTOR' else 'P', pid)
            for tid, t in pl[3].items():
                trees[tid] = (t[0], t[1], pid)
        cash = players[me][0]
        my_bases = [p for _, p in bases[me]]
        en_bases = [p for _, p in bases[1 - me]]
        base_cells = set(my_bases) | set(en_bases)
        visible = []
        for wid, w in workers.items():
            if w[3] == me:
                if w[2] == 'C':
                    visible.append(w[0])
                else:
                    visible.extend(NB[w[0]])
        if self.belief is None or step == 0:
            self.belief = Belief(me, my_bases + en_bases)
            self.stat = {}; self.watch = []; self.watch2 = []; self.last_dest = {}; self.camp = {}; self.camp_lost = 0; self.camp_loss_t = 0; self.ncamp_mid = 0; self.en_deliv = 0.0; self.ehist = {}; self.orbit_on = False; self.ecar = 0.0; self.no_p_until = -1; self.siege_cnt = 0; self.probing = False; self.probe_on = 0; self.block_ok = False; self.eval_bad = 0; self.deliv_hist = []; self.ec_hist = []; self.camp_deaths = []; self.camp_level = None; self.eval_on = 0; self.eval_d0 = 0.0; self.eval_lost0 = 0; self.eval_ec = 0.0
        bel = self.belief
        # --- learn opponent habits from last turn's watch list
        for wid, pos, key in self.watch:
            w = workers.get(wid)
            s = self.stat.setdefault(key, [0, 0])
            s[1] += 1
            if w is None or w[0] == pos:
                s[0] += 1
        self.watch = []
        for wid, plist, key in self.watch2:
            w = workers.get(wid)
            s_ = self.stat.setdefault(key, [0, 0])
            s_[1] += 1
            if w is None:
                s_[0] += 1
            elif key == 'G2':
                if w[0] in plist:
                    s_[0] += 1
            elif any(DIST[w[0]][x] <= 1 for x in plist):
                s_[0] += 1
        self.watch2 = []
        en_base_now = set(en_bases)
        dl = 0.0
        for wid, w in workers.items():
            if w[3] != me and w[2] == 'C' and w[1] == 0 and w[0] in en_base_now:
                o = bel.prev_workers.get(wid)
                if o is not None and o[1] > 0:
                    dl += o[1]
        self.en_deliv += dl
        self.deliv_hist.append(dl)
        hv = 0.0; hn = 0
        for wid, w in workers.items():
            if w[3] == me and w[2] == 'C' and wid not in self.camp:
                hn += 1
                o = bel.prev_workers.get(wid)
                if o is not None and o[0] == w[0] and w[1] > o[1]:
                    hv += w[1] - o[1]
        self.myh_hist.append(hv)
        self.myn_hist.append(hn)
        self.ec_hist.append(sum(1 for w in workers.values() if w[3] != me and w[2] == 'C'))
        bel.update(workers, trees, visible, obs['carbon'], step == 0, self.last_dest)
        est = bel.estimate()

        # --- unit lists and maps
        myC = []; myP = []; enC = []; enP = []
        occ = {}
        for wid, w in workers.items():
            occ[w[0]] = wid
            if w[3] == me:
                (myC if w[2] == 'C' else myP).append(wid)
            else:
                (enC if w[2] == 'C' else enP).append(wid)
        nwork = len(myC) + len(myP)
        tree_at = {}
        for tid, t in trees.items():
            tree_at[t[0]] = tid
        adj_cnt = [0] * N; adj_mine = [0] * N
        tax_me = [0] * N        # enemy trees taxing my collectors at this cell
        for tid, t in trees.items():
            for n in NB[t[0]]:
                adj_cnt[n] += 1
                if t[2] == me:
                    adj_mine[n] += 1
            if t[2] != me:
                tax_me[t[0]] += 1
                for n in NB8[t[0]]:
                    tax_me[n] += 1
        n_alive_trees = sum(1 for t in trees.values() if t[1] < 50)
        my_tree_n = sum(1 for t in trees.values() if t[2] == me)
        price = 20.0 if my_tree_n == 0 else 20.0 + 5.0 * 1.235 ** n_alive_trees

        # enemy collector reach maps
        minreach = [INF] * N        # lightest enemy collector able to reach the cell next turn
        cnt2 = {}                   # cell -> list of enemy collector cargos within distance 2
        for wid in enC:
            w = workers[wid]
            for z in N1[w[0]]:
                if w[1] < minreach[z]:
                    minreach[z] = w[1]
            for z in N2[w[0]]:
                cnt2.setdefault(z, []).append(w[1])
        en_p_reach = set()
        en_p_stand = set()
        for wid in enP:
            en_p_reach.update(N1[workers[wid][0]])
            en_p_stand.add(workers[wid][0])
        anyreach = [m < INF for m in minreach]

        # --- carbon dynamics for planning
        H = P['H']; S1 = P['S'] + 1; gamma = P['gamma']
        endgame = R <= P['endgame']
        Hh = max(1, min(H, R - 1 if (endgame and R >= 3) else R))
        val = np.array(est, dtype=np.float64)
        g = np.full(N, REGEN, dtype=np.float64)
        mask = np.ones(N, dtype=np.float64)
        for i in range(N):
            if adj_cnt[i]:
                g[i] = 1.0 - ABS * adj_cnt[i]
        for i in tree_at:
            mask[i] = 0.0
        for i in base_cells:
            mask[i] = 0.0
        # cells an enemy collector is sitting on are half as attractive
        en_occ = [(workers[wid][0], workers[wid][1]) for wid in enC]
        if not P['displace']:
            for z_, c_ in en_occ:
                mask[z_] *= 0.5
        bg = np.empty((Hh, N), dtype=np.float64)
        cur = val * mask
        capm = CAP * mask
        for t in range(Hh):
            bg[t] = COLLECT * cur
            cur = np.minimum(cur * g, capm)
        pw = np.array([0.7 ** s for s in range(S1)], dtype=np.float64)
        shift = np.array(list(range(1, S1)) + [S1 - 1], dtype=np.int64)
        info = np.zeros(N, dtype=np.float64)
        if bel.cls_seen < 63:
            for i in range(N):
                if not bel.known[i]:
                    info[i] = 1.0
        dbase = np.array([min(DIST[i][b] for b in my_bases) for i in range(N)], dtype=np.float64)
        mb = np.array(my_bases, dtype=np.int64)

        def run_dp(G, lam, C0, C1, Cr, base_val, term, want_policy):
            """Backward induction. Returns V at tau=1 (S1,N) and optional policy."""
            V = term
            pol = [] if want_policy else None
            for t in range(Hh - 1, 0, -1):
                C = C1 if t == 1 else Cr
                arr = gamma * V[0] - C
                mv = arr[NBR].max(axis=1) - lam
                st = G[t] + gamma * V[shift] - C - lam
                Vn = np.maximum(st, mv)
                if base_val is not None:
                    Vn[:, mb] = base_val
                if want_policy:
                    pol.append((st > mv, arr[NBR].argmax(axis=1)))
                V = Vn
            if want_policy:
                pol.reverse()
            return V, pol

        zeros_sn = np.zeros((S1, N), dtype=np.float64)
        zeroN = np.zeros(N, dtype=np.float64)
        Gfree = bg[:, None, :] * pw[None, :, None]
        # value of an empty collector standing at each cell (used for home bases)
        if not endgame:
            V1f, _ = run_dp(Gfree, 0.0, zeroN, zeroN, zeroN, None, zeros_sn, False)
            arr = gamma * V1f[0]
            Wfree = np.maximum(Gfree[0][0] + gamma * V1f[1], arr[NBR].max(axis=1))
        else:
            Wfree = zeroN

        early = nwork < 10 and R > 60
        rho = P['rho_early'] if early else P['rho']
        mu = P['mu_early'] if early else P['mu']

        Q = {}          # wid -> list of 5 action values
        claim = np.ones(N, dtype=np.float64)
        en_base_set = set(en_bases)
        my_base_set = set(my_bases)
        en_tree_cells = {t[0] for t in trees.values() if t[2] != me}
        my_tree_cells = {t[0]: tid for tid, t in trees.items() if t[2] == me}

        blockading = (self.camp_level is None or self.camp_level > 0) and not P['def_always']
        en_carry_avg = sum(workers[w][1] for w in enC) / max(1, len(enC))
        self.ecar = 0.9 * self.ecar + 0.1 * en_carry_avg
        # enemy collectors' movement history -> next-cell prediction for periodic movers
        pred_next = {}
        for ewid in enC:
            h = self.ehist.setdefault(ewid, [])
            h.append(workers[ewid][0])
            if len(h) > 26:
                del h[0]
            if P['predict'] and not blockading:
                n_ = len(h)
                for per in (1, 8, 4, 2, 6, 12):
                    if n_ >= per + 2 and h[-1] == h[-1 - per] and h[-2] == h[-2 - per]:
                        pred_next[ewid] = h[-per] if per > 1 else h[-1]
                        break
        for k_ in [k_ for k_ in self.ehist if k_ not in workers]:
            del self.ehist[k_]
        # siege: light enemy collectors loitering at my bases while my cargo cannot get home
        siege_bases = set()
        for ewid in enC:
            ew = workers[ewid]
            if ew[1] <= P['camper_max']:
                for b in my_bases:
                    if DIST[ew[0]][b] <= 2:
                        siege_bases.add(b)
        self.siege_cnt = self.siege_cnt + 1 if siege_bases else 0
        gate_out = {}      # besieged base -> list of (outer cell, approach cell)
        if self.siege_cnt >= P['siege_T']:
            for b in siege_bases:
                r_, c_ = divmod(b, SIZE)
                lst = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    j_ = ((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE
                    x_ = ((r_ + 2 * dr) % SIZE) * SIZE + (c_ + 2 * dc) % SIZE
                    lst.append((x_, j_))
                gate_out[b] = lst
        need_sweeper = False
        if P['siege_spawn'] and not blockading and self.siege_cnt >= P['siege_T'] and sum(workers[w][1] for w in myC) >= P['siege_carry'] and R > 8:
            have = any(workers[w][1] == 0 and min(DIST[workers[w][0]][b] for b in siege_bases) <= 5 for w in myC)
            need_sweeper = not have

        # ---------------- planters ------------------------------------
        L = max(0, min(50, R - 1))
        TV = [-1e9] * N
        if L > 0:
            for q in range(N):
                if q in tree_at or q in base_cells:
                    continue
                tot = 0.0
                for n in NB[q]:
                    if n in tree_at or n in base_cells:
                        continue
                    v = est[n]
                    if v <= 0:
                        continue
                    m = adj_cnt[n]; mine = adj_mine[n]
                    ea = v * (1.0 - (1.0 - ABS * (m + 1)) ** L)
                    if m:
                        eb = v * (1.0 - (1.0 - ABS * m) ** L)
                        tot += ea * (mine + 1) / (m + 1) - eb * mine / m
                        tot += P['steal_w'] * (eb * (m - mine) / m - ea * (m - mine) / (m + 1))
                    else:
                        tot += ea
                TV[q] = tot * P['hold_p'] - P['cell_loss_w'] * est[q]
        SV = {}
        for tid, t in trees.items():
            Lr = max(0, min(50 - t[1], R - 1))
            tot = 0.0
            for n in NB[t[0]]:
                if n in tree_at or n in base_cells:
                    continue
                m = max(1, adj_cnt[n])
                tot += est[n] * (1.0 - (1.0 - ABS * m) ** Lr) / m
            SV[tid] = tot
        plant_tasks = []
        if L > 3:
            for q in range(N):
                tv = TV[q]
                if tv - price >= P['plant_min'] and tv >= P['plant_ratio'] * price:
                    plant_tasks.append((tv - price, q))
            plant_tasks.sort(reverse=True)
            plant_tasks = plant_tasks[:40]
        seize_tasks = []
        for tid, t in trees.items():
            if t[2] != me:
                w = (1.0 + P['steal_w']) * SV[tid] - 20.0
                if w >= P['seize_min']:
                    seize_tasks.append((w, t[0]))
        en_p_pos = [workers[w][0] for w in enP]
        guard_tasks = []
        for q, tid in my_tree_cells.items():
            sv = SV[tid]
            thr = min([DIST[q][p] for p in en_p_pos] or [99])
            if sv >= P['guard_min'] and thr <= 3:
                guard_tasks.append((P['guard_w'] * sv, q))
        spend = 0.0         # cash committed to planting / seizing this turn
        taken = set()
        pd = P['pdisc']
        # process planters closest to a good task first
        porder = sorted(myP, key=lambda w: workers[w][0])
        retirees = set()
        keep_p = P['early_planters'] if self.block_ok else P['retire_keep']
        stock_low = sum(est) < P['retire_stock']
        if need_sweeper and nwork >= P['max_units'] and myP and cash >= 30 and myC:
            # every unit slot is taken and nobody is empty: give up a planter so the base can field a sweeper
            retirees.add(min(myP, key=lambda w: min(DIST[workers[w][0]][workers[cw][0]] for cw in myC)))
            self.no_p_until = step + P['no_p_T']
        elif step >= P['retire_from'] and stock_low and R >= P['retire_min_left'] and len(myP) > keep_p and cash >= 30 and myC:
            # late trees no longer pay for the unit slot: free it for a collector
            def _pv(w):
                p_ = workers[w][0]
                return max([wv * pd ** DIST[p_][q] for wv, q in plant_tasks] + [wv * pd ** DIST[p_][q] for wv, q in seize_tasks] + [0.0])
            for w in sorted(myP, key=_pv)[:len(myP) - keep_p]:
                retirees.add(w)
        for wid in porder:
            p = workers[wid][0]
            if wid in retirees:
                dests = [p] + NB[p]
                q5 = []
                for a in range(5):
                    z = dests[a]
                    d = min(DIST[z][workers[cw][0]] for cw in myC)
                    q5.append(-float(d) - (1e6 if anyreach[z] else 0.0) - (0.5 if a == 0 else 0.0))
                if p not in tree_at and p not in base_cells and cash - spend >= price:
                    q5[0] = -1e5
                Q[wid] = q5
                continue
            tasks = []
            for wv, q in plant_tasks:
                if q in taken or any(DIST[q][z] <= 1 for z in taken):
                    continue
                tasks.append((wv, q, 0))
            for wv, q in seize_tasks:
                if q not in taken:
                    tasks.append((wv, q, 1))
            for wv, q in guard_tasks:
                if q not in taken:
                    tasks.append((wv, q, 2))
            q5 = [0.0] * 5
            dests = [p] + NB[p]
            best_t = None; best_v = 0.0
            for a in range(5):
                z = dests[a]
                bv = 0.0; bt = None
                for wv, q, kind in tasks:
                    d = DIST[z][q]
                    if a == 0 and d == 0:
                        if kind == 0 and cash - spend < price:
                            v = wv * pd * pd
                        elif kind == 1 and cash - spend < 20:
                            v = wv * pd * pd
                        else:
                            v = wv * pd if kind != 2 else wv * pd
                    elif a == 0:
                        continue
                    else:
                        v = wv * pd ** (d + 2)
                    if v > bv:
                        bv = v; bt = (q, kind)
                q5[a] = bv
                if bv > best_v:
                    best_v = bv; best_t = bt
            # staying on an empty cell with enough cash plants a tree: forbid unless wanted
            if p not in tree_at and p not in base_cells and cash - spend >= price:
                tv = TV[p]
                if not (tv - price >= P['plant_min'] and tv >= P['plant_ratio'] * price):
                    q5[0] = -1e5
            if p in en_tree_cells and cash - spend >= 20 and not any(q == p for _, q in seize_tasks):
                q5[0] = -1e5
            if bel.cls_seen < 63:
                for a in range(1, 5):
                    z = dests[a]
                    q5[a] += P['info_p'] * sum(1 for n in NB[z] if not bel.known[n])
            # safety
            for a in range(5):
                z = dests[a]
                if anyreach[z]:
                    q5[a] -= 1e6
                elif z in cnt2:
                    q5[a] -= 4.0 * len(cnt2[z])
                if z in en_p_stand and a != 0:
                    if z in my_tree_cells:
                        q5[a] += (1.0 + P['steal_w']) * SV[my_tree_cells[z]] - 10.0
                    else:
                        q5[a] -= P['pp_stand']
                elif z in en_p_reach:
                    # head-on planter collisions cost a unit each
                    if not (a == 0 and p in my_tree_cells):
                        q5[a] -= P['pp_cost']
                if z in en_base_set:
                    q5[a] -= 20.0
            if best_t is not None:
                taken.add(best_t[0])
                if q5[0] >= max(q5[1:]) and best_t[0] == p:
                    if best_t[1] == 0 and cash - spend >= price:
                        spend += price
                    elif best_t[1] == 1 and cash - spend >= 20:
                        spend += 20.0
            Q[wid] = q5

        # ---------------- defence / attack bonuses for collectors ---------
        bonus = {wid: [0.0] * 5 for wid in myC}
        new_watch = []
        new_watch2 = []
        cpos = {wid: workers[wid][0] for wid in myC}
        for ewid in enP:
            e = workers[ewid][0]
            on_my_tree = e in my_tree_cells
            key = 'PT' if on_my_tree else 'P'
            ps = self.pstay(key, 4.0, 5.0) if on_my_tree else self.pstay(key, 1.5, 5.0)
            threatened = False
            for wid in myC:
                p = cpos[wid]
                if DIST[p][e] == 1:
                    threatened = True
                    a = NB[p].index(e) + 1
                    val_k = P['kill_p'] + (SV[my_tree_cells[e]] if on_my_tree else 0.0)
                    bonus[wid][a] += ps * val_k
            if threatened or on_my_tree:
                new_watch.append((ewid, e, key))
            elif P['lunge'] > 0:
                near = [wid for wid in myC if DIST[cpos[wid]][e] == 2]
                if near:
                    pa = self.pstay('P2', 0.6, 6.0) * P['lunge']
                    for wid in near:
                        p = cpos[wid]
                        mids = [m for m in NB[p] if DIST[m][e] == 1]
                        for m in mids:
                            bonus[wid][NB[p].index(m) + 1] += pa / len(mids) * P['kill_p']
                    new_watch2.append((ewid, [cpos[wid] for wid in near], 'P2'))
            # intercept: planter next to one of my trees -> step onto / next to the tree
            if not on_my_tree:
                for q in NB[e]:
                    if q in my_tree_cells and SV[my_tree_cells[q]] >= P['guard_min']:
                        best = None
                        for wid in myC:
                            d = DIST[cpos[wid]][q]
                            if d <= 2 and (best is None or d < best[0]):
                                best = (d, wid)
                        if best is not None:
                            wid = best[1]; p = cpos[wid]
                            dests = [p] + NB[p]
                            for a in range(5):
                                if DIST[dests[a]][q] <= 1:
                                    bonus[wid][a] += P['guard_w'] * SV[my_tree_cells[q]]
        for ewid in enC:
            ew = workers[ewid]
            e = ew[0]
            threatened = False
            for wid in myC:
                p = cpos[wid]
                if DIST[p][e] == 1 and workers[wid][1] < ew[1]:
                    threatened = True
                    a = NB[p].index(e) + 1
                    ps = self.pstay('C', 1.0, 5.0)
                    bonus[wid][a] += ps * (ew[1] + P['kill_c'])
            if threatened:
                new_watch.append((ewid, e, 'C'))
            elif P['lunge'] > 0:
                near = [wid for wid in myC if DIST[cpos[wid]][e] == 2 and workers[wid][1] < ew[1]]
                if near:
                    pa = self.pstay('C2', 0.6, 6.0) * P['lunge']
                    for wid in near:
                        p = cpos[wid]
                        mids = [m for m in NB[p] if DIST[m][e] == 1]
                        for m in mids:
                            bonus[wid][NB[p].index(m) + 1] += pa / len(mids) * (ew[1] + P['kill_c'])
                    new_watch2.append((ewid, [cpos[wid] for wid in near], 'C2'))
        # intercept loaded enemy collectors on the last cell before their base
        for ewid in enC:
            ew = workers[ewid]
            if ew[1] <= 0:
                continue
            e = ew[0]
            for b in en_bases:
                if DIST[e][b] != 2:
                    continue
                appr = [a for a in NB[b] if DIST[a][e] == 1]
                seen = []
                pg = self.pstay('G2', 2.0, 5.0)
                for wid in myC:
                    if workers[wid][1] >= ew[1]:
                        continue
                    p = cpos[wid]
                    dests = [p] + NB[p]
                    for a in appr:
                        if a in dests:
                            pg_ = pg
                            if wid in self.camp:
                                # a posted camper loses nothing by lunging unless that uncovers
                                # another loaded enemy waiting at the other entrance it guards
                                exposed = False
                                for a2 in NB[b]:
                                    if a2 == a or DIST[a2][p] > 1:
                                        continue
                                    for ew2id in enC:
                                        if ew2id != ewid and workers[ew2id][1] > workers[wid][1] and DIST[workers[ew2id][0]][a2] == 1:
                                            exposed = True
                                if not exposed:
                                    bonus[wid][dests.index(a)] += P['camp_lunge'] / len(appr)
                                pg_ = max(pg, P['lunge_floor'])
                            bonus[wid][dests.index(a)] += pg_ / len(appr) * (ew[1] + P['kill_c'])
                            if a not in seen:
                                seen.append(a)
                if seen:
                    new_watch2.append((ewid, seen, 'G2'))
        # sweepers: enemy collectors that hunt my loaded collectors or camp at my bases
        if P['base_def'] > 0:
            carried = sum(workers[w][1] for w in myC)
            for ewid in enC:
                ew = workers[ewid]
                e = ew[0]; ce = ew[1]
                tv = 0.0
                if min(DIST[e][b] for b in my_bases) <= 2:
                    tv = P['sweep_base'] + 0.1 * carried
                for wid in myC:
                    w = workers[wid]
                    if w[1] > ce and DIST[w[0]][e] <= 3:
                        tv = max(tv, P['sweep_w'] * w[1] + (P['sweep_base'] if tv > 0 else 0.0))
                if tv <= 0:
                    continue
                tv = min(tv, P['sweep_max'])
                best = None
                for wid in myC:
                    if workers[wid][1] != ce:
                        continue
                    d = DIST[cpos[wid]][e]
                    if d <= 3 and (best is None or d < best[0]):
                        best = (d, wid)
                if best is None:
                    continue
                d, wid = best
                p = cpos[wid]
                z_ = pred_next.get(ewid)
                if z_ is not None and z_ != e:
                    # a mover: meet it on the cell it is about to enter
                    hit = None
                    for wid2 in myC:
                        if workers[wid2][1] == ce and DIST[cpos[wid2]][z_] <= 1 and (hit is None or DIST[cpos[wid2]][z_] < DIST[cpos[hit]][z_]):
                            hit = wid2
                    if hit is not None:
                        p2 = cpos[hit]
                        a_ = 0 if p2 == z_ else NB[p2].index(z_) + 1
                        bonus[hit][a_] += tv + P['tie_cost'] + ce + 25.0
                        continue
                    for a in range(1, 5):
                        if DIST[NB[p][a - 1]][z_] < DIST[p][z_]:
                            bonus[wid][a] += 0.5 * tv + (P['tie_cost'] + ce if DIST[NB[p][a - 1]][e] <= 1 else 0.0)
                    continue
                if d == 1:
                    bonus[wid][NB[p].index(e) + 1] += tv + P['tie_cost'] + ce
                else:
                    for a in range(1, 5):
                        if DIST[NB[p][a - 1]][e] < d:
                            bonus[wid][a] += 0.5 * tv + (P['tie_cost'] + ce if DIST[NB[p][a - 1]][e] <= 1 else 0.0)
        self.watch = new_watch
        self.watch2 = new_watch2
        # campers: empty collectors posted diagonally to enemy bases
        ncamp = P['campers'] if (len(myC) >= P['camp_from'] and R > P['camp_until']) else 0
        for wid in list(self.camp):
            if wid not in workers:
                self.camp_lost += 1
                self.camp_loss_t = step
                self.camp_deaths.append(step)
                del self.camp[wid]
                if P['camp_orbit'] == 2 and not self.orbit_on and R > P['camp_end_from'] and \
                        sum(1 for t_ in self.camp_deaths if step - t_ <= P['orbit_win']) >= P['orbit_kill']:
                    # standing posts are being rammed: start circling instead of giving up
                    self.orbit_on = True
                    self.camp_deaths = []
        if self.camp_level is None:
            self.probing = P['probe_n'] > 0
            self.camp_level = P['probe_n'] if self.probing else P['campers']
        if self.probing and self.camp_level > 0:
            # a small probe first: an opponent that rams it with empty units gets no second serving
            if sum(1 for t_ in self.camp_deaths if step - t_ <= P['probe_win']) >= P['probe_kill']:
                self.camp_level = 0
                self.probing = False
        recent = sum(1 for t_ in self.camp_deaths if step - t_ <= P['kill_win'])
        if (recent >= P['kill_n'] or len(self.camp_deaths) >= P['lost_max']) and self.camp_level > 0:
            # the opponent trades units for my posts; give up only if its deliveries still flow
            wnd = P['kill_win']
            got = sum(self.deliv_hist[-wnd:])
            expect = P['eff_rate'] * sum(self.ec_hist[-wnd:])
            if self.flow(wnd) > P['r2_kill']:
                self.camp_level = 0
            elif len(self.camp_deaths) >= P['lost_max']:
                self.camp_deaths = self.camp_deaths[-(P['kill_n'] - 1):]
        if P['siege_hold'] and self.siege_cnt >= P['siege_hold']:
            # they are sitting on my bases: a mutual siege beats a one-sided one, keep mine up
            self.camp_level = P['campers']
            self.probing = False
            self.eval_bad = 0
        if self.camp_lost and step - self.camp_loss_t >= P['camp_decay']:
            self.camp_lost -= 1
            self.camp_loss_t = step
        if self.camp_level is None:
            self.camp_level = P['campers']
        if ncamp:
            ncamp = max(P['camp_min'], min(ncamp, self.camp_level) - self.camp_lost // P['camp_div'])
            # is the blockade actually stopping deliveries?  judge every eval_T turns in position
            inpos = sum(1 for w_, post_ in self.camp.items() if DIST[workers[w_][0]][post_] <= 1)
            if self.probing:
                if self.camp_deaths and step - self.camp_deaths[-1] <= 1:
                    self.probe_on = 0
                elif inpos >= min(ncamp, P['probe_n']):
                    self.probe_on += 1
                    if self.probe_on >= P['probe_T']:
                        self.probing = False
                        self.camp_level = P['campers']
            elif inpos >= max(1, min(3, ncamp)):
                if self.eval_on == 0:
                    self.eval_d0 = self.en_deliv; self.eval_lost0 = self.camp_lost; self.eval_ec = 0.0
                self.eval_on += 1
                self.eval_ec += len(enC)
                if self.eval_on >= P['eval_T']:
                    got = self.en_deliv - self.eval_d0
                    expect = P['eff_rate'] * self.eval_ec
                    ratio = got / max(1.0, expect)
                    ratio = self.flow(P['eval_T'])
                    if ratio > P['r2_hi'] or (ratio > P['r2_lo'] and self.camp_lost - self.eval_lost0 >= 3):
                        self.eval_bad += 1
                        if self.eval_bad >= 2:
                            self.camp_level = max(0, self.camp_level - 2)
                    elif ratio < P['r2_lo']:
                        self.eval_bad = 0
                        self.block_ok = True
                        self.camp_level = P['campers']
                    self.eval_on = 0
        self.ncamp_mid = ncamp
        if R <= P['camp_end_from'] and len(myC) >= 2:
            ncamp = max(ncamp, P['camp_end'])
        for wid in list(self.camp):
            if workers[wid][1] > P['camp_keep'] or len(self.camp) > ncamp:
                del self.camp[wid]
        if ncamp:
            posts = []; posts2 = []
            for b in en_bases:
                r_, c_ = divmod(b, SIZE)
                pa = [((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE for dr, dc in ((1, 1), (-1, -1))]
                pb = [((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE for dr, dc in ((1, -1), (-1, 1))]
                # prefer the diagonal pair with less carbon underfoot (campers must stay empty)
                ca = sum(0.0 if x in tree_at else est[x] for x in pa)
                cb = sum(0.0 if x in tree_at else est[x] for x in pb)
                if any(x in self.camp.values() for x in pb) or (ca > cb + 1.0 and not any(x in self.camp.values() for x in pa)):
                    pa, pb = pb, pa
                posts.extend(pa); posts2.extend(pb)
            posts.extend(posts2)
            used = set(self.camp.values())
            for post in posts:
                if len(self.camp) >= ncamp:
                    break
                if post in used:
                    continue
                best = None
                for wid in myC:
                    if wid in self.camp or workers[wid][1] > 0:
                        continue
                    d = DIST[cpos[wid]][post]
                    if best is None or d < best[0]:
                        best = (d, wid)
                if best is not None:
                    self.camp[best[1]] = post
                    used.add(post)

        # ---------------- collectors ----------------------------------
        corder = sorted(myC, key=lambda w: (-workers[w][1], -est[workers[w][0]]))
        for wid in corder:
            p, c, _, _ = workers[wid]
            if P['displace'] and en_occ:
                occm = claim.copy()
                for z_, c_ in en_occ:
                    if c_ <= c:
                        occm[z_] *= 0.5
                G = Gfree * occm[None, None, :]
            else:
                G = Gfree * claim[None, None, :]
            taxc = np.array(tax_me, dtype=np.float64) * (0.1 * max(c, 5.0))
            Cr = taxc.copy()
            C1 = taxc.copy()
            C0 = taxc - P['info_c'] * info
            if enC and c > 0:
                a1 = P['avoid1'] * (c + 30.0); a2 = P['avoid2'] * (c + 30.0)
                for z in range(N):
                    if minreach[z] <= c and z not in my_base_set:
                        C1[z] += a1
                        Cr[z] += a2
            # soft risk for the next two arrivals
            if enC:
                for z, lst in cnt2.items():
                    n2 = sum(1 for x in lst if x <= c)
                    if n2:
                        pen = P['soft_w'] * (c + 30.0) * (1.0 if n2 >= 3 else (0.25 if n2 == 2 else 0.06))
                        C0[z] += pen
                        C1[z] += 0.5 * pen
            if endgame:
                lam = 0.0
                term = np.full((S1, N), -1e4, dtype=np.float64)
                term[:, mb] = 0.0
                base_val = np.full(len(my_bases), float(c))
                base_val = base_val[None, :].repeat(S1, axis=0)
            else:
                lam = rho * c
                term = np.empty((S1, N), dtype=np.float64)
                term[:] = -lam * dbase * P['term_k']
                if c > 0:
                    base_val = (Wfree[mb] + mu * c)[None, :].repeat(S1, axis=0)
                else:
                    base_val = None
            V1, pol = run_dp(G, lam, C0, C1, Cr, base_val, term, True)
            if Hh >= 2:
                arr = gamma * V1[0] - C0
                qs = G[0][0][p] + gamma * V1[1][p] - C0[p] - lam
            else:
                arr = gamma * term[0] - C0
                qs = G[0][0][p] + gamma * term[0][p] - C0[p]
            dests = [p] + NB[p]
            q5 = [float(qs)] + [float(arr[z]) - lam for z in NB[p]]
            if endgame and c > 0:
                # delivered cargo counts fully when stepping into a base
                for a in range(1, 5):
                    if dests[a] in my_base_set:
                        q5[a] = float(c)
                if p in my_base_set:
                    q5[0] = float(c)
            if P['gate'] and gate_out and not endgame and not blockading:
                hb = min(my_bases, key=lambda b: DIST[p][b])
                if hb in gate_out:
                    if c >= P['gate_min']:
                        # besieged: wait on the cell straight out from an entrance, step in when it is uncovered
                        gw = min(P['gate_cap'], P['gate_w'] * c)
                        outs = gate_out[hb]
                        here = [j_ for x_, j_ in outs if x_ == p]
                        if DIST[p][hb] == 1:
                            pass
                        elif here:
                            q5[NB[p].index(here[0]) + 1] += gw
                            q5[0] += 0.6 * gw
                            for a in range(1, 5):
                                if dests[a] != here[0]:
                                    q5[a] += 0.3 * gw
                        else:
                            dnow = min(DIST[p][x_] for x_, j_ in outs)
                            for a in range(5):
                                dn = min(DIST[dests[a]][x_] for x_, j_ in outs)
                                q5[a] += 0.4 * gw * (dnow - dn)
                    elif c == 0 and DIST[p][hb] <= 3 and est[p] > 1.0 and p not in tree_at:
                        q5[0] -= 20.0       # a sweeper that loads up stops being one
            bn = bonus[wid]
            post = self.camp.get(wid)
            tie_cost = P['tie_cost']
            if post is not None:
                tie_cost = P['camp_tie']
                tgt = post
                if P['camp_orbit'] == 1 or (P['camp_orbit'] == 2 and self.orbit_on):
                    # circle the enemy base instead of standing still (never collects, hard to ram)
                    b_ = min(en_bases, key=lambda b: DIST[b][post])
                    r_, c_ = divmod(b_, SIZE)
                    ring = [((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE for dr, dc in
                            ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))]
                    if post in ring:
                        tgt = ring[(step + 1 + ring.index(post)) % 8]
                for a in range(5):
                    q5[a] = -P['camp_w'] * DIST[dests[a]][tgt] - (2.0 if a == 0 and est[p] > 1 and p != post else 0.0)
            for a in range(5):
                z = dests[a]
                q5[a] += bn[a]
                m = minreach[z]
                if z in en_base_set:
                    q5[a] -= 1e7
                elif m < c:
                    if z in my_base_set:
                        q5[a] -= 15.0
                    else:
                        if endgame and R <= 6:
                            q5[a] -= (0.7 if a == 0 else 0.45) * (c + 30.0)
                        else:
                            # lethal: ordered by how likely the hunter really goes there
                            q5[a] -= 1e6 + (500.0 if a == 0 else 0.0)
                elif m == c:
                    q5[a] -= tie_cost + c
            Q[wid] = q5
            # claim the cells this collector plans to harvest
            a0 = max(range(5), key=lambda a: q5[a])
            z = dests[a0]; s = 0
            if a0 == 0:
                claim[z] *= 0.7; s = 1
            for t in range(1, Hh):
                stay_m, dirs = pol[t - 1]
                if z in my_base_set and c > 0:
                    break
                if stay_m[min(s, S1 - 1)][z]:
                    claim[z] *= 0.7
                    s += 1
                else:
                    z = NB[z][int(dirs[z])]
                    s = 0

        # ---------------- joint resolution ----------------------------
        units = list(Q.keys())
        pos = {wid: workers[wid][0] for wid in units}
        choice = {}
        for wid in units:
            q5 = Q[wid]
            choice[wid] = max(range(5), key=lambda a: q5[a])

        def dest_of(wid):
            a = choice[wid]
            return pos[wid] if a == 0 else NB[pos[wid]][a - 1]

        for _ in range(40):
            dm = {}
            for wid in units:
                dm.setdefault(dest_of(wid), []).append(wid)
            conflict = [(z, lst) for z, lst in dm.items() if len(lst) > 1]
            if not conflict:
                break
            for z, lst in conflict:
                if endgame and R <= 1 and z in my_base_set:
                    continue        # last move: cargo is banked even if own units collide
                # the unit that loses most by yielding keeps the cell
                regs = []
                for wid in lst:
                    q5 = Q[wid]
                    a = choice[wid]
                    alt = max([q5[b] for b in range(5) if b != a])
                    regs.append((q5[a] - alt, wid))
                regs.sort(reverse=True)
                for _, wid in regs[1:]:
                    Q[wid][choice[wid]] = -1e8
                    q5 = Q[wid]
                    choice[wid] = max(range(5), key=lambda a: q5[a])
        if retirees:
            cdest = {}
            for cw in myC:
                cdest.setdefault(dest_of(cw), []).append(cw)
            for wid in retirees:
                p = pos[wid]
                dests = [p] + NB[p]
                for a in range(5):
                    z = dests[a]
                    if len(cdest.get(z, ())) == 1 and z not in en_base_set:
                        choice[wid] = a
                        break
        actions = {}
        self.last_dest = {}
        final_dest = set()
        for wid in units:
            a = choice[wid]
            z = dest_of(wid)
            final_dest.add(z)
            self.last_dest[wid] = z
            if a:
                actions[wid] = ACT[a - 1]

        # ---------------- recruiting ----------------------------------
        budget = cash - spend
        nC = len(myC); nP = len(myP)
        free_bases = [(bid, b) for bid, b in bases[me] if b not in final_dest]
        if step == 0:
            order = P['open']
        else:
            order = None
        k = 0
        if need_sweeper:
            for bid, b in sorted(free_bases, key=lambda t_: 0 if t_[1] in siege_bases else 1):
                if nwork < P['max_units'] and budget >= 30:
                    actions[bid] = 'RECCOLLECTOR'
                    budget -= 30; nwork += 1; nC += 1
                    break
        for bid, b in free_bases:
            if bid in actions:
                continue
            if nwork >= P['max_units'] or budget < 30 or R <= P['recruit_until']:
                break
            if order is not None:
                typ = order[k] if k < len(order) else None
                k += 1
                if typ is None:
                    break
            else:
                want_p = 0 if step < self.no_p_until else (P['early_planters'] if (step < P['retire_from'] - 12 or self.block_ok or not stock_low) else P['retire_keep'])
                if nP < want_p and nC >= P['planter_from'] * (nP + 0) and nC >= 2:
                    typ = 'P'
                else:
                    typ = 'C'
            # never spawn into an enemy collector stepping on the base with equal cargo
            actions[bid] = 'RECPLANTER' if typ == 'P' else 'RECCOLLECTOR'
            budget -= 30
            nwork += 1
            if typ == 'P':
                nP += 1
            else:
                nC += 1
        return actions


def _fallback_agent(observation, configuration):
    """Minimal safe policy used only if the main brain raises unexpectedly:
    loaded collectors walk to the nearest own base, everyone else holds,
    bases recruit a collector when affordable."""
    try:
        me = observation['player']
        players = observation['players']
        try:
            size = configuration.get('size', 15)
        except Exception:
            size = getattr(configuration, 'size', 15)
        cash, centers, workers, _trees = players[me]
        actions = {}

        def dist(a, b):
            ar, ac = divmod(a, size)
            br, bc = divmod(b, size)
            return min(abs(ar - br), size - abs(ar - br)) + min(abs(ac - bc), size - abs(ac - bc))

        def step_toward(pos, target):
            pr, pc = divmod(pos, size)
            qr, qc = divmod(target, size)
            dr = qr - pr
            if dr > size // 2:
                dr -= size
            elif dr < -size // 2:
                dr += size
            dc = qc - pc
            if dc > size // 2:
                dc -= size
            elif dc < -size // 2:
                dc += size
            if dr != 0 and (abs(dr) >= abs(dc) or dc == 0):
                return 'DOWN' if dr > 0 else 'UP'
            return 'RIGHT' if dc > 0 else 'LEFT'

        base_positions = list(centers.values())
        for wid, w in workers.items():
            pos, cargo, kind = w[0], w[1], w[2]
            if kind == 'COLLECTOR' and cargo > 0 and base_positions and pos != min(base_positions, key=lambda b: dist(pos, b)):
                actions[wid] = step_toward(pos, min(base_positions, key=lambda b: dist(pos, b)))
        for bid, _pos in centers.items():
            if cash >= 30 and len(workers) < 10:
                actions[bid] = 'RECCOLLECTOR'
        return actions
    except Exception:
        return {}



def agent(observation, configuration):
    global _BOT
    try:
        step = observation['step']
    except Exception:
        step = observation.get('step', 0)
    try:
        try:
            _size = configuration.get('size', 15)
        except Exception:
            _size = getattr(configuration, 'size', 15)
        try:
            _steps = configuration.get('episodeSteps', 300)
        except Exception:
            _steps = getattr(configuration, 'episodeSteps', 300)
        if _size != SIZE or _steps < 10:
            return _fallback_agent(observation, configuration)
    except Exception:
        pass
    if step == 0:
        _BOT = Bot()
    try:
        return _BOT.act(observation, configuration)
    except Exception:
        if os.environ.get('CARBON_DEBUG'):
            raise
        return _fallback_agent(observation, configuration)

