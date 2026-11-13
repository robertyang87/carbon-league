"""m2_bot -- whole-game rule bot imitating RyanX-AIPowercore-M2, built only from decision trees fitted to M2's replays
(m2fit/*.py -> m2_bot_ctrees.json [collectors], m2_bot_ptrees.json [planters + bases]).  No neural net.
Carbon comes from a cnet2_feat.Feat belief (the trees were fitted on the full carbon map).

  base       if cash >= 30 and workers < 10: tree b_recruit says whether (drawn from the leaf frequency with a per-game
             seeded RNG, M2's recruits look stochastic given our features); tree b_ptype says planter or collector.
  collector  on a carbon cell: tree c_stay says stay (collect) or leave; leaving / travelling: tree c_home says go home
             (nearest own base) or collect; collect: tree c_tgt scores every carbon cell within 7 steps, best wins
             (two of our collectors never share a target). Walk: approach the target, safety first.
  planter    tree p_tgt scores every cell within 7 steps as the next stay cell (plant / seize / wait); walk to it, stay.
  safety     never step where an enemy collector with cargo <= ours can also arrive (when it matters), never two of our
             units into one cell, never onto an enemy base; force home when the steps left only just cover the trip."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cnet2_feat as F
from m2fit.feats import board, col_cands, D, NBR, DIRS, CNAMES
from m2fit.policy import load_trees, col_target, col_opts, pl_target, pl_opts, OPTS, ONAMES_C, ONAMES_P
from mfit.extract import state_feats
from mfit.extract_tgt import enrich

BF = ["step", "left", "cash", "price", "own_tr", "en_tr", "n_col", "n_pl", "en_col", "en_pl", "n_wk", "rec_pl", "rec_col", "bidx",
      "base_occ", "cargo", "ecash"]


def _bfs(src, cost):
    """cheapest path cost to src where entering cell q costs 1 + cost[q] (Dijkstra on the torus)"""
    import heapq
    dist = np.full(225, 1e9); dist[src] = 0.0; h = [(0.0, src)]
    while h:
        d, p = heapq.heappop(h)
        if d > dist[p]:
            continue
        for q in NBR[p]:
            q = int(q); nd = d + 1 + cost[p]          # reverse search: moving q -> p costs entering p
            if nd < dist[q]:
                dist[q] = nd; heapq.heappush(h, (nd, q))
    return dist


_KS = np.arange(1, 6)
_G = 1 - 0.7 ** _KS                                   # share of a cell collected after k stays


def hc_target(b, u, step, taken, prev, hc):
    """hand-coded collector target (cfg HC=1): best collection rate gain/(steps to cell + stays), game facts only.
    returns (target cell, go_home)"""
    pos, cargo = b["ocol"][u]
    car = b["car"]; left = 300 - step
    dpos = D[pos]; dob = b["dobase"]
    if not b["obase"]:
        return pos, False
    hb = min(b["obase"], key=lambda q: (D[pos, q], q))
    # go home: enough cargo, or the end is near
    cap = hc["CAP"] if step >= hc["CAP_T"] else hc["CAP0"]
    if cargo > 0 and (cargo >= cap or left <= dob[pos] + hc["HM"]):
        return hb, True
    ok = (car > hc["MINCAR"]) & ~b["isbase"] & (b["istree"] == 0)
    if taken:
        tk = np.zeros(225, bool); tk[[t for t in taken if t != pos]] = True; ok &= ~tk
    cells = np.nonzero(ok)[0]
    if not len(cells):
        return (hb, True) if cargo > 0 else (pos, False)
    d = dpos[cells].astype(float)
    c = np.minimum(car[cells] * 1.03 ** np.where(cells == pos, 0, d), 100.0)
    c = np.where(cells == pos, car[pos], c)
    g = c[:, None] * _G[None]                                         # cells x k
    e9 = b["etree9"][cells].astype(float)
    if hc["ETA"]:                                                     # enemy-tree toll on the cargo while staying / passing
        g = g - 0.1 * e9[:, None] * (cargo + 0.5 * g) * _KS[None] * hc["ETA"]
    tot = d[:, None] + _KS[None] + hc["BW"] * dob[cells][:, None] / 2.0
    r = (g / tot ** hc["ALPHA"]).max(1)
    if hc["COMP"] and len(b["ecol_pos"]):                             # an enemy car arrives first: it collects / blocks
        de = D[np.ix_(cells, b["ecol_pos"])].min(1)
        r = r * np.where((de < d) & (cells != pos), 1 - hc["COMP"], 1.0)
    if hc["THR"] and len(b["ecol_pos"]) and cargo > 0:                # loaded: avoid cells weaker enemy cars can reach
        th = b["ecol_pos"][b["ecol_c"] < cargo]
        if len(th):
            dt = D[np.ix_(cells, th)].min(1)
            r = r * np.where(dt <= d + 1, 1 - hc["THR"], 1.0)
    if prev is not None and hc["HY"]:
        r = r * np.where(cells == prev, 1 + hc["HY"], 1.0)
    if hc["STAYB"]:
        r = r * np.where(cells == pos, 1 + hc["STAYB"], 1.0)
    k = int(np.argmax(r))
    tg = int(cells[k])
    # a loaded car next to a base with poor prospects: deposit
    if cargo > 0 and dob[pos] <= hc["NEARB"] and cargo >= hc["NEARC"] and tg != pos:
        return hb, True
    return tg, False


def plh_target(b, pos, ptaken, cash, price, etrees_age, ph):
    """hand-coded planter target (cfg PLH=1): plant where the 4 neighbours hold the most carbon, or seize a nearby
    enemy tree with life left; returns target cell"""
    car = b["car"]; d = D[pos].astype(float)
    tr = b["istree"] > 0
    adjtr = (b["tree4"] > 0)                                       # a neighbour of a tree: carbon shared with it
    nb = b["nb"] - (car * 0)                                       # 4-neighbour carbon
    ok = ~b["isbase"] & ~tr
    for q in ptaken:
        ok[q] = False
    val = nb * (1 - ph["ADJ"] * adjtr) - ph["CELLW"] * car         # planting zeroes the cell's own carbon
    val = val - ph["EPL"] * (b["depl"] <= d) * val                 # an enemy planter gets there first / clash
    val = val - ph["BASEW"] * b["dobase"]
    sc = np.where(ok, val / (d + ph["D0"]), -1e9)
    if cash >= 20 and etrees_age:
        for q, a_ in etrees_age:
            if q in ptaken:
                continue
            v = ph["SZW"] * b["nb"][q] * (50 - a_) / 50.0 / (d[q] + ph["D0"])
            if v > sc[q] or sc[q] < -1e8:
                sc[q] = v
    return int(np.argmax(sc))


# chosen on host-4 bigeval vs dg5 (60 games each, 2026-11-10): v6a = -5899/game, the best of ~25 variants
DEFAULT = dict(SAFE=10, ROUTE=1, THRC=30, ETW=2, HOME_MARGIN=8, HOME_P=0.3, ETH=3)


def make_agent(cfg=None):
    cfg = dict(DEFAULT, **(cfg or {}))
    T = load_trees(HERE)
    SAFE = float(cfg.get("SAFE", 0)); ETW = float(cfg.get("ETW", 0)); ROUTE = int(cfg.get("ROUTE", 0)); HOME_MARGIN = int(cfg.get("HOME_MARGIN", 2)); THRC = float(cfg.get("THRC", 4)); FUTC = float(cfg.get("FUTC", 2)); TGTW = float(cfg.get("TGTW", 0)); SAFE2 = float(cfg.get("SAFE2", 0)); MINC = int(cfg.get("MINC", 0)); SIDEW = float(cfg.get("SIDEW", 0)); SEIZE_R = int(cfg.get("SEIZE_R", 0)); HYST = float(cfg.get("HYST", 0)); COMPW = float(cfg.get("COMPW", 0)); PLOPT = int(cfg.get("PLOPT", 1)); LOCKW = float(cfg.get("LOCKW", 0)); NOBACK = int(cfg.get("NOBACK", 0)); LOCK0 = float(cfg.get("LOCK0", 0)); ETH = float(cfg.get("ETH", 0)); PLOCKW = float(cfg.get("PLOCKW", 0)); SEIZE_AGE = int(cfg.get("SEIZE_AGE", 40)); MINC_T = int(cfg.get("MINC_T", 250)); OPT = int(cfg.get("OPT", 1)); EXCL = int(cfg.get("EXCL", 1)); FORCE_HOME = int(cfg.get("FORCE_HOME", 1))
    COPT = int(cfg.get("COPT", 1)); SCOUT = int(cfg.get("SCOUT", 0)); SC_T = int(cfg.get("SC_T", 30)); UNSEEN_W = float(cfg.get("UNSEEN_W", 1.0)); FILL = int(cfg.get("FILL", 0)); FILL_T = int(cfg.get("FILL_T", 270)); FILL_CASH = float(cfg.get("FILL_CASH", 30)); KILL = int(cfg.get("KILL", 0)); KILL_C = float(cfg.get("KILL_C", 50)); SZG = int(cfg.get("SZG", 0)); SZ_AGE = int(cfg.get("SZ_AGE", 35)); SZ_D = int(cfg.get("SZ_D", 6)); SZ_R = int(cfg.get("SZ_R", 8)); SZ_LOCK = float(cfg.get("SZ_LOCK", 1)); PSAFE = float(cfg.get("PSAFE", 0)); HUNT = int(cfg.get("HUNT", 0)); HUNT_C = float(cfg.get("HUNT_C", 0)); PLH = int(cfg.get("PLH", 0)); PLH_T = int(cfg.get("PLH_T", 300)); MAXP = int(cfg.get("MAXP", -1)); MAXP_C = int(cfg.get("MAXP_C", 0)); MAXC = int(cfg.get("MAXC", 0))
    PHP = dict(ADJ=0.5, CELLW=1.0, EPL=0.5, BASEW=0.0, D0=2.0, SZW=1.0, SAFE=0.0)
    PHP.update({k[3:]: float(v) for k, v in cfg.items() if k.startswith("PH_")})
    HC = int(cfg.get("HC", 0)); HOME2 = int(cfg.get("HOME2", 0)); HC_T = int(cfg.get("HCT", 0)); CAPH = float(cfg.get("CAPH", 0)); CAPH_S = float(cfg.get("CAPH_S", 0)); SAFEN = int(cfg.get("SAFEN", 0)); OWNC = float(cfg.get("OWNC", 0))
    HCP = dict(CAP=150.0, CAP0=100.0, CAP_T=60, HM=4, MINCAR=2.0, ETA=1.0, BW=0.0, COMP=0.3, THR=0.3, HY=0.1, STAYB=0.0, NEARB=1, NEARC=40.0, ALPHA=1.0)
    HCP.update({k[3:]: float(v) for k, v in cfg.items() if k.startswith("HC_")})
    STAY_P = float(cfg.get("STAY_P", 0.5)); HOME_P = float(cfg.get("HOME_P", 0.5)); SAMPLE = int(cfg.get("SAMPLE", 1))
    import m2fit.policy as _pol
    _pol.ECON = bool(int(cfg.get("ECON", 1)))
    _pol.TOPK = int(cfg.get("TOPK", 0)); _pol.TOPK_D = float(cfg.get("TOPK_D", 1.0))
    st = {}

    def agent(obs, config=None):
        step = int(obs["step"]); me = int(obs["player"])
        if step == 0 or "f" not in st or st.get("me") != me:
            st.clear(); st.update(f=F.Feat(me), me=me, ptg={}, left={}, seen={"PLANTER": set(), "COLLECTOR": set()},
                                  rng=np.random.RandomState(int(cfg.get("SEED", 2024)) + me))
        f = st["f"]
        global LAST_F
        LAST_F = f
        try:
            f.observe(obs)
        except Exception:
            pass
        P = obs["players"]; mine = P[me][2]
        for u, w in mine.items():
            if w[2] == "PLANTER" and st.get("scout_pend") and u not in st["seen"]["PLANTER"]:
                st.setdefault("scouts", set()).add(u); st["scout_pend"] -= 1
            st["seen"][w[2]].add(u)
        car = np.asarray(obs["full_carbon"], float) if cfg.get("FULL") and "full_carbon" in obs else np.asarray(f.est, float)
        if UNSEEN_W != 1.0:                            # never seen (nor any mirror cell): the prior is an average over empty and rich cells
            car = car.copy(); car[f.last_step[F.MIRROR].max(1) < 0] *= UNSEEN_W
        b = board(P, me, car)
        acts = {}
        chosen = set()                              # cells our units occupy next step
        ebase = set(b["ebase"])
        EP, EC = b["ecol_pos"], b["ecol_c"]
        ecur = {uid: (int(w[0]), float(w[1])) for uid, w in P[1 - me][2].items() if w[2] == "COLLECTOR"}
        eprev = st.get("eprev", {}); st["eprev"] = ecur
        estay = [(p_, c_) for uid, (p_, c_) in ecur.items() if uid in eprev and eprev[uid][0] == p_]   # enemy cars that stayed last step
        prey_t = set()
        left = 300 - step
        # ---------------- collectors (most loaded first: they need the safe cells most)
        taken = set(); hunted = set()
        for u in sorted(b["ocol"], key=lambda u: (-b["ocol"][u][1], u)):
            pos, cargo = b["ocol"][u]
            if HC and step >= HC_T:
                tg, home = hc_target(b, u, step, taken, st['ptg'].get(u), HCP)
                rf = dict(dbase_me=float(b["dobase"][pos]), on_base=float(b["dobase"][pos] == 0), car_here=float(car[pos])); sp = hp = 0.0
            else:
                tg, home, rf, sp, hp = col_target(T, b, u, step, taken if EXCL else None, STAY_P, HOME_P, FORCE_HOME, TGTW, SIDEW, st['ptg'].get(u), HYST, COMPW,
                                          {c for c, t0 in st['left'].get(u, []) if step - t0 <= NOBACK} if NOBACK else None)
            if NOBACK and rf['on_base'] == 0 and rf['car_here'] > 0 and tg != pos:
                st['left'].setdefault(u, []).append((pos, step))
            if CAPH and cargo >= CAPH + CAPH_S * step and b["obase"] and not home:   # cap the cargo at risk: deposit
                home = True; tg = min(b["obase"], key=lambda q: (D[pos, q], q))
            st["ptg"][u] = tg
            taken.add(tg)
            if FORCE_HOME and cargo > 0 and left <= rf["dbase_me"] + HOME_MARGIN and b["obase"]:
                home = True; tg = min(b["obase"], key=lambda q: (D[pos, q], q))
            X = col_opts(b, u, tg, home, sp, hp, step)
            sc = T["c_opt"].pm(X, ONAMES_C) if (OPT and not (HC and step >= HC_T) and COPT) else -X[:, 1] + 10 * X[:, 0] * X[:, 2]
            if KILL and estay:                          # an adjacent loaded enemy car that is collecting: step onto it (less cargo wins)
                pr = [(c_, p_) for p_, c_ in estay if D[pos, p_] == 1 and c_ >= KILL_C and cargo < c_ - 1e-6 and p_ not in prey_t
                      and not any(D[p_, q] <= 1 for q, c2 in zip(EP, EC) if q != p_ and c2 <= cargo)]
                if pr:
                    c_, q = max(pr); prey_t.add(q)
                    sc = np.array([-float(D[pos if d is None else int(NBR[pos][k - 1]), q]) for k, d in enumerate(OPTS)]) * 50.0 + 0.01 * sc
            if HUNT and cargo <= HUNT_C and not home and b["epl"]:   # a collector beats a planter on the same cell: hunt nearby enemy planters
                cand = [(int(D[pos, q]), q) for q in b["epl"] if D[pos, q] <= HUNT and q not in hunted and q not in ebase]
                if cand:
                    _, q = min(cand); hunted.add(q)
                    sc = np.array([-float(D[pos if d is None else int(NBR[pos][k - 1]), q]) for k, d in enumerate(OPTS)]) * 3.0 + 0.01 * sc
            if OPT and LOCK0 and cargo <= 0:              # empty: follow the target (do not flee equal-cargo enemies)
                sc = sc + LOCK0 * (-X[:, 1] + 2 * X[:, 0] * X[:, 2] - X[:, 0] * (1 - X[:, 2]))
            if OPT and LOCKW:                             # blend: follow the target (approach / stay on it), the option tree breaks ties
                sc = sc + LOCKW * (-X[:, 1] + 2 * X[:, 0] * X[:, 2] - X[:, 0] * (1 - X[:, 2]))
            thr_cells = None
            if SAFE and len(EP) and cargo > 0:
                th = EP[EC <= cargo + 1e-9]            # enemy collectors that beat (or tie) us
                thr_cells = (D[:, th] <= 1).any(1) if len(th) else np.zeros(225, bool)
                thr2 = (D[:, th] <= 2).sum(1) if len(th) else np.zeros(225)
                thrn = (D[:, th] <= 1).sum(1) if len(th) else np.zeros(225)
            hd = None
            if home and ROUTE:                            # shortest safe path home (BFS around threatened cells)
                # future cells: mild cost for currently threatened ones (threats move); the immediate cell pays THRC below
                cost = np.zeros(225) if thr_cells is None else FUTC * thr_cells.astype(float) + (0 if not SAFE2 else 2 * SAFE2 * thr2)
                if ETH and cargo > 0:                     # enemy-tree absorption on the way home (10%/step of cargo)
                    cost = cost + ETH * b["etree9"] * cargo / 100.0
                for q in ebase: cost[q] = 50
                hd = _bfs(tg, cost)
                if HOME2 and len(b["obase"]) > 1:            # route to whichever own base is cheaper (one may be blocked)
                    for q in b["obase"]:
                        if q != tg:
                            hd = np.minimum(hd, _bfs(q, cost))
            best = None
            for k, d in enumerate(OPTS):
                nxt = pos if d is None else int(NBR[pos][k - 1])
                ownp = OWNC if (OWNC and cargo > 0) else 100
                pen = (ownp if nxt in chosen else 0) + (50 if nxt in ebase else 0)
                if thr_cells is not None and thr_cells[nxt]:
                    pen += SAFE * (1 + cargo / 100.0) * (thrn[nxt] if SAFEN else 1)
                if thr_cells is not None and SAFE2:
                    pen += SAFE2 * thr2[nxt] * min(1.0, cargo / 100.0)
                if ETW and cargo > 0:
                    pen += ETW * b["etree9"][nxt] * cargo / 100.0
                if hd is not None:
                    c_ = hd[nxt] + (ownp if nxt in chosen else 0) + (THRC * (thrn[nxt] if SAFEN else 1) if thr_cells is not None and thr_cells[nxt] else 0) + (0.5 if d is None and nxt != tg else 0) \
                        + (2 * SAFE2 * thr2[nxt] if thr_cells is not None and SAFE2 else 0) + ETH * b["etree9"][nxt] * cargo / 100.0
                    key = (-c_, sc[k], -k)
                else:
                    key = (sc[k] - pen, -X[k, 1], -k)
                if best is None or key > best[0]:
                    best = (key, d, nxt)
            _, mv, nxt = best
            if os.environ.get("M2DBG") and cargo > 20 and thr_cells is not None and thr_cells[nxt]:
                print("DBG", step, u[-9:], "cargo", round(cargo), "home", home, "tg", tg, "pos", pos, "mv", mv, "thr", [bool(thr_cells[pos if d is None else int(NBR[pos][k - 1])]) for k, d in enumerate(OPTS)],
                      "chosen", [ (pos if d is None else int(NBR[pos][k - 1])) in chosen for k, d in enumerate(OPTS)], "sc", np.round(sc, 2).tolist(), file=sys.stderr)
            chosen.add(nxt)
            if mv is not None:
                acts[u] = mv
        # ---------------- planters
        pls = sorted(u for u, w in mine.items() if w[2] == "PLANTER")
        if pls:
            ctx, arr = state_feats(P, me, car, step)
            enrich(P, me, arr)
            age = {int(p): a for q in (0, 1) for p, a in P[q][3].values()}
            ptaken = set(); cash_pl = float(P[me][0])
            etrees = [(int(p_), int(a_)) for p_, a_ in P[1 - me][3].values()]
            tcells = {int(p_) for q in (0, 1) for p_, _ in P[q][3].values()}
            ntr = len(tcells); price_now = 20.0 + 5.0 * 1.235 ** ntr
            szas = {}
            if SZG:                                      # seize young enemy trees in our zone: nearest planter per tree
                cand = sorted((int(b["dobase"][q]), q, a_) for q, a_ in etrees if a_ <= SZ_AGE and b["dobase"][q] <= SZ_D)
                free = set(pls)
                for _, q, a_ in cand:
                    pu = [(int(D[int(mine[v][0]), q]), v) for v in free]
                    pu = [x for x in pu if x[0] <= SZ_R]
                    if pu:
                        dd, v = min(pu); szas[v] = q; free.discard(v)
            if SCOUT and step < SC_T:
                unseen = f.last_step[F.MIRROR].max(1) < 0
                ucls = lambda cells: len({int(F.MIRROR[c].min()) for c in cells if unseen[c]})
            for u in pls:
                pos = int(mine[u][0])
                if SCOUT and step < SC_T and u in st.setdefault("scouts", set()) and unseen.any():
                    best = None                              # scout: walk where the 4-neighbourhoods reveal the most unseen cells (x4 by symmetry)
                    for k, d in enumerate(OPTS[1:], 1):
                        n1 = int(NBR[pos][k - 1])
                        if n1 in chosen or n1 in ebase or (len(b["epl"]) and (D[np.array(b["epl"]), n1] <= 1).any()):
                            continue
                        g1 = set(int(x) for x in NBR[n1])
                        v = ucls(g1) + 0.5 * max(ucls(g1 | set(int(x) for x in NBR[int(n2)])) - ucls(g1) for n2 in NBR[n1]) + 1e-3 * k
                        if best is None or v > best[0]:
                            best = (v, d, n1)
                    if best is not None and best[0] >= 1:
                        acts[u] = best[1]; chosen.add(best[2]); ptaken.add(best[2])
                        continue
                if u in szas:
                    tg = szas[u]; tp = 1.0
                    if D[pos, tg] <= 1:
                        st["wait_sz"] = True
                elif PLH and step < PLH_T:
                    tg = plh_target(b, pos, ptaken, cash_pl, price_now, [(q, a_) for q, a_ in etrees], PHP); tp = 1.0
                else:
                    tg, tp = pl_target(T, ctx, arr, age, pos, ptaken if EXCL else None)
                if SEIZE_R and cash_pl >= 20:
                    et = [(int(D[pos, q]), a_, q) for q, a_ in etrees if D[pos, q] <= SEIZE_R and a_ <= SEIZE_AGE and q not in ptaken]
                    if et:
                        tg = min(et)[2]; tp = 1.0
                ptaken.add(tg)
                X = pl_opts(arr, ctx, pos, tg, tp, b)
                sc = T["p_opt"].pm(X, ONAMES_P) if (OPT and PLOPT and not (PLH and step < PLH_T)) else -X[:, 1] + 10 * X[:, 0] * X[:, 2]
                if PSAFE and b["epl"]:                         # fitted planter: do not step where an enemy planter can also step
                    epl_ = np.array(b["epl"], int)
                    for k, d in enumerate(OPTS):
                        nxt = pos if d is None else int(NBR[pos][k - 1])
                        sc[k] -= PSAFE * (D[epl_, nxt] <= 1).any()
                if PLH and step < PLH_T and PHP["SAFE"]:                       # avoid cells an enemy planter can step onto (both planters die)
                    epl_ = np.array(b["epl"], int)
                    if len(epl_):
                        for k, d in enumerate(OPTS):
                            nxt = pos if d is None else int(NBR[pos][k - 1])
                            sc[k] -= PHP["SAFE"] * (D[epl_, nxt] <= 1).any()
                if u in szas and SZ_LOCK:
                    sc = sc + SZ_LOCK * (-X[:, 1] + 2 * X[:, 0] * X[:, 2] - X[:, 0] * (1 - X[:, 2]))
                if OPT and PLOPT and PLOCKW:
                    sc = sc + PLOCKW * (-X[:, 1] + 2 * X[:, 0] * X[:, 2] - X[:, 0] * (1 - X[:, 2]))
                best = None
                for k, d in enumerate(OPTS):
                    nxt = pos if d is None else int(NBR[pos][k - 1])
                    pen = (100 if nxt in chosen else 0) + (50 if nxt in ebase else 0)
                    key = (sc[k] - pen, -X[k, 1], -k)
                    if best is None or key > best[0]:
                        best = (key, d, nxt)
                _, mv, nxt = best
                chosen.add(nxt)
                if tg not in tcells and D[pos, tg] <= 1:          # planter on (or one step from) a plantable target
                    st["wait_pl"] = True
                    if mv is None and pos == tg:
                        dg = st.setdefault("diag", {"on_tg": 0, "poor": 0, "rec_same": 0, "poor_after_rec": 0})
                        dg["on_tg"] += 1; dg["poor"] += cash_pl < price_now; st["diag_pend"] = st.get("diag_pend", 0) + 1
                if mv is not None:
                    acts[u] = mv
        # ---------------- bases
        cash = float(P[me][0]); nwk = len(mine); ncol_now = len(b["ocol"]); npl_now = nwk - ncol_now
        res = 0.0
        if cfg.get("PL_RES") and st.pop("wait_pl", False):       # keep the planting price in hand for a waiting planter
            res = 20.0 + 5.0 * 1.235 ** sum(len(P[q][3]) for q in (0, 1))
        st.pop("wait_pl", None)
        if st.pop("wait_sz", False):
            res = max(res, 20.0)
        ctx = dict(b["ctx"]); occ_pos = {int(w[0]) for w in mine.values()}
        for bi, (bid, bpos) in enumerate(sorted(P[me][1].items())):
            if cash < 30 + res or nwk >= 10:
                break
            x = dict(step=step, left=left, cash=cash, price=ctx["price"], own_tr=ctx["own_tr"], en_tr=ctx["en_tr"], n_col=ctx["n_col"],
                     n_pl=ctx["n_pl"], en_col=ctx["en_col"], en_pl=ctx["en_pl"], n_wk=nwk, rec_pl=len(st["seen"]["PLANTER"]),
                     rec_col=len(st["seen"]["COLLECTOR"]), bidx=bi, base_occ=int(int(bpos) in occ_pos), cargo=ctx["cargo_all"],
                     ecash=ctx["ecash"])
            pr = T["b_recruit"].p(x, "1"); pp = T["b_ptype"].p(x, "P")
            if SCOUT and step < 3 and st.get("scout_rec", 0) < SCOUT:
                kind = "P"; st["scout_rec"] = st.get("scout_rec", 0) + 1; st["scout_pend"] = st.get("scout_pend", 0) + 1
            elif MINC and ncol_now < MINC and step < MINC_T:
                kind = "C"
            elif SAMPLE:
                if st["rng"].rand() >= pr and not (FILL and step < FILL_T and cash >= FILL_CASH + res):
                    continue
                kind = "P" if st["rng"].rand() < pp else "C"
            else:
                if pr < 0.5:
                    continue
                kind = "P" if pp >= 0.5 else "C"
            if MAXC and kind == "C" and ncol_now >= MAXC:
                kind = "P"
            if MAXP >= 0 and kind == "P" and npl_now >= MAXP:
                if not MAXP_C:
                    continue
                kind = "C"
            if int(bpos) in chosen:                 # one of our units arrives on the base: a recruit would collide
                continue
            acts[bid] = "RECPLANTER" if kind == "P" else "RECCOLLECTOR"
            cash -= 30; nwk += 1; ncol_now += kind == "C"; npl_now += kind == "P"
        n_on = st.pop("diag_pend", 0)
        if n_on:
            nrec = sum(1 for v in acts.values() if v in ("RECPLANTER", "RECCOLLECTOR"))
            if nrec:
                st["diag"]["rec_same"] += n_on
                st["diag"]["poor_after_rec"] += n_on * (float(P[me][0]) - 30 * nrec < 20.0 + 5.0 * 1.235 ** sum(len(P[q][3]) for q in (0, 1)))
        if step == 299 and os.environ.get("M2BOT_DIAG"):
            import json as _j
            open(os.environ["M2BOT_DIAG"], "a").write(_j.dumps(dict(st.get("diag", {}), res=bool(cfg.get("PL_RES")))) + "\n")
        return acts

    return agent


_AG = None
LAST_F = None


def agent(obs, config=None):
    global _AG
    if _AG is None or obs["step"] == 0:
        _AG = make_agent()
    return _AG(obs, config)
