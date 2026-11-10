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


# chosen on host-4 bigeval vs dg5 (60 games each, 2026-11-10): v6a = -5899/game, the best of ~25 variants
DEFAULT = dict(SAFE=10, ROUTE=1, THRC=30, ETW=2, HOME_MARGIN=8, HOME_P=0.3, ETH=3)


def make_agent(cfg=None):
    cfg = dict(DEFAULT, **(cfg or {}))
    T = load_trees(HERE)
    SAFE = float(cfg.get("SAFE", 0)); ETW = float(cfg.get("ETW", 0)); ROUTE = int(cfg.get("ROUTE", 0)); HOME_MARGIN = int(cfg.get("HOME_MARGIN", 2)); THRC = float(cfg.get("THRC", 4)); FUTC = float(cfg.get("FUTC", 2)); TGTW = float(cfg.get("TGTW", 0)); SAFE2 = float(cfg.get("SAFE2", 0)); MINC = int(cfg.get("MINC", 0)); SIDEW = float(cfg.get("SIDEW", 0)); SEIZE_R = int(cfg.get("SEIZE_R", 0)); HYST = float(cfg.get("HYST", 0)); COMPW = float(cfg.get("COMPW", 0)); PLOPT = int(cfg.get("PLOPT", 1)); LOCKW = float(cfg.get("LOCKW", 0)); NOBACK = int(cfg.get("NOBACK", 0)); LOCK0 = float(cfg.get("LOCK0", 0)); ETH = float(cfg.get("ETH", 0)); PLOCKW = float(cfg.get("PLOCKW", 0)); SEIZE_AGE = int(cfg.get("SEIZE_AGE", 40)); MINC_T = int(cfg.get("MINC_T", 250)); OPT = int(cfg.get("OPT", 1)); EXCL = int(cfg.get("EXCL", 1)); FORCE_HOME = int(cfg.get("FORCE_HOME", 1))
    STAY_P = float(cfg.get("STAY_P", 0.5)); HOME_P = float(cfg.get("HOME_P", 0.5)); SAMPLE = int(cfg.get("SAMPLE", 1))
    import m2fit.policy as _pol
    _pol.ECON = bool(int(cfg.get("ECON", 1)))
    st = {}

    def agent(obs, config=None):
        step = int(obs["step"]); me = int(obs["player"])
        if step == 0 or "f" not in st or st.get("me") != me:
            st.clear(); st.update(f=F.Feat(me), me=me, ptg={}, left={}, seen={"PLANTER": set(), "COLLECTOR": set()},
                                  rng=np.random.RandomState(int(cfg.get("SEED", 2024)) + me))
        f = st["f"]
        try:
            f.observe(obs)
        except Exception:
            pass
        P = obs["players"]; mine = P[me][2]
        for u, w in mine.items():
            st["seen"][w[2]].add(u)
        car = np.asarray(obs["full_carbon"], float) if cfg.get("FULL") and "full_carbon" in obs else np.asarray(f.est, float)
        b = board(P, me, car)
        acts = {}
        chosen = set()                              # cells our units occupy next step
        ebase = set(b["ebase"])
        EP, EC = b["ecol_pos"], b["ecol_c"]
        left = 300 - step
        # ---------------- collectors (most loaded first: they need the safe cells most)
        taken = set()
        for u in sorted(b["ocol"], key=lambda u: (-b["ocol"][u][1], u)):
            pos, cargo = b["ocol"][u]
            tg, home, rf, sp, hp = col_target(T, b, u, step, taken if EXCL else None, STAY_P, HOME_P, FORCE_HOME, TGTW, SIDEW, st['ptg'].get(u), HYST, COMPW,
                                          {c for c, t0 in st['left'].get(u, []) if step - t0 <= NOBACK} if NOBACK else None)
            if NOBACK and rf['on_base'] == 0 and rf['car_here'] > 0 and tg != pos:
                st['left'].setdefault(u, []).append((pos, step))
            st["ptg"][u] = tg
            taken.add(tg)
            if FORCE_HOME and cargo > 0 and left <= rf["dbase_me"] + HOME_MARGIN and b["obase"]:
                home = True; tg = min(b["obase"], key=lambda q: (D[pos, q], q))
            X = col_opts(b, u, tg, home, sp, hp, step)
            sc = T["c_opt"].pm(X, ONAMES_C) if OPT else -X[:, 1] + 10 * X[:, 0] * X[:, 2]
            if OPT and LOCK0 and cargo <= 0:              # empty: follow the target (do not flee equal-cargo enemies)
                sc = sc + LOCK0 * (-X[:, 1] + 2 * X[:, 0] * X[:, 2] - X[:, 0] * (1 - X[:, 2]))
            if OPT and LOCKW:                             # blend: follow the target (approach / stay on it), the option tree breaks ties
                sc = sc + LOCKW * (-X[:, 1] + 2 * X[:, 0] * X[:, 2] - X[:, 0] * (1 - X[:, 2]))
            thr_cells = None
            if SAFE and len(EP) and cargo > 0:
                th = EP[EC <= cargo + 1e-9]            # enemy collectors that beat (or tie) us
                thr_cells = (D[:, th] <= 1).any(1) if len(th) else np.zeros(225, bool)
                thr2 = (D[:, th] <= 2).sum(1) if len(th) else np.zeros(225)
            hd = None
            if home and ROUTE:                            # shortest safe path home (BFS around threatened cells)
                # future cells: mild cost for currently threatened ones (threats move); the immediate cell pays THRC below
                cost = np.zeros(225) if thr_cells is None else FUTC * thr_cells.astype(float) + (0 if not SAFE2 else 2 * SAFE2 * thr2)
                if ETH and cargo > 0:                     # enemy-tree absorption on the way home (10%/step of cargo)
                    cost = cost + ETH * b["etree9"] * cargo / 100.0
                for q in ebase: cost[q] = 50
                hd = _bfs(tg, cost)
            best = None
            for k, d in enumerate(OPTS):
                nxt = pos if d is None else int(NBR[pos][k - 1])
                pen = (100 if nxt in chosen else 0) + (50 if nxt in ebase else 0)
                if thr_cells is not None and thr_cells[nxt]:
                    pen += SAFE * (1 + cargo / 100.0)
                if thr_cells is not None and SAFE2:
                    pen += SAFE2 * thr2[nxt] * min(1.0, cargo / 100.0)
                if ETW and cargo > 0:
                    pen += ETW * b["etree9"][nxt] * cargo / 100.0
                if hd is not None:
                    c_ = hd[nxt] + (100 if nxt in chosen else 0) + (THRC if thr_cells is not None and thr_cells[nxt] else 0) + (0.5 if d is None and nxt != tg else 0) \
                        + (2 * SAFE2 * thr2[nxt] if thr_cells is not None and SAFE2 else 0) + ETH * b["etree9"][nxt] * cargo / 100.0
                    key = (-c_, sc[k], -k)
                else:
                    key = (sc[k] - pen, -X[k, 1], -k)
                if best is None or key > best[0]:
                    best = (key, d, nxt)
            _, mv, nxt = best
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
            for u in pls:
                pos = int(mine[u][0])
                tg, tp = pl_target(T, ctx, arr, age, pos, ptaken if EXCL else None)
                if SEIZE_R and cash_pl >= 20:
                    et = [(int(D[pos, q]), a_, q) for q, a_ in etrees if D[pos, q] <= SEIZE_R and a_ <= SEIZE_AGE and q not in ptaken]
                    if et:
                        tg = min(et)[2]; tp = 1.0
                ptaken.add(tg)
                X = pl_opts(arr, ctx, pos, tg, tp, b)
                sc = T["p_opt"].pm(X, ONAMES_P) if (OPT and PLOPT) else -X[:, 1] + 10 * X[:, 0] * X[:, 2]
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
                if mv is not None:
                    acts[u] = mv
        # ---------------- bases
        cash = float(P[me][0]); nwk = len(mine); ncol_now = len(b["ocol"])
        ctx = dict(b["ctx"]); occ_pos = {int(w[0]) for w in mine.values()}
        for bi, (bid, bpos) in enumerate(sorted(P[me][1].items())):
            if cash < 30 or nwk >= 10:
                break
            x = dict(step=step, left=left, cash=cash, price=ctx["price"], own_tr=ctx["own_tr"], en_tr=ctx["en_tr"], n_col=ctx["n_col"],
                     n_pl=ctx["n_pl"], en_col=ctx["en_col"], en_pl=ctx["en_pl"], n_wk=nwk, rec_pl=len(st["seen"]["PLANTER"]),
                     rec_col=len(st["seen"]["COLLECTOR"]), bidx=bi, base_occ=int(int(bpos) in occ_pos), cargo=ctx["cargo_all"],
                     ecash=ctx["ecash"])
            pr = T["b_recruit"].p(x, "1"); pp = T["b_ptype"].p(x, "P")
            if MINC and ncol_now < MINC and step < MINC_T:
                kind = "C"
            elif SAMPLE:
                if st["rng"].rand() >= pr:
                    continue
                kind = "P" if st["rng"].rand() < pp else "C"
            else:
                if pr < 0.5:
                    continue
                kind = "P" if pp >= 0.5 else "C"
            if int(bpos) in chosen:                 # one of our units arrives on the base: a recruit would collide
                continue
            acts[bid] = "RECPLANTER" if kind == "P" else "RECCOLLECTOR"
            cash -= 30; nwk += 1; ncol_now += kind == "C"
        return acts

    return agent


_AG = None


def agent(obs, config=None):
    global _AG
    if _AG is None or obs["step"] == 0:
        _AG = make_agent()
    return _AG(obs, config)
