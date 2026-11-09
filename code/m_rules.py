"""m_rules -- opening (steps < MR_STEPS, default 100) fitted to the M family (RyanX-AIPowercore-M2/M3/M4) by shallow decision
trees (mfit/fit.py -> m_rules_trees.json). Replaces planter moves and base recruits; collectors keep the wrapped agent's moves.
Carbon comes from a cnet2_feat.Feat belief (the trees were fitted on full carbon; the belief fits identically in CV).
  base:    if cash >= 30 and workers < 10: tree 'recruit' says whether; tree 'ptype' says planter or collector.
  planter: tree 'stay' says stay (plant / seize / wait) or move; tree 'dir' scores each of the 4 moves, best wins
           (ties: keep the previous direction, then higher destination 4-neighbour carbon)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cnet2_feat as F
from mfit.extract import state_feats, planter_feats, dir_feats, DIRS, NBR, D
from mfit.extract_tgt import cand_feats, enrich, best_at

OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


class Tree:
    def __init__(self, d):
        self.__dict__.update(d)
        self.cls = d["classes"]

    def proba(self, x):
        n = 0
        while self.left[n] != -1:
            n = self.left[n] if x[self.feats[self.feat[n]]] <= self.thr[n] else self.right[n]
        return dict(zip(self.cls, self.val[n]))


def wrap(inner, cfg=None):
    cfg = dict(cfg or {})
    RO = int(cfg.get("MR_STEPS", 100))
    T = {k: Tree(v) for k, v in json.load(open(cfg.get("TREES") or os.path.join(HERE, "m_rules_trees.json"))).items()}
    STAY_P = float(cfg.get("STAY_P", 0.5)); REC_P = float(cfg.get("REC_P", 0.5)); PL_P = float(cfg.get("PL_P", 0.5))
    OWNC = float(cfg.get("OWNC", 1.0))        # tgt mode: avoid cells our own collectors move into
    ECOL = float(cfg.get("ECOL", 1.0))        # tgt mode: avoid cells within 1 of an enemy collector (M flees them)
    SAMPLE = bool(cfg.get("SAMPLE", True))
    FULL = bool(cfg.get("FULL", False))      # debug only: use obs['full_carbon'] if present
    SAFE = bool(cfg.get("SAFE", True))        # two of our planters never step into the same cell
    MODE = cfg.get("MODE", "tgt")             # 'tgt': target tree picks a cell, walk to it, stay there; 'dir': stay tree + dir tree
    EXCL = bool(cfg.get("EXCL", True))        # tgt mode: two of our planters never pick the same target
    st = {}

    def agent(obs, config=None):
        acts = inner(obs, config)
        step = obs["step"]; me = obs["player"]
        if step == 0 or "f" not in st or st.get("me") != me:
            st.clear(); st["f"] = F.Feat(me); st["me"] = me; st["seen"] = {"PLANTER": set(), "COLLECTOR": set()}; st["prev"] = {}
            st["rng"] = np.random.RandomState(int(cfg.get("SEED", 12345)) + me)
        f = st["f"]
        try:
            f.observe(obs)
        except Exception:
            pass
        if step >= RO:
            return acts
        acts = dict(acts or {})
        P = obs["players"]; mine = P[me][2]
        for u, w in mine.items():
            st["seen"][w[2]].add(u)
        car = obs["full_carbon"] if FULL and "full_carbon" in obs else np.asarray(f.est, float)
        ctx, arr = state_feats(P, me, car, step)
        ctx = dict(ctx, rec_pl=len(st["seen"]["PLANTER"]), rec_col=len(st["seen"]["COLLECTOR"]))
        # ---- recruiting
        for b in P[me][1]:
            acts.pop(b, None)
        c = dict(ctx); occ_pos = {int(w[0]) for w in mine.values()}
        for bi, (bid, bpos) in enumerate(sorted(P[me][1].items())):
            if c["cash"] < 30 or c["n_wk"] >= 10:
                break
            x = dict(ctx, bidx=bi, base_occ=int(int(bpos) in occ_pos))   # trees see the pre-step state for both bases
            pr, pp = T["recruit"].proba(x).get("1", 0), T["ptype"].proba(x).get("P", 0)
            if SAMPLE:                            # draw from the leaf frequencies (M's recruits look stochastic given our features)
                if st["rng"].rand() >= pr:
                    continue
                kind = "P" if st["rng"].rand() < pp else "C"
            else:
                if pr < REC_P:
                    continue
                kind = "P" if pp >= PL_P else "C"
            acts[bid] = "RECPLANTER" if kind == "P" else "RECCOLLECTOR"
            c["cash"] -= 30; c["n_wk"] += 1
        # ---- planters
        ctx["afford"] = int(ctx["cash"] >= ctx["price"])
        chosen = set(); prev = st["prev"]; newprev = {}; taken = set()
        if MODE == "tgt":
            enrich(P, me, arr)
            age = {int(p): a for q in (0, 1) for p, a in P[q][3].values()}
            own_next = set()                      # where our collectors (driven by the wrapped agent) will be next step
            for u_, w_ in mine.items():
                if w_[2] == "COLLECTOR":
                    m_ = acts.get(u_)
                    own_next.add(int(NBR[int(w_[0])][DIRS.index(m_)]) if m_ in DIRS else int(w_[0]))
        for u in sorted(u for u, w in mine.items() if w[2] == "PLANTER"):
            pos = int(mine[u][0]); pv = prev.get(u, "NONE")
            if MODE == "tgt":
                arr["best_at"][pos] = best_at(arr, pos)
                cells, cols = cand_feats(pos, ctx, arr, dict(age=age))
                tt = T["target"]; fi = tt.feats
                best = None
                for i, cell in enumerate(cells):
                    if EXCL and int(cell) in taken:
                        continue
                    x = {k: cols[k][i] for k in fi}
                    key = (tt.proba(x).get("1", 0), -cols["c_dist"][i], cols["c_nb"][i])
                    if best is None or key > best[0]:
                        best = (key, int(cell))
                tg = pos if best is None else best[1]
                taken.add(tg)
                r0, c0 = divmod(pos, 15); r1, c1 = divmod(tg, 15)
                dr = min((r1 - r0) % 15, (r0 - r1) % 15); dc = min((c1 - c0) % 15, (c0 - c1) % 15)
                sc = []
                for k, d in enumerate([None] + DIRS):
                    nxt = pos if d is None else int(NBR[pos][k - 1])
                    if d is None:
                        base = 10 if tg == pos else -10
                    else:
                        ap = int(D[nxt, tg] < D[pos, tg])
                        base = 2 * ap + int(pv == d) + 0.5 * int((d in ("UP", "DOWN")) == (dr > dc))
                    dang = (5 if SAFE and nxt in chosen else 0) + OWNC * (nxt in own_next) + ECOL * (arr["decol"][nxt] <= 1)
                    sc.append((-dang * 100 + base, arr["nb"][nxt], d or ""))
                sc.sort(reverse=True); mv = sc[0][2] or None
                nxt = pos if mv is None else int(NBR[pos][DIRS.index(mv)])
                chosen.add(nxt); newprev[u] = "STAY" if mv is None else mv
                if mv is None:
                    acts.pop(u, None)
                else:
                    acts[u] = mv
                continue
            pf = planter_feats(pos, dict(ctx, prev_moved=int(pv in OPP)), arr)
            pf["afford"] = ctx["afford"]; pf["prev_moved"] = int(pv in OPP)
            stay = T["stay"].proba(pf).get("1", 0) >= STAY_P
            if stay and not (SAFE and pos in chosen):
                mv = None
            else:
                sc = []
                for k, d in enumerate(DIRS):
                    x = dict(pf); x.update(dir_feats(pos, d, ctx, arr, pf)); x["d_same"] = int(pv == d); x["d_rev"] = int(pv == OPP[d])
                    nxt = int(NBR[pos][k])
                    s = T["dir"].proba(x).get("1", 0) - (5.0 if SAFE and nxt in chosen else 0.0)
                    sc.append((s, x["d_same"], x["d_nb"], d, nxt))
                sc.sort(reverse=True)
                mv = sc[0][3]
            nxt = pos if mv is None else int(NBR[pos][DIRS.index(mv)])
            chosen.add(nxt)
            newprev[u] = "STAY" if mv is None else mv
            if mv is None:
                acts.pop(u, None)
            else:
                acts[u] = mv
        st["prev"] = newprev
        return acts

    agent.stats = getattr(inner, "stats", None)
    return agent


def make_from(path, cfg=None):
    """load an agent module (exp/*.py with agent() or make_agent()) and wrap it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mr_inner_" + str(abs(hash(path))), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    inner = m.make_agent() if hasattr(m, "make_agent") else m.agent
    w = wrap(inner, cfg)

    def agent(obs, config=None):
        a = w(obs, config)
        agent.last_stats = getattr(inner, "last_stats", None) or getattr(m.agent, "last_stats", None)
        return a
    return agent
