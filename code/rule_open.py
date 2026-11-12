"""rule_open -- M2/M4-style opening rules (steps < RO_STEPS, default 100) on top of any agent; after that the wrapped
agent plays alone. Replaces only planter moves and base recruit actions; collectors keep the wrapped agent's moves.

Spec (2026-11-09, from the M2/M4 replay statistics):
  recruiting  first planter at step FIRST_P (8); by step 25 >= 3 planters and >= 4 cars, by 50 7 / 6, by 100 11 / 8
              (cumulative, counting every unit we ever owned of that type); no recruit without the cash for it.
  planters    target = the free cell within 3 steps with the highest 4-neighbour carbon, only if >= THR (100); none ->
              the nearest free cell anywhere with >= THR. On the target: stay (plant). Never seize (no target on a tree,
              and a planter never stays on an enemy tree). If the next cell could also be entered by an enemy planter
              (enemy planter within 1 of it), take the next-best move. Ties in carbon: closer to our nearest base first.
              Two of our planters never share a target.
Carbon comes from a cnet2_feat.Feat belief (the observation only shows cells next to our own units)."""
import os

import numpy as np

import cnet2_feat as F

S = 15
MV = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}


def _step(pos, mv):
    r, c = divmod(pos, S); dr, dc = MV[mv]
    return ((r + dr) % S) * S + (c + dc) % S


def _nb4(est):
    g = np.asarray(est, dtype=np.float64).reshape(S, S)
    return (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)).ravel()


def _target_count(step, sched):
    """piecewise-linear cumulative target from [(step, count), ...] (count 0 before the first point)."""
    if step < sched[0][0]:
        return 0
    for (s0, c0), (s1, c1) in zip(sched, sched[1:]):
        if step <= s1:
            return c0 + (c1 - c0) * (step - s0) / max(1, s1 - s0)
    return sched[-1][1]


def wrap(inner, cfg=None):
    cfg = dict(cfg or {})
    RO = int(cfg.get("RO_STEPS", 100)); THR = float(cfg.get("THR", 100.0)); FP = int(cfg.get("FIRST_P", 8))
    PS = cfg.get("PS") or [(FP, 1), (25, 3), (50, 7), (100, 11)]
    CS = cfg.get("CS") or [(0, int(cfg.get("C0", 1))), (25, 4), (50, 6), (100, 8)]
    SAFE = int(cfg.get("SAFE", 0))            # target cells at least SAFE steps from every enemy planter (0 = off)
    RESERVE = bool(cfg.get("RESERVE", False))  # recruit only if the plant price is still covered afterwards
    st = {}

    def agent(obs, config=None):
        acts = inner(obs, config)
        step = obs["step"]; me = obs["player"]
        if step == 0 or "f" not in st or st.get("me") != me:
            st.clear(); st["f"] = F.Feat(me); st["me"] = me; st["seen"] = {"PLANTER": set(), "COLLECTOR": set()}
        f = st["f"]
        try:
            f.observe(obs)
        except Exception:
            pass
        if step >= RO:
            return acts
        acts = dict(acts or {})
        P = obs["players"]; cash = float(P[me][0]); bases = [int(b) for b in P[me][1].values()]
        mine = P[me][2]
        for u, w in mine.items():
            st["seen"][w[2]].add(u)
        # ---- recruiting (replaces the wrapped agent's base actions)
        net_base = {b: acts.get(b) for b in P[me][1]}
        for b in P[me][1]:
            acts.pop(b, None)
        nP, nC = len(st["seen"]["PLANTER"]), len(st["seen"]["COLLECTOR"])
        cost = 30.0
        own_ = len(P[me][3]); live_ = sum(1 for q in (0, 1) for _, a_ in P[q][3].values() if a_ < 50)
        price_ = 20.0 if own_ == 0 else 20.0 + 5.0 * 1.235 ** live_
        for bid in P[me][1]:
            if cash < cost or (RESERVE and nP >= 1 and cash - cost < price_):
                break
            if cfg.get("CARS_FIRST") and nC < _target_count(step, CS):    # economy first: cars before planters
                acts[bid] = "RECCOLLECTOR"; cash -= cost; nC += 1
            elif nP < _target_count(step, PS):
                acts[bid] = "RECPLANTER"; cash -= cost; nP += 1
            elif cfg.get("NET_CARS"):              # cars: whatever the wrapped agent recruits (its economy)
                if net_base.get(bid) == "RECCOLLECTOR":
                    acts[bid] = "RECCOLLECTOR"; cash -= cost; nC += 1
            elif nC < _target_count(step, CS):
                acts[bid] = "RECCOLLECTOR"; cash -= cost; nC += 1
        # ---- planters
        est = np.asarray(f.est, dtype=np.float64); nb = _nb4(est)
        occupied = np.zeros(S * S, bool)
        for q in (0, 1):
            for pos, _ in P[q][3].values():
                occupied[int(pos)] = True
            for b in P[q][1].values():
                occupied[int(b)] = True
        enemy_tree = np.zeros(S * S, bool)
        for pos, _ in P[1 - me][3].values():
            enemy_tree[int(pos)] = True
        good = (~occupied) & (nb >= THR)
        epl0 = [int(w[0]) for w in P[1 - me][2].values() if w[2] == "PLANTER"]
        if SAFE and epl0:
            good &= np.min(F.DIST[:, epl0], axis=1) >= SAFE
        dbase = np.min(F.DIST[:, bases], axis=1) if bases else np.zeros(S * S)
        if cfg.get("NB_MAX") is not None and bases:  # option A: plant only within NB_MAX steps of one of our bases
            good &= dbase <= int(cfg["NB_MAX"])
        epl = [int(w[0]) for w in P[1 - me][2].values() if w[2] == "PLANTER"]
        taken = set()
        # ---- safety layer: where can a planter be killed next step?
        #   our own collectors' next cells (the wrapped agent drives them, it does not know our planter moves),
        #   cells next to an enemy collector (it may step in) or an enemy planter (planter-planter: both die),
        #   cells our other planters already chose.
        own_next = set()
        for u, w in mine.items():
            if w[2] == "COLLECTOR":
                mv_ = acts.get(u)
                own_next.add(_step(int(w[0]), mv_) if mv_ in MV else int(w[0]))
        ecol = [int(w[0]) for w in P[1 - me][2].values() if w[2] == "COLLECTOR"]
        chosen = set()

        EPW = float(cfg.get("EPL_W", 1.0))         # weight of enemy-planter adjacency (0 = M-like: accept clashes)

        def danger(c):
            return (2 if c in own_next else 0) + (2 if c in chosen else 0) \
                + sum(1 for e in ecol if F.DIST[c, e] <= 1) + EPW * sum(1 for e in epl if F.DIST[c, e] <= 1)

        for u in sorted(u for u, w in mine.items() if w[2] == "PLANTER"):
            pos = int(mine[u][0]); d = F.DIST[pos]
            ok_ = good
            if cfg.get("SEIZE"):                  # M-like: profitable enemy trees within reach are targets too
                ok_ = good | (enemy_tree & (nb >= float(cfg.get("SEIZE_MIN", 40.0))))
            cand = np.nonzero(ok_ & (d <= 3))[0]
            if cfg.get("SEIZE_FIRST") and cfg.get("SEIZE"):   # an enemy tree in reach beats any free cell
                sz_ = [c for c in cand if enemy_tree[c]]
                if sz_:
                    cand = np.array(sz_)
            cand = [c for c in cand if c not in taken]
            if not cand:
                far = [c for c in np.nonzero(good)[0] if c not in taken]
                tgt = min(far, key=lambda c: (d[c], -nb[c], dbase[c])) if far else None
            elif cfg.get("NEAREST", False):     # v2 (2026-11-09): nearest acceptable cell, ties -> nearer our base
                tgt = min(cand, key=lambda c: (d[c], dbase[c], -nb[c]))
            else:
                tgt = max(cand, key=lambda c: (nb[c], -dbase[c], -d[c]))
            if tgt is not None:
                taken.add(tgt)
            if cfg.get("PRICE_CHECK") and tgt is not None and not enemy_tree[tgt]:
                # plant only when the tree is worth its price: nb4 * (1 - 0.9625^30) - current plant price > 0;
                # otherwise do not stand on a free cell (that would plant): wait on our nearest own tree instead
                if nb[tgt] * (1.0 - 0.9625 ** 30) - price_ <= 0:
                    own_t = [int(q) for q, _ in P[me][3].values()]
                    tgt = min(own_t, key=lambda c: d[c]) if own_t else None
            # options: stay (plants on a free target) and the 4 moves; rank by (danger, progress, no-seize)
            opts = []
            for mv in [None] + list(MV):
                nxt = pos if mv is None else _step(pos, mv)
                if mv is None and enemy_tree[pos] and not (cfg.get("SEIZE") and pos == tgt):
                    continue                      # staying on an enemy tree would seize (only allowed on a seize target)
                prog = 0 if tgt is None else (F.DIST[nxt, tgt] - d[tgt])     # -1 closer, 0 same, +1 away
                stay_bonus = -1 if (mv is None and tgt == pos) else 0       # on target: staying = planting
                if mv is None and cfg.get("PRICE_CHECK") and tgt != pos and not occupied[pos]:
                    stay_bonus = 1                # off target on a free cell: staying would plant a tree not worth it
                opts.append((danger(nxt), stay_bonus if mv is None else prog, mv, nxt))
            if not opts:
                continue
            opts.sort(key=lambda o: (o[0], o[1]))
            _, _, mv, nxt = opts[0]
            chosen.add(nxt)
            if mv is None:
                acts.pop(u, None)
            else:
                acts[u] = mv
        # never stay on an enemy tree (that would seize) unless seizing is on
        for u, w in mine.items():
            if not cfg.get("SEIZE") and w[2] == "PLANTER" and enemy_tree[int(w[0])] and u not in acts:
                for mv in MV:
                    if not enemy_tree[_step(int(w[0]), mv)]:
                        acts[u] = mv; break
        return acts

    agent.stats = getattr(inner, "stats", None)
    return agent


def make_from(path, cfg=None):
    """load an agent module (exp/*.py with agent() or make_agent()) and wrap it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ro_inner_" + str(abs(hash(path))), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    inner = m.make_agent() if hasattr(m, "make_agent") else m.agent
    w = wrap(inner, cfg)

    def agent(obs, config=None):
        a = w(obs, config)
        agent.last_stats = getattr(inner, "last_stats", None) or getattr(m.agent, "last_stats", None)
        return a
    return agent
