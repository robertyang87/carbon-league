"""nn_search2 -- multi-step rollout search on top of the v9 NN policy + rule layer.

Per real step:
  1. Default joint action: our policy + rule layer (exactly v9) -> acts0.
  2. Trigger: "key" collectors (loaded and an enemy collector with cargo <= ours within KEY_R, or empty
     next to a loaded enemy collector within HUNT_R).  For each key unit (most cargo at stake first) the 5
     moves are tried: branch = FastCarbon built from our observation + belief carbon map; first step
     = acts0 with that unit's move overridden (opponent: our net from its point of view, fed our belief);
     then H-1 more steps with both sides played by the policy (batched across branches).
  3. Leaf score (ours minus theirs): cash + CARGO_W * cargo + WORKER_V * workers (+ tree value)
     [+ VALUE_W * value-net delta if a value net is given].  A move replaces the default only when it
     beats the default branch by more than MARGIN.
Time guarded: no new branch set is started after BUDGET seconds.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nnfeat as F  # noqa: E402
import nn_agent as A  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

A.ENDGAME_BRAVE = False
A.STALL_OFF_LEFT = 30

DEFAULTS = dict(
    H=6, KEY_R=2, HUNT_R=2, MAX_KEY=3, MARGIN=30.0, BUDGET=0.55, CARGO_W=0.9, WORKER_V=40.0,
    TREE_V=3.0, VALUE_W=0.0, LOCAL_R=5, HUNT=True, ENDGAME_STOP=2, HARD=0.75, SAFE_FILTER=True, ROBUST=False, LOG=False, SEARCH=True, OPP_W=None,
)


def conv_fast(x, wm, b):
    """same im2col column order and matmul as nn_agent.conv (bit-identical), wrap-padding instead of np.roll."""
    B, H, W_, C = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
    cols = np.empty((B, H, W_, 9 * C), dtype=np.float32)
    for k, (dr, dc) in enumerate(A.SHIFTS):
        cols[..., k * C:(k + 1) * C] = xp[:, 1 + dr:1 + dr + H, 1 + dc:1 + dc + W_, :]
    out = cols.reshape(B * H * W_, 9 * C) @ wm
    out += b
    return out.reshape(B, H, W_, wm.shape[1])


class FastPolicy(A.Policy):
    def _forward(self, x, s):
        p = self.sd; wm = self.wm
        x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        h = np.maximum(conv_fast(x, wm["stem.weight"], p["stem.bias"]), 0)
        for i in range(self.nb):
            t = np.maximum(conv_fast(h, wm[f"blocks.{i}.0.weight"], p[f"blocks.{i}.0.bias"]), 0)
            h = np.maximum(h + conv_fast(t, wm[f"blocks.{i}.1.weight"], p[f"blocks.{i}.1.bias"]), 0)
        patch = h[:, 6:9, 6:9, :].transpose(0, 3, 1, 2).reshape(h.shape[0], -1)
        g = h.mean(axis=(1, 2))
        z = np.concatenate([patch, g, s], axis=1)
        z = np.maximum(z @ p["fc.weight"].T + p["fc.bias"], 0)
        return z @ p["pi.weight"].T + p["pi.bias"]


def tr_copy(tr):
    n = F.Tracker.__new__(F.Tracker)
    n.me = tr.me
    n.bel = tr.bel.copy()
    n.seen = tr.seen.copy()
    n.prev_enemy = dict(tr.prev_enemy)
    n.step = tr.step
    return n


class ValueNet:
    def __init__(self, sd):
        self.sd = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items() if k != "meta"}
        self.nb = len({k.split(".")[1] for k in self.sd if k.startswith("blocks.")})
        self.wm = {k: A._wmat(v) for k, v in self.sd.items() if k.endswith(".weight") and v.ndim == 4}
        import json as _j
        meta = sd.get("meta")
        try:
            self.meta = _j.loads(str(meta)) if meta is not None else {}
        except Exception:
            self.meta = {}
        self.resid = bool(self.meta.get("resid"))

    def __call__(self, G, S):
        p, wm = self.sd, self.wm
        x = np.ascontiguousarray(G.transpose(0, 2, 3, 1).astype(np.float32))
        h = np.maximum(A.conv(x, wm["stem.weight"], p["stem.bias"]), 0)
        for i in range(self.nb):
            t = np.maximum(A.conv(h, wm[f"blocks.{i}.0.weight"], p[f"blocks.{i}.0.bias"]), 0)
            h = np.maximum(h + A.conv(t, wm[f"blocks.{i}.1.weight"], p[f"blocks.{i}.1.bias"]), 0)
        g = h.mean(axis=(1, 2))
        z = np.maximum(np.concatenate([g, S.astype(np.float32)], 1) @ p["fc.weight"].T + p["fc.bias"], 0)
        return (z @ p["out.weight"].T + p["out.bias"])[:, 0]


def batch_policy(pol, items, cap=10, subset=None):
    """items: list of (tracker, obs, stall_dict) with tracker already updated -> list of acts dicts.
    One network forward for all units of all items.  subset: only these worker ids act (others stay,
    bases do not recruit) -- used in rollouts, where far units behave identically in every branch."""
    xs, ss, meta = [], [], []
    for j, (tr, obs, _) in enumerate(items):
        G, scal, info = tr.global_planes(obs)
        info["bel"] = tr.bel
        mine = obs["players"][tr.me]
        units = []
        for wid, w in mine[2].items():
            if subset is not None and wid not in subset:
                continue
            k = 0 if w[2] == "COLLECTOR" else 1
            x, s = F.unit_inputs(G, scal, w[0], k, w[1])
            units.append((wid, k, w[0])); xs.append(x); ss.append(s)
        if subset is None:
            for bid, b in mine[1].items():
                x, s = F.unit_inputs(G, scal, b, 2, 0.0)
                units.append((bid, 2, b)); xs.append(x); ss.append(s)
        meta.append((units, info))
    if not xs:
        return [{} for _ in items]
    logits = pol.forward(np.stack(xs).astype(np.float32), np.stack(ss).astype(np.float32))
    out, o = [], 0
    saved = A._stall
    for (tr, obs, stall), (units, info) in zip(items, meta):
        n = len(units)
        if n == 0:
            out.append({}); continue
        A._stall = stall
        try:
            acts, _ = A.select_actions(units, logits[o:o + n], obs, info, cap)
        finally:
            A._stall = saved
        out.append(acts); o += n
    return out


def hunter_override(sim, me, acts_en, R=3):
    """pessimistic opponent: every enemy collector with a heavier collector of ours within R steps
    greedily closes in on the nearest such target (move minimizing distance)."""
    ours = [w for w in sim.workers.values() if w.player == me and w.is_collector]
    for w in sim.workers.values():
        if w.player == me or not w.is_collector:
            continue
        tg = [o.pos for o in ours if o.carbon > w.carbon and int(F.tdist(w.pos, o.pos)) <= R]
        if not tg:
            continue
        best, bm = 99, None
        for m in F.MOVES:
            c = F.nb(w.pos, m)
            d = min(int(F.tdist(c, t)) for t in tg)
            if d < best:
                best, bm = d, m
        if bm == "NONE":
            acts_en.pop(w.id, None)
        else:
            acts_en[w.id] = bm
    return acts_en


def key_units(obs, cfg):
    me = obs["player"]
    P = obs["players"]
    en = [w for w in P[1 - me][2].values() if w[2] == "COLLECTOR"]
    keys = []
    for uid, w in P[me][2].items():
        pos, cargo, kind = w
        if kind != "COLLECTOR":
            continue
        if cargo > 0:
            thr = [e for e in en if e[1] <= cargo and int(F.tdist(pos, e[0])) <= cfg["KEY_R"]]
            if thr:
                keys.append((cargo, uid))
        elif cfg["HUNT"]:
            prey = [e[1] for e in en if e[1] > 20 and int(F.tdist(pos, e[0])) <= cfg["HUNT_R"]]
            if prey:
                keys.append((max(prey), uid))
    keys.sort(reverse=True)
    return [u for _, u in keys[:cfg["MAX_KEY"]]]


def make_search_agent(conf=None, weights=None, vweights=None):
    cfg = dict(DEFAULTS)
    cfg.update(conf or {})
    weights = weights or os.path.join(HERE, "..", "data", "weights", "nn_master64.npz")
    pol = FastPolicy(A.load_weights(weights))
    opol = FastPolicy(A.load_weights(cfg["OPP_W"])) if cfg["OPP_W"] else pol
    vnet = ValueNet(A.load_weights(vweights)) if vweights else None
    st = {"tr": None, "prev_my": {}}
    pool = [FastCarbon() for _ in range(10)]
    stats = {"steps": 0, "searched": 0, "units": 0, "changed": 0, "branch_steps": 0, "worst_ms": 0.0,
             "gain": 0.0, "errors": 0, "filtered": 0}
    log = []

    def leaf_mat(sim, me):
        m = [float(sim.cash[0]), float(sim.cash[1])]
        for w in sim.workers.values():
            m[w.player] += 0.9 * w.carbon
        return m[me] - m[1 - me]

    def leaf(sim, me, left):
        m = [float(sim.cash[0]), float(sim.cash[1])]
        wv = cfg["WORKER_V"] * min(1.0, left / 100.0)
        for w in sim.workers.values():
            m[w.player] += cfg["CARGO_W"] * w.carbon + wv
        if cfg["TREE_V"]:
            for t in sim.trees.values():
                m[t.player] += cfg["TREE_V"] * max(0, min(50 - t.age, left))
        return m[me] - m[1 - me]

    def agent(obs, config):
        t0 = time.time()
        me = obs["player"]
        step = obs["step"]
        if step == 0 or st["tr"] is None:
            st["tr"] = F.Tracker(me); st["prev_my"] = {}; st["tro"] = F.Tracker(1 - me)
            for key in [k_ for k_ in A._stall if k_[0] == me]:
                del A._stall[key]
        tr = st["tr"]
        tr.update(obs)
        # opponent belief tracker: its own visibility, carbon values from our belief
        obs_o = dict(obs); obs_o["player"] = 1 - me
        vis = [0.0] * F.N
        for w in obs["players"][1 - me][2].values():
            for c in ((w[0],) if w[2] == "COLLECTOR" else F.NB4[w[0]]):
                vis[int(c)] = float(tr.bel[c])
        obs_o["carbon"] = vis
        st["tro"].update(obs_o)
        acts = batch_policy(pol, [(tr, obs, A._stall)], A._cfg(config, "workerLimit", 10))[0]
        prev_my = st["prev_my"]
        st["prev_my"] = {wid: w[0] for wid, w in obs["players"][me][2].items()}
        stats["steps"] += 1
        left = 299 - step
        if not cfg["SEARCH"] or left < cfg["ENDGAME_STOP"]:
            return acts
        try:
            return _search(obs, acts, prev_my, left, t0)
        except Exception as e:  # never let the search layer lose a game
            stats['errors'] += 1; stats['last_error'] = repr(e)[:200]
            return acts

    def _search(obs, acts, prev_my, left, t0):
        me = obs['player']; step = obs['step']; tr = st['tr']
        keys = key_units(obs, cfg)
        if not keys:
            return acts
        stats["searched"] += 1
        H = min(cfg["H"], left + 1)
        # root full state (belief carbon map) and opponent tracker fed with our belief
        full = {"step": step, "players": obs["players"], "trees": obs.get("trees") or {},
                "full_carbon": [float(v) for v in tr.bel], "carbon": obs["carbon"],
                "remainingOverageTime": 60}
        for p in (0, 1):
            for b in obs["players"][p][1].values():
                full["full_carbon"][b] = 0.0
        tro = tr_copy(st["tro"])
        obs_o = dict(obs); obs_o["player"] = 1 - me
        acts_o = batch_policy(opol, [(tro, obs_o, {})])[0]
        cur = dict(acts)
        for uid in keys:
            if time.time() - t0 > cfg["BUDGET"]:
                break
            stats["units"] += 1
            cands = []
            for m in F.MOVES:
                a = dict(cur)
                if m == "NONE":
                    a.pop(uid, None)
                else:
                    a[uid] = m
                cands.append(a)
            dflt = next(i for i, a in enumerate(cands) if a == cur)
            kpos, kcargo = obs["players"][me][2][uid][0], obs["players"][me][2][uid][1]
            allowed = [True] * len(cands)
            if cfg["SAFE_FILTER"] and kcargo > 0:
                # hard shield: never switch to a cell an enemy collector with cargo <= ours can enter next step
                for ci, m in enumerate(F.MOVES):
                    tgt = F.nb(kpos, m)
                    if ci != dflt and any(w[2] == "COLLECTOR" and w[1] <= kcargo and
                                          (w[0] == tgt or tgt in F.NB4[w[0]])
                                          for w in obs["players"][1 - me][2].values()):
                        allowed[ci] = False
                if sum(allowed) <= 1:
                    stats["filtered"] += 1
                    continue
            kpos = obs["players"][me][2][uid][0]
            local = {w_: 1 for p in (0, 1) for w_, w in obs["players"][p][2].items()
                     if int(F.tdist(kpos, w[0])) <= cfg["LOCAL_R"]} if cfg["LOCAL_R"] else None
            sims, trs, stalls = [], [], []
            models = (0, 1) if cfg["ROBUST"] else (0,)
            bmodel = []
            for mdl in models:
              for ci, a in enumerate(cands):
                s = pool[len(sims)]; s.load_observation(full)
                s._tid_counter = [c + 500 for c in s._tid_counter]   # seized trees keep foreign ids
                ao0 = hunter_override(s, me, dict(acts_o)) if mdl else acts_o
                bmodel.append(mdl)
                if me == 0:
                    s.step(a, ao0)
                else:
                    s.step(ao0, a)
                sims.append(s)
                t_me, t_op = tr_copy(tr), tr_copy(tro)
                trs.append((t_me, t_op))
                stalls.append(({k_: v for k_, v in A._stall.items() if k_[0] == me}, {}))
            depth = 1
            for h in range(1, H):
                lft = left - h
                if lft < 0 or time.time() - t0 > cfg["HARD"]:
                    break
                depth = h + 1
                it_me, it_op = [], []
                for s, (t_me, t_op), (s_me, s_op) in zip(sims, trs, stalls):
                    om, oo = s.obs(me), s.obs(1 - me)
                    t_me.update(om); t_op.update(oo)
                    it_me.append((t_me, om, s_me)); it_op.append((t_op, oo, s_op))
                if opol is pol:
                    both = batch_policy(pol, it_me + it_op, subset=local)
                    am, ao = both[:len(it_me)], both[len(it_me):]
                else:
                    am = batch_policy(pol, it_me, subset=local)
                    ao = batch_policy(opol, it_op, subset=local)
                for s, x, y, mdl in zip(sims, am, ao, bmodel):
                    if mdl:
                        y = hunter_override(s, me, dict(y))
                    if me == 0:
                        s.step(x, y)
                    else:
                        s.step(y, x)
                stats["branch_steps"] += len(sims)
            lft = max(0, left - depth)
            sc = np.array([leaf(s, me, lft) for s in sims])
            if vnet is not None and cfg["VALUE_W"]:
                Gs, Ss = [], []
                for s, (t_me, _) in zip(sims, trs):
                    om = s.obs(me)
                    tt = tr_copy(t_me); tt.update(om)
                    G, scal, _ = tt.global_planes(om)
                    Gs.append(G.reshape(F.C_PLANES, 15, 15)); Ss.append(scal)
                v = vnet(np.stack(Gs), np.stack(Ss)) * 5000.0
                if vnet.resid:     # net predicts final margin minus material (cash + 0.9 cargo): add it back
                    v = v + np.array([leaf_mat(s, me) for s in sims])
                v = v.reshape(len(models), len(cands))
                sc = sc + cfg["VALUE_W"] * (v - v[:, dflt:dflt + 1]).reshape(-1)
            if len(models) > 1:      # pessimistic over opponent models
                sc = sc.reshape(len(models), len(cands)).min(axis=0)
            sc = np.where(allowed, sc, -1e18)
            j = int(np.argmax(sc))
            if cfg["LOG"]:
                log.append((step, uid, float(kcargo), dflt, j, [round(float(x)) for x in sc]))
            if j != dflt and sc[j] > sc[dflt] + cfg["MARGIN"]:
                stats["changed"] += 1
                stats["gain"] += float(sc[j] - sc[dflt])
                cur = cands[j]
        dt = (time.time() - t0) * 1000
        stats["worst_ms"] = max(stats["worst_ms"], dt)
        return cur

    agent.stats = stats
    agent.log = log
    return agent
