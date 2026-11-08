"""cnet2_search -- test-time rollout search on top of the cnet2 (v12) policy.

Per real step:
  1. Default joint action = v12 exactly (cnet2 b128, teacher 1, greedy, no rules).
  2. Trigger: "key" collectors -- one of ours within KEY_R (torus manhattan) of an enemy collector where
     the lighter one could capture the heavier one's cargo (ours loaded and threatened, or ours lighter next
     to a loaded enemy = hunt chance).  Stake = the larger cargo; at most MAX_KEY units, biggest stake first.
  3. Branches: for each key unit and each of its other moves, the joint action is re-decoded by the b128
     decoder with the prefix (earlier slots) forced to the default and the unit forced to the move, so the
     later units respond to it (autoregressive).  The opponent's root action is predicted by b128 from its
     own seat (tracked belief Feat fed our carbon belief at its visible cells).
  4. Every branch is a FastCarbon built from our observation + our carbon belief; after the root step both
     sides are played for up to H-1 more steps by the rollout net (default s64) from their own seat,
     batched across branches and both seats (one encode + one lock-step decode per rollout step).
  5. Leaf (ours - theirs): cash + CARGO_W * deliverable cargo + WORKER_V * workers + TREE_V * tree life.
     The best branch replaces the default only if it beats the default by > MARGIN.
Wall-clock guards: no rollout step starts if the projected time passes BUDGET; any exception -> default.
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
if "__file__" in globals() and HERE not in sys.path:
    sys.path.insert(0, HERE)
import cnet2_feat as F  # noqa: E402
import cnet2_np as NP  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

NEG = NP.NEG

DEFAULTS = dict(
    H=5, KEY_R=2, MAX_KEY=3, MINC=10.0, HUNT=True, MARGIN=20.0, BUDGET=0.38, MIN_DEPTH=3,
    CARGO_W=0.9, WORKER_V=30.0, TREE_V=2.0, ENDGAME_STOP=1, TEACHER=1, OPP_TEACHER=1, ROLL="s64",
    PLANT=False, LOG=False, SEARCH=True, MAX_BR=16, LOCAL_R=0, ROBUST=0.0, OPPID=False, MODE="sweep", MIX=False, ONLINE_T=False, TEMPS=(0.7, 1.0, 1.4, 2.0), EPS=1e-3, C_PUCT=1.5, OPEN_T=0.0, OPEN_STEPS=50, OPEN_SEED=0, LEAF_Q=None, GUARD=False, LOWW=2, GUARD_NARROW=True, GUARD_CASH=40.0, RISK_TOL=None, START_DL=None, ABORT_DL=None, MIN_N=1, SE_K=0.0, VS=100.0, ROOT_S=12, RSEED=0, STYLES=(0, 1, 2, 3, 4), ID_DECAY=0.97, ID_PRIOR=3.0, ID_MIN=5, PROJ=1.3, CLOCK="wall", HARD=0.6,
)


def load_weights(src):
    if isinstance(src, dict):
        return src
    d = np.load(src, allow_pickle=False)
    return {k: d[k] for k in d.files}


class CachePolicy(NP.Policy):
    """NP.Policy that keeps the last encode() output (same numerics as v12)."""

    def encode(self, x, teacher):
        r = NP.Policy.encode(self, x, teacher)
        self.last_enc = r
        return r


# ---------------------------------------------------------------- batched inference
def _lin(pol, x, n):
    return x @ pol.w[n + ".weightT"] + pol.w[n + ".bias"]


def _gn(pol, x, n, relu=False):
    """GroupNorm via per-channel moments (2 passes + 1 affine pass)."""
    B, N_, C = x.shape
    G = pol.G
    s1 = x.sum(1).reshape(B, G, C // G).sum(-1)
    s2 = np.einsum("bnc,bnc->bc", x, x).reshape(B, G, C // G).sum(-1)
    cnt = float(N_ * (C // G))
    m = s1 / cnt
    v = np.maximum(s2 / cnt - m * m, 0)
    inv = 1.0 / np.sqrt(v + 1e-5)
    sc = np.repeat(inv, C // G, 1) * pol.w[n + ".weight"]
    sh = pol.w[n + ".bias"] - np.repeat(m, C // G, 1) * sc
    y = x * sc[:, None, :]
    y += sh[:, None, :]
    if relu:
        np.maximum(y, 0, out=y)
    return y


def _ln(pol, x, n):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * pol.w[n + ".weight"] + pol.w[n + ".bias"]


_IMC = {}


def _conv(pol, x, n):
    B, N_, C = x.shape
    idx = _IMC.get(B)
    if idx is None:
        idx = _IMC[B] = (np.arange(B)[:, None, None] * N_ + NP.IM2COL[None]).reshape(-1)
    cols = x.reshape(B * N_, C)[idx].reshape(B * N_, 9 * C)
    return (cols @ pol.wm[n] + pol.w[n + ".bias"]).reshape(B, N_, -1)


def _stack(xs, k):
    return np.stack([x[k] for x in xs])


def tok_features_b(X):
    kind = np.eye(6, dtype=np.float32)[X["tok_kind"]]
    lc = np.minimum(np.log1p(np.maximum(X["tok_cargo"], 0)) / F.LOGC, 1.5).astype(np.float32)
    pos = X["tok_pos"]
    r = (pos // F.S) / 14.0; c = (pos % F.S) / 14.0
    hc = X["hist_cell"]
    valid = (hc >= 0).astype(np.float32)
    hcc = np.maximum(hc, 0)
    dr = ((hcc // F.S) - (pos // F.S)[..., None]) % F.S
    dc = ((hcc % F.S) - (pos % F.S)[..., None]) % F.S
    dr = np.where(dr > 7, dr - F.S, dr) / 7.0
    dc = np.where(dc > 7, dc - F.S, dc) / 7.0
    hl = np.minimum(np.log1p(np.maximum(X["hist_cargo"], 0)) / F.LOGC, 1.5)
    mv = np.eye(6, dtype=np.float32)[X["hist_move"]]
    h = np.concatenate([valid[..., None], (dr * valid)[..., None], (dc * valid)[..., None],
                        (hl * valid)[..., None], mv * valid[..., None]], -1)
    B = pos.shape[0]
    return np.concatenate([kind, lc[..., None], r[..., None], c[..., None], h.reshape(B, F.T, -1)],
                          -1).astype(np.float32)


def encode_b(pol, X, teacher):
    """X: dict of stacked x fields [B,...]; teacher: [B] ints -> tok [B,T,D], ctx [B,Dc]."""
    tok, pre = trunk_b(pol, X)
    return tok, ctx_b(pol, pre, teacher)


def ctx_b(pol, pre, teacher):
    """style enters only here (after the trunk/tokens): pre [B,K] + teach[teacher] -> ctx."""
    w = pol.w
    return np.maximum(_lin(pol, np.concatenate([pre, w["teach.weight"][teacher]], -1), "ctx"), 0).astype(np.float32)


def trunk_b(pol, X):
    """style-independent part of encode: -> tok [B,T,D], pre = [h.mean, tmean, scal-mlp] [B,K]."""
    w = pol.w
    m = (X["maps"].astype(np.float32) / np.float32(255.0)).transpose(0, 2, 1).copy()   # [B,N,P]
    h = _gn(pol, _conv(pol, m, "stem"), "stem_n", True)
    for i in range(pol.NB):
        n = f"blocks.{i}"
        t = _gn(pol, _conv(pol, h, n + ".c1"), n + ".n1", True)
        h = h + _gn(pol, _conv(pol, t, n + ".c2"), n + ".n2")
        np.maximum(h, 0, out=h)
    B = h.shape[0]
    pos = X["tok_pos"]
    tf = tok_features_b(X)
    t = _lin(pol, np.maximum(_lin(pol, tf, "tok1"), 0), "tok2")
    tok = _lin(pol, np.concatenate([h[np.arange(B)[:, None], pos], t], -1), "fuse")
    mask = X["tok_mask"]
    mf_ = mask.astype(np.float32)[..., None]
    tok = tok * mf_
    r = pos // F.S; c = pos % F.S
    rel = ((r[:, :, None] - r[:, None, :]) % F.S) * F.S + (c[:, :, None] - c[:, None, :]) % F.S
    D = tok.shape[-1]; NH = pol.NH; dh = D // NH
    for li in range(pol.L):
        n = f"attn.{li}"
        qkv = _lin(pol, tok, n + ".qkv").reshape(B, F.T, 3, NH, dh).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        sc = q @ k.transpose(0, 1, 3, 2) / np.float32(dh ** 0.5) + w[n + ".rel.weight"][rel].transpose(0, 3, 1, 2)
        sc = np.where(mask[:, None, None, :], sc, NEG)
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        sc /= sc.sum(-1, keepdims=True)
        a = (sc @ v).transpose(0, 2, 1, 3).reshape(B, F.T, D)
        tok = _ln(pol, tok + _lin(pol, a, n + ".out"), n + ".n1")
        tok = _ln(pol, tok + _lin(pol, np.maximum(_lin(pol, tok, n + ".f1"), 0), n + ".f2"), n + ".n2")
        tok = tok * mf_
    tmean = tok.sum(1) / np.maximum(mask.sum(1), 1)[:, None]
    s = _lin(pol, np.maximum(_lin(pol, X["scal"], "sc1"), 0), "sc2")
    return tok.astype(np.float32), np.concatenate([h.mean(1), tmean, s], -1)


def _gru_b(pol, x, h):
    w = pol.w
    gi = x @ pol.gih + w["gru.bias_ih"]
    gh = h @ pol.ghh + w["gru.bias_hh"]
    D = h.shape[1]
    r = 1 / (1 + np.exp(-(gi[:, :D] + gh[:, :D])))
    z = 1 / (1 + np.exp(-(gi[:, D:2 * D] + gh[:, D:2 * D])))
    n = np.tanh(gi[:, 2 * D:] + r * gh[:, 2 * D:])
    return (1 - z) * n + z * h


AUXK = ("est", "en_min", "en_max", "en_pl", "thr", "dob", "own_base", "en_base", "own_tree", "en_tree",
        "toll", "own_adj")
_M0 = (np.arange(5) == 0).astype(np.float32)


def decode_b(pol, X, auxs, tok, ctx, forced=None, logp=None, noise=None, temp=None, lg_out=None):
    """Lock-step greedy autoregressive decode of B boards.  forced: [B, OWN] int (-1 = free).
    Mirrors NP.Policy.act + F.Prefix (vectorised; numerics equal up to float summation order)."""
    w = pol.w
    B = tok.shape[0]
    A = {k: np.stack([a[k] for a in auxs]).astype(np.float64) for k in AUXK}
    mask = X["tok_mask"]; tpos = X["tok_pos"]; tkind = X["tok_kind"]; tcargo = X["tok_cargo"].astype(np.float64)
    left = np.array([x for x in X["left"]], np.float64)
    reserved = np.zeros((B, F.N)); undecided = np.zeros((B, F.N))
    for s in range(10):
        ii = np.nonzero(mask[:, s])[0]
        np.add.at(undecided, (ii, tpos[ii, s]), 1)
    cash = np.array(X["cash"], np.float64); slots = np.array(X["slots"], np.float64); plants = np.zeros(B)
    h = _lin(pol, ctx, "h0")
    prev = np.full(B, 8, np.int64)
    labels = np.full((B, F.OWN), -1, np.int64)
    for s in range(F.OWN):
        idx = np.nonzero(mask[:, s])[0]
        n = len(idx)
        if n == 0:
            continue
        pos = tpos[idx, s]
        uf = np.zeros((n, F.UF)); mf = np.zeros((n, 5, F.MF)); legal = np.ones((n, 5), bool)
        uf[:, 6] = np.log1p(np.maximum(cash[idx], 0)) / 9
        uf[:, 7] = slots[idx] / 10
        uf[:, 8] = np.minimum(plants[idx], 3) / 3
        if s < 10:
            isc = tkind[idx, s] == 0
            cargo = tcargo[idx, s]
            undecided[idx, pos] -= 1
            uf[:, 0] = isc; uf[:, 1] = ~isc
            uf[:, 3] = np.minimum(np.log1p(np.maximum(cargo, 0)) / F.LOGC, 1.5)
            dob0 = A["dob"][idx, pos]
            uf[:, 4] = dob0 / 14.0
            uf[:, 5] = (left[idx] - dob0) <= 1
            cells = F.DEST[pos]                     # [n,5]
            stay = np.broadcast_to(_M0, (n, 5))
            cg = np.broadcast_to(cargo[:, None], (n, 5))
            iscb = np.broadcast_to(isc[:, None], (n, 5))
        else:
            uf[:, 2] = 1
            ok = (cash[idx] >= 30) & (slots[idx] > 0)
            legal[:, 1] = ok; legal[:, 2] = ok; legal[:, 3] = False; legal[:, 4] = False
            cells = np.stack([pos] * 5, 1)
            stay = np.ones((n, 5))
            cg = np.zeros((n, 5))
            iscb = np.broadcast_to(np.array([True, True, False, True, True]), (n, 5))
        ii = idx[:, None]
        emin = A["en_min"][ii, cells]; emax = A["en_max"][ii, cells]; epl = A["en_pl"][ii, cells]
        f = mf
        f[..., 0] = np.minimum(reserved[ii, cells], 2) / 2
        f1c = emin <= cg
        f1p = np.isfinite(emin) | (epl > 0)
        f[..., 1] = np.where(iscb, f1c, f1p)
        f[..., 2] = np.where(iscb & (emax > cg), np.minimum(np.log1p(np.maximum(emax, 0)) / F.LOGC, 1.5), 0.0)
        f[..., 3] = np.where(iscb, epl, 0.0)
        f[..., 4] = np.minimum(A["thr"][ii, cells], 4) / 4
        f[..., 5] = A["own_base"][ii, cells]; f[..., 6] = A["en_base"][ii, cells]
        f[..., 7] = A["est"][ii, cells]
        f[..., 8] = A["own_tree"][ii, cells]; f[..., 9] = A["en_tree"][ii, cells]
        f[..., 10] = A["dob"][ii, cells] / 14.0
        f[..., 11] = stay
        f[..., 12] = np.minimum(undecided[ii, cells], 2) / 2
        f[..., 13] = np.minimum(A["toll"][ii, cells], 4) / 4
        f[..., 14] = A["own_adj"][ii, cells] / 4
        if s >= 10:
            f[:, 0] = 0; f[:, 3:] = 0
        uf = F.f16(uf); mf = F.f16(mf)
        inp = _lin(pol, np.concatenate([tok[idx, s], ctx[idx], uf, mf.reshape(n, -1)], 1), "din") + \
            w["aemb.weight"][prev[idx]]
        hn = _gru_b(pol, inp, h[idx])
        h[idx] = hn
        if s < 10:
            base = _lin(pol, hn, "whead")
        else:
            base = np.concatenate([_lin(pol, hn, "bhead"), np.zeros((n, 2), np.float32)], 1)
        lg = base + np.einsum("nme,ne->nm", _lin(pol, mf, "V"), _lin(pol, hn, "U"))
        lg = np.where(legal, lg, NEG)
        if lg_out is not None:
            lg_out[idx, s] = lg
        a = lg.argmax(1)
        if noise is not None:
            tr = temp[idx]
            sm = tr > 0
            if sm.any():
                z = lg[sm] / tr[sm][:, None]
                z = np.exp(z - z.max(1, keepdims=True)) * legal[sm]
                pz = z / z.sum(1, keepdims=True)
                cz = np.cumsum(pz, 1)
                u = noise[idx[sm], s][:, None] * cz[:, -1:]
                a = a.copy()
                a[sm] = np.minimum((cz < u).sum(1), 4)
        if forced is not None:
            fz = forced[idx, s]
            a = np.where(fz >= 0, fz, a)
        if logp is not None:
            z = lg - lg.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            logp[idx, s] = lp[np.arange(n), a]
        labels[idx, s] = a
        if s < 10:
            d = F.DEST[pos, a]
            reserved[idx, d] += 1
            pl = (a == 0) & (tkind[idx, s] == 1)
            if pl.any():
                pl &= ~(A["own_tree"][idx, pos] > 0) & ~(A["own_base"][idx, pos] > 0) & ~(A["en_base"][idx, pos] > 0)
                plants[idx] += pl
            prev[idx] = a
        else:
            r = a > 0
            if r.any():
                reserved[idx[r], pos[r]] += 1
                cash[idx[r]] -= 30
                slots[idx[r]] -= 1
            prev[idx] = a + 5
    return labels


def stack_x(xs):
    X = {k: _stack(xs, k) for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
                                    "hist_cell", "hist_cargo", "hist_move")}
    X["cash"] = [x["cash"] for x in xs]; X["slots"] = [x["slots"] for x in xs]; X["left"] = [x["left"] for x in xs]
    return X


def act_b(pol, xs, auxs, teachers, forced=None):
    X = {k: _stack(xs, k) for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
                                    "hist_cell", "hist_cargo", "hist_move")}
    X["cash"] = [x["cash"] for x in xs]; X["slots"] = [x["slots"] for x in xs]; X["left"] = [x["left"] for x in xs]
    tok, ctx = encode_b(pol, X, np.asarray(teachers))
    return decode_b(pol, X, auxs, tok, ctx, forced)


# ---------------------------------------------------------------- helpers
def feat_copy(f):
    n = F.Feat.__new__(F.Feat)
    n.seat = f.seat; n.step = f.step
    n.est = f.est.copy(); n.last = f.last.copy(); n.last_step = f.last_step.copy()
    n.prev_w = f.prev_w; n.prev_trees = f.prev_trees
    n.hist = {k: list(v) for k, v in f.hist.items()}
    n.own_uid = list(getattr(f, "own_uid", []))
    n.aux = getattr(f, "aux", None)
    return n


def key_units(obs, cfg):
    """-> list of (stake, uid) for our collectors in a capture situation, biggest stake first."""
    me = obs["player"]; P = obs["players"]
    en = [w for w in P[1 - me][2].values() if w[2] == "COLLECTOR"]
    keys = []
    for uid, w in P[me][2].items():
        pos, cargo, kind = w
        if kind != "COLLECTOR":
            continue
        best = 0.0
        for e in en:
            if F.DIST[pos, e[0]] > cfg["KEY_R"]:
                continue
            if e[1] <= cargo and cargo >= cfg["MINC"]:
                best = max(best, cargo * cfg.get("PRIO_THREAT", 1.0))
            elif cfg["HUNT"] and e[1] > cargo and e[1] >= cfg["MINC"]:
                best = max(best, e[1])
        if best > 0:
            keys.append((best, uid))
    keys.sort(reverse=True)
    if cfg.get("SPLIT"):
        # one escape unit (loaded, threatened) + one hunter (lighter than a loaded enemy), each by stake
        thr, hun = [], []
        for uid, w in P[me][2].items():
            pos, cargo, kind = w
            if kind != "COLLECTOR":
                continue
            t_ = max([cargo for e in en if F.DIST[pos, e[0]] <= cfg["KEY_R"] and e[1] <= cargo and cargo >= cfg["MINC"]], default=0)
            h_ = max([e[1] for e in en if F.DIST[pos, e[0]] <= cfg["KEY_R"] and e[1] > cargo and e[1] >= cfg.get("PREY_MIN", cfg["MINC"])], default=0)
            if t_:
                thr.append((t_, uid))
            if h_:
                hun.append((h_, uid))
        thr.sort(reverse=True); hun.sort(reverse=True)
        out = thr[:1]
        for h_ in hun:
            if h_[1] not in [u for _, u in out]:
                out.append(h_); break
        return out
    mk = cfg["MAX_KEY"]
    if cfg.get("MAX_KEY_LATE") is not None and obs["step"] >= cfg.get("KEY_LATE_STEP", 999):
        mk = cfg["MAX_KEY_LATE"]          # endgame: several loaded units run home at once -> search more of them
    return keys[:mk]


def hunter_acts(obs, me, acts_o, cfg):
    """aggressive opponent variant: each enemy collector within 2 of a heavier collector of ours (cargo >= MINC)
    moves to minimise the distance to it (ties keep the predicted move)."""
    P = obs["players"]
    ours = [w for w in P[me][2].values() if w[2] == "COLLECTOR" and w[1] >= cfg["MINC"]]
    out = dict(acts_o)
    for eid, e in P[1 - me][2].items():
        if e[2] != "COLLECTOR":
            continue
        tg = [o[0] for o in ours if o[1] > e[1] and F.DIST[e[0], o[0]] <= 2]
        if not tg:
            continue
        cur = acts_o.get(eid, "NONE")
        best, bm = 99, cur
        for m in [cur] + [m_ for m_ in F.MOVES if m_ != cur]:
            c = int(F.DEST[e[0], F.MOVES.index(m)])
            d = min(int(F.DIST[c, t]) for t in tg)
            if d < best:
                best, bm = d, m
        if bm == "NONE":
            out.pop(eid, None)
        else:
            out[eid] = bm
    return out


def make_search_agent(conf=None, weights=None, rweights=None):
    cfg = dict(DEFAULTS)
    cfg.update(conf or {})
    wdir = os.path.join(HERE, "..", "data", "weights")
    weights = weights if weights is not None else os.path.join(wdir, "cnet2_b128_q8.npz")
    pol = CachePolicy(load_weights(weights))
    if rweights is None and cfg["ROLL"] == "s64":
        rweights = os.path.join(wdir, cfg.get("RW", "cnet2_s64.npz"))
    rpol = NP.Policy(load_weights(rweights)) if rweights is not None else pol
    xw = cfg.get("XNET")
    if isinstance(xw, str) and not os.path.isabs(xw):
        xw = os.path.join(HERE, "..", xw)
    xpol = NP.Policy(load_weights(xw)) if xw is not None else None      # extra (structurally different) mixture component
    XT = int(cfg.get("XNET_T", 1))
    st = {"f": None, "fo": None}
    pool = []
    stats = {"steps": 0, "searched": 0, "branches": 0, "changed": 0, "branch_steps": 0, "worst_ms": 0.0,
             "gain": 0.0, "errors": 0, "shallow": 0, "depth_sum": 0}
    log = []
    T, OT = cfg["TEACHER"], cfg["OPP_TEACHER"]

    def leaf(sim, me, left):
        m = [float(sim.cash[0]), float(sim.cash[1])]
        wv = cfg["WORKER_V"] * min(1.0, left / 100.0)
        bases = {p: [b for _, b in sim.bases[p]] for p in (0, 1)}
        for w in sim.workers.values():
            v = wv
            if w.carbon > 0:
                bs = bases[w.player]
                d = min(F.DIST[w.pos, b] for b in bs) if bs else 99
                if d <= left:
                    v += cfg["CARGO_W"] * w.carbon
            m[w.player] += v
        if cfg["TREE_V"]:
            for t in sim.trees.values():
                m[t.player] += cfg["TREE_V"] * max(0, min(50 - t.age, left))
        return m[me] - m[1 - me]

    clk = time.process_time if cfg["CLOCK"] == "cpu" else time.time

    def agent(obs, config=None):
        t0 = clk(); tw = time.time()
        me = obs["player"]
        if st["f"] is None or obs["step"] == 0 or st["f"].seat != me:
            st["f"] = F.Feat(me); st["fo"] = F.Feat(1 - me)
        f, fo = st["f"], st["fo"]
        if obs["step"] == 0 or "lpost" not in st:
            st["lpost"] = np.array([cfg["ID_PRIOR"] if k == OT else 0.0 for k in cfg["STYLES"]])
            st["ot"] = OT; st["oprev"] = None; st["pw"] = None; st["ntrk"] = 0; st["lostw"] = 0; st["prevw"] = set()
            KS, KT = len(cfg["STYLES"]) + (1 if xpol is not None else 0), len(cfg["TEMPS"])
            st["L"] = np.zeros((KS, KT)); st["M"] = np.zeros(KT)
            st["lprior"] = np.array([cfg["ID_PRIOR"] if k == OT else 0.0 for k in cfg["STYLES"]] + ([0.0] if xpol is not None else []))
            st["ti"] = int(np.argmin(np.abs(np.array(cfg["TEMPS"]) - 1.0))); st["t1"] = st["ti"]
            st["ow"] = np.exp(st["lprior"] - st["lprior"].max()); st["ow"] /= st["ow"].sum()
        x = f.observe(obs)
        st["x"] = x; st["rec"] = None
        if obs["step"] == 0 or "orng" not in st:
            st["orng"] = np.random.default_rng(int(cfg["OPEN_SEED"]) * 7 + me)
        st["sampled"] = bool(cfg["OPEN_T"] and obs["step"] < cfg["OPEN_STEPS"])
        if st["sampled"]:
            # opening diversity (data generation): sample the net's moves at temperature OPEN_T
            y, lg0, _ = pol.act(x, f.aux, teacher=T, sample_rng=st["orng"], temp=cfg["OPEN_T"])
        else:
            # phase style switch: TEACHER early, TEACHER_LATE from step LATE_STEP on (default off)
            TL = cfg.get("TEACHER_LATE")
            # Master-line detection: the opponent's FIRST recruit is a collector (CC...) -> switch styles earlier
            if obs["step"] <= 12 and "opp_first" not in st:
                ow = [w for w in obs["players"][1 - me][2].values()]
                if ow:
                    st["opp_first"] = ow[0][2]
            if obs["step"] == 0:
                st.pop("opp_first", None)
            ls = cfg.get("LATE_STEP", 999)
            if cfg.get("LATE_STEP_MASTER") is not None and st.get("opp_first") == "COLLECTOR":
                ls = cfg["LATE_STEP_MASTER"]
            st["T_now"] = TL if (TL is not None and obs["step"] >= ls) else T   # rollouts use the same slot
            y, lg0, _ = pol.act(x, f.aux, teacher=st["T_now"])
        st["lg0"] = lg0; st["y0"] = y.copy()
        if cfg["GUARD"]:
            try:
                y = bankrupt_guard(x, f.aux, y, lg0)
            except Exception as e:
                stats["errors"] += 1; stats["last_error"] = repr(e)[:300]
        acts = f.to_actions(y)
        stats["steps"] += 1
        try:
            track_losses(obs)
        except Exception as e:
            stats["errors"] += 1; stats["last_error"] = repr(e)[:300]
        try:
            # opponent belief tracker (its seat, its visibility, our carbon belief)
            obs_o = dict(obs); obs_o["player"] = 1 - me
            vis = np.zeros(F.N)
            for wv in obs["players"][1 - me][2].values():
                cs = [wv[0]] if wv[2] == "COLLECTOR" else F.DEST[wv[0], 1:]
                vis[cs] = f.est[cs]
            obs_o["carbon"] = vis.tolist()
            xo = fo.observe(obs_o)
            st["oenc"] = None
            if cfg["MIX"] or cfg["MODE"] == "mcts":
                opp_model(obs, xo)
            elif cfg["OPPID"]:
                opp_id(obs, xo)
            left = 299 - obs["step"]
            if not cfg["SEARCH"] or left < cfg["ENDGAME_STOP"]:
                return acts
            if cfg["MODE"] == "mcts":
                out = _mcts(obs, x, y, acts, xo, left, t0, tw)
            else:
                out = _search(obs, x, y, acts, xo, left, t0, tw)
            stats["worst_ms"] = max(stats["worst_ms"], (clk() - t0) * 1000)
            stats["worst_wall_ms"] = max(stats.get("worst_wall_ms", 0.0), (time.time() - tw) * 1000)
            return out
        except Exception as e:  # never let the search layer lose a game
            stats["errors"] += 1; stats["last_error"] = repr(e)[:300]
            return acts

    def f_own_uids():
        return [u for u in st["f"].own_uid[:10] if u is not None]

    def bankrupt_guard(x, aux, y, lg0):
        """anti-bankruptcy: (1) with no own collector (and none recruited this step) a base recruits a collector
        when cash allows; (2) with <= LOWW own workers, a unit never moves/stays where an enemy unit that would
        kill it can arrive (planter: any enemy unit; collector: enemy collector with cargo <= ours), if a safe
        move exists (best logit among safe).  Later slots are re-decoded after the first forced change."""
        m = x["tok_mask"]; kind = x["tok_kind"]; pos = x["tok_pos"]; cargo = x["tok_cargo"]
        nw = int(sum(1 for s_ in range(10) if m[s_]))
        ncol = int(sum(1 for s_ in range(10) if m[s_] and kind[s_] == 0))
        fz = np.full(F.OWN, -1, np.int64); first = None
        cur = {u for u in f_own_uids()}
        st["lostw"] = st.get("lostw", 0) + len(st.get("prevw", set()) - cur)
        st["prevw"] = cur
        risky = nw <= cfg["LOWW"]
        if cfg["GUARD_NARROW"]:
            # only a real bankruptcy risk: we already lost a unit and a further death could not be re-bought
            risky = risky and st["lostw"] > 0 and x["cash"] < cfg["GUARD_CASH"]
        if risky:
            for s_ in range(10):
                if not m[s_]:
                    continue
                d = F.DEST[int(pos[s_])]
                if kind[s_] == 0:
                    bad = aux["en_min"][d] <= float(cargo[s_])
                else:
                    bad = aux["thr"][d] > 0
                if bad[y[s_]] and not bad.all():
                    ok = [k for k in np.argsort(-lg0[s_]) if not bad[k]]
                    fz[s_] = int(ok[0]); first = s_ if first is None else first
                    stats["guard_moves"] = stats.get("guard_moves", 0) + 1
                    if cfg.get("GUARD_LOG"):
                        p_ = int(pos[s_]); en = [(int(F.DIST[p_, int(pos[e])]), int(kind[e]), float(cargo[e])) for e in range(12, 22) if m[e]]
                        stats.setdefault("guard_log", []).append({
                            "step": 299 - int(x["left"]), "nw": nw, "cash": round(float(x["cash"])), "lost": int(st.get("lostw", 0)),
                            "uid": st["f"].own_uid[s_], "kind": "C" if kind[s_] == 0 else "P", "cargo": round(float(cargo[s_])),
                            "dob": int(aux["dob"][p_]), "nearest_enemy": min(en) if en else None,
                            "net_move": int(y[s_]), "chosen": int(ok[0])})
        if ncol == 0 and x["cash"] >= 30 and x["slots"] > 0 and x["left"] > 10:
            bs = [s_ for s_ in (10, 11) if m[s_]]
            if bs and not any(y[s_] == 1 for s_ in bs):
                fz[bs[0]] = 1; first = bs[0] if first is None else min(first, bs[0])
                stats["guard_recruit"] = stats.get("guard_recruit", 0) + 1
        if first is None:
            return y
        fz[:first] = y[:first]
        tok, ctx = pol.last_enc
        X1 = {k: x[k][None] for k in ("tok_pos", "tok_kind", "tok_mask", "tok_cargo")}
        X1["cash"] = [x["cash"]]; X1["slots"] = [x["slots"]]; X1["left"] = [x["left"]]
        y2 = decode_b(pol, X1, [aux], tok[None], ctx[None], fz[None])[0]
        return y2

    def track_losses(obs):
        """our / their collectors that vanished while carrying cargo (caught), by load."""
        me = obs["player"]; P = obs["players"]
        cur = {u: (p, w[1]) for p in (0, 1) for u, w in P[p][2].items()}
        pw = st["pw"]
        if pw is not None:
            for u, (p, c) in pw.items():
                if u not in cur and c > 0:
                    stats.setdefault("lost_ours" if p == me else "lost_theirs", []).append(round(c))
        st["pw"] = {u: v for u, v in cur.items() if P[v[0]][2][u][2] == "COLLECTOR"}

    def opp_id(obs, xo):
        """style posterior from the opponent's last observed moves (teacher-forced, all styles batched)."""
        fo = st["fo"]
        prev = st["oprev"]
        if prev is not None:
            Xp, tokp, prep, auxp, uidp, wp = prev
            lab = np.full(F.OWN, -1, np.int64)
            for s_, u in enumerate(uidp):
                if u is None:
                    continue
                if s_ < 10:
                    h = fo.hist.get(u)
                    if h is not None and h[-1][2] < 5 and u in wp:
                        lab[s_] = h[-1][2]
                else:
                    bpos = int(Xp["tok_pos"][0, s_]); a = 0
                    for u2, h in fo.hist.items():
                        if u2 not in wp and len(h) == 1 and h[0][0] == bpos:
                            a = 1 if obs["players"][1 - obs["player"]][2][u2][2] == "COLLECTOR" else 2
                    lab[s_] = a
            cnt = lab >= 0
            if cnt.any():
                K = len(cfg["STYLES"])
                rep = {k: np.repeat(v, K, 0) for k, v in Xp.items() if isinstance(v, np.ndarray)}
                for k in ("cash", "slots", "left"):
                    rep[k] = Xp[k] * K
                lp = np.zeros((K, F.OWN), np.float64)
                decode_b(pol, rep, [auxp] * K, np.repeat(tokp, K, 0),
                         ctx_b(pol, np.repeat(prep, K, 0), np.array(cfg["STYLES"])),
                         np.repeat(lab[None], K, 0), lp)
                ll = np.maximum(lp[:, cnt], -12.0).sum(1)
                st["lpost"] = cfg["ID_DECAY"] * st["lpost"] + ll
                st["ntrk"] += 1
                if st["ntrk"] >= cfg["ID_MIN"]:
                    st["ot"] = int(cfg["STYLES"][int(np.argmax(st["lpost"]))])
        Xo = stack_x([xo])
        toko, preo = trunk_b(pol, Xo)
        st["oenc"] = (Xo, toko, preo)
        wnow = set(obs["players"][1 - obs["player"]][2])
        st["oprev"] = (Xo, toko, preo, fo.aux, list(fo.own_uid), wnow)
        stp = obs["step"]
        if stp in (10, 20, 50, 100, 150, 200, 299):
            z = st["lpost"] - st["lpost"].max(); pz = np.exp(z); pz /= pz.sum()
            stats.setdefault("id", {})[str(stp)] = [st["ot"], [round(float(v), 3) for v in pz]]

    def _labels_prev(obs, prev):
        fo = st["fo"]
        Xp, tokp, prep, auxp, uidp, wp = prev
        lab = np.full(F.OWN, -1, np.int64)
        for s_, u in enumerate(uidp):
            if u is None:
                continue
            if s_ < 10:
                h = fo.hist.get(u)
                if h is not None and h[-1][2] < 5 and u in wp:
                    lab[s_] = h[-1][2]
            else:
                bpos = int(Xp["tok_pos"][0, s_]); a = 0
                for u2, h in fo.hist.items():
                    if u2 not in wp and len(h) == 1 and h[0][0] == bpos:
                        a = 1 if obs["players"][1 - obs["player"]][2][u2][2] == "COLLECTOR" else 2
                lab[s_] = a
        return lab

    def opp_model(obs, xo):
        """online mixture over style heads (+ temperature): weights = decayed likelihood of the opponent's
        observed moves (teacher-forced decode of every style, batched); predictive diagnostics in stats."""
        fo = st["fo"]
        prev = st["oprev"]
        S_ = cfg["STYLES"]; TT = np.array(cfg["TEMPS"], np.float64); K = len(S_)
        if prev is not None:
            Xp, tokp, prep, auxp, uidp, wp = prev[:6]
            lab = _labels_prev(obs, prev[:6])
            cnt = np.nonzero(lab >= 0)[0]
            if len(cnt):
                rep = {k: np.repeat(v, K, 0) for k, v in Xp.items() if isinstance(v, np.ndarray)}
                for k in ("cash", "slots", "left"):
                    rep[k] = Xp[k] * K
                LG = np.full((K, F.OWN, 5), NEG, np.float32)
                decode_b(pol, rep, [auxp] * K, np.repeat(tokp, K, 0), ctx_b(pol, np.repeat(prep, K, 0), np.array(S_)),
                         np.repeat(lab[None], K, 0), lg_out=LG)
                if xpol is not None:
                    tkx, prx = prev[6]
                    LGx = np.full((1, F.OWN, 5), NEG, np.float32)
                    decode_b(xpol, Xp, [auxp], tkx, ctx_b(xpol, prx, np.array([XT])), lab[None], lg_out=LGx)
                    LG = np.concatenate([LG, LGx]); K = K + 1
                lg = LG[:, cnt].astype(np.float64)                     # [K,n,5]
                legal = lg > NEG / 2
                a = lab[cnt]
                nl = legal.sum(-1, keepdims=True)
                z = lg[None] / TT[:, None, None, None]                  # [T,K,n,5]
                z = np.where(legal[None], z, -np.inf)
                z = np.exp(z - z.max(-1, keepdims=True))
                p = z / z.sum(-1, keepdims=True)
                p = (1 - cfg["EPS"]) * p + cfg["EPS"] * legal[None] / nl[None]
                pa = p[:, :, np.arange(len(cnt)), a]                      # [T,K,n]
                ll = np.log(pa).sum(-1).T                                 # [K,T]
                # ---- predictive diagnostics (weights before this update)
                lw = st["L"] + st["lprior"][:, None]                      # [K,T]
                lw = lw - lw.max(0, keepdims=True)
                w = np.exp(lw); w /= w.sum(0, keepdims=True)
                mixll = np.log((w * np.exp(ll - ll.max(0, keepdims=True))).sum(0)) + ll.max(0)   # [T]
                pm = (w.T[:, :, None, None] * p).sum(1)                  # [T,n,5]
                d = stats.setdefault("pred", {"n_units": 0, "n_steps": 0, "a_ll": 0.0, "a_acc": 0,
                                              "b_ll": 0.0, "b_acc": 0, "c_ll": 0.0, "c_acc": 0})
                k1 = list(S_).index(OT) if OT in S_ else 0
                t1, ti = st["t1"], st["ti"]
                d["n_units"] += len(cnt); d["n_steps"] += 1
                d["a_ll"] += float(np.log(pa[t1, k1]).sum()); d["a_acc"] += int((lg[k1].argmax(-1) == a).sum())
                d["b_ll"] += float(mixll[t1]); d["b_acc"] += int((pm[t1].argmax(-1) == a).sum())
                d["c_ll"] += float(mixll[ti]); d["c_acc"] += int((pm[ti].argmax(-1) == a).sum())
                # ---- update
                dec = cfg["ID_DECAY"]
                st["L"] = dec * st["L"] + ll
                st["M"] = dec * st["M"] + mixll
                st["ntrk"] += 1
                if cfg["ONLINE_T"] and st["ntrk"] >= cfg["ID_MIN"]:
                    st["ti"] = int(np.argmax(st["M"]))
                lw = st["L"][:, st["ti"]] + st["lprior"]
                if not cfg["MIX"]:
                    lw = np.where(np.arange(K) == k1, 0.0, -np.inf)
                w = np.exp(lw - lw.max()); st["ow"] = w / w.sum()
        Xo = stack_x([xo])
        toko, preo = trunk_b(pol, Xo)
        st["oenc"] = (Xo, toko, preo)
        st["oprev"] = (Xo, toko, preo, fo.aux, list(fo.own_uid), set(obs["players"][1 - obs["player"]][2]))
        if xpol is not None:
            st["xenc"] = trunk_b(xpol, Xo)
            st["oprev"] = st["oprev"] + (st["xenc"],)
        stp = obs["step"]
        if stp in (10, 20, 50, 100, 150, 200, 299):
            stats.setdefault("id", {})[str(stp)] = [float(cfg["TEMPS"][st["ti"]]), [round(float(v), 3) for v in st["ow"]]]

    def _mcts(obs, x, y, acts, xo, left, t0, tw):
        """sampled PUCT over root candidates (default + single-unit deviations of trigger units).
        A simulation = opponent root reply sampled from the online style mixture (b128), then H-1 steps with
        our side greedy and the opponent sampled (style drawn per simulation, temperature tau) by the rollout
        net.  Simulation index i shares its random numbers across candidates (common random numbers); the
        override decision uses the paired mean difference vs the default over common indices."""
        me = obs["player"]; step = obs["step"]
        f, fo = st["f"], st["fo"]
        keys = key_units(obs, cfg)
        if not keys:
            return acts
        stats["searched"] += 1
        tok, ctx = pol.last_enc
        lg0 = st["lg0"]
        uid2slot = {u: s for s, u in enumerate(f.own_uid) if u is not None}
        forced, meta, pri = [], [], []
        pdef = 0.0
        for _, uid in keys:
            s = uid2slot.get(uid)
            if s is None or s >= 10:
                continue
            z = lg0[s].astype(np.float64); pz = np.exp(z - z.max()); pz /= pz.sum()
            pdef = max(pdef, pz[y[s]])
            for m in range(5):
                if m == y[s]:
                    continue
                fz = np.full(F.OWN, -1, np.int64)
                fz[:s] = y[:s]; fz[s] = m
                forced.append(fz); meta.append((uid, m)); pri.append(pz[m])
        forced = forced[:cfg["MAX_BR"] - 1]; meta = meta[:cfg["MAX_BR"] - 1]
        if not forced:
            return acts
        B0 = len(forced)
        X1 = {k: np.repeat(x[k][None], B0, 0) for k in ("tok_pos", "tok_kind", "tok_mask", "tok_cargo")}
        X1["cash"] = [x["cash"]] * B0; X1["slots"] = [x["slots"]] * B0; X1["left"] = [x["left"]] * B0
        ys = decode_b(pol, X1, [f.aux] * B0, np.repeat(tok[None], B0, 0), np.repeat(ctx[None], B0, 0),
                      np.stack(forced))
        cand = [acts]; seen = {tuple(y)}; cmeta = [None]; P = [pdef]
        for yy, mm, pp in zip(ys, meta, pri):
            k = tuple(yy)
            if k in seen:
                continue
            seen.add(k); cand.append(f.to_actions(yy)); cmeta.append(mm); P.append(pp)
        nc = len(cand)
        if nc < 2:
            return acts
        P = np.array(P); P = P / P.sum()
        # ---- per-simulation-index randomness: style draw, root reply, rollout noise
        rng = np.random.default_rng(cfg["RSEED"] * 1000003 + step * 7919 + me)
        RS = cfg["ROOT_S"]
        if cfg.get("ROOT_S_LATE") is not None and step >= cfg.get("KEY_LATE_STEP", 999):
            RS = cfg["ROOT_S_LATE"]       # endgame: per-unit sims shrink as MAX_KEY_LATE grows (same total budget)
        S_ = np.array(list(cfg["STYLES"]) + ([XT] if xpol is not None else [])); w = st["ow"]
        ci = np.minimum(np.searchsorted(np.cumsum(w), (np.arange(RS) + rng.random(RS)) / RS * w.sum()), len(S_) - 1)
        ks = S_[ci]
        isx = (ci == len(cfg["STYLES"])) if xpol is not None else np.zeros(RS, bool)
        tau = float(cfg["TEMPS"][st["ti"]])
        Xo, toko, preo = st["oenc"]
        rep = {k: np.repeat(v, RS, 0) for k, v in Xo.items() if isinstance(v, np.ndarray)}
        for k in ("cash", "slots", "left"):
            rep[k] = Xo[k] * RS
        rn0 = rng.random((RS, F.OWN))
        yo = decode_b(pol, rep, [fo.aux] * RS, np.repeat(toko, RS, 0), ctx_b(pol, np.repeat(preo, RS, 0), ks),
                      noise=rn0, temp=np.full(RS, tau))
        if isx.any():
            nx = int(isx.sum()); tkx, prx = st["xenc"]
            repx = {k: np.repeat(v, nx, 0) for k, v in Xo.items() if isinstance(v, np.ndarray)}
            for k in ("cash", "slots", "left"):
                repx[k] = Xo[k] * nx
            yo[isx] = decode_b(xpol, repx, [fo.aux] * nx, np.repeat(tkx, nx, 0), ctx_b(xpol, np.repeat(prx, nx, 0), np.full(nx, XT)),
                               noise=rn0[isx], temp=np.full(nx, tau))
        roots = [fo.to_actions(v) for v in yo]
        H = min(cfg["H"], left + 1)
        rnoise = rng.random((RS, max(H - 1, 1), F.OWN))
        full = {"step": step, "players": obs["players"], "trees": obs.get("trees") or {},
                "full_carbon": [float(v) for v in f.est], "carbon": obs["carbon"], "remainingOverageTime": 60}
        vals = [[] for _ in range(nc)]
        n = np.zeros(nc)
        rounds = 0; per_sim = None; sims_done = 0
        while True:
            # ---- choose this round's (child, index) pairs
            if rounds == 0:
                batch = [(c, 0) for c in range(nc)]
            else:
                nv = n.copy(); batch = []
                q = np.array([np.mean([b for _, b in v]) if v else 0.0 for v in vals])
                for _ in range(nc):
                    sc = (q - q[0]) / cfg["VS"] + cfg["C_PUCT"] * P * np.sqrt(nv.sum()) / (1 + nv)
                    sc[nv >= RS] = -np.inf
                    c = int(np.argmax(sc))
                    if not np.isfinite(sc[c]):
                        break
                    batch.append((c, int(nv[c]))); nv[c] += 1
                # the default must keep pace (paired comparison)
                need = int(max((nv[1:]).max(), 1)) if nc > 1 else 1
                while nv[0] < min(need, RS) and len(batch) < 2 * nc:
                    batch.append((0, int(nv[0]))); nv[0] += 1
            if not batch:
                break
            now = clk()
            if per_sim is not None and now - t0 + cfg["PROJ"] * per_sim * len(batch) > cfg["BUDGET"]:
                break
            if rounds == 0 and now - t0 > 0.4 * cfg["BUDGET"]:
                break
            if per_sim is not None and time.time() - tw + cfg["PROJ"] * per_wall * len(batch) > cfg["HARD"]:
                break
            if cfg["START_DL"] is not None and time.time() - tw > cfg["START_DL"]:
                break
            if cfg["ABORT_DL"] is not None and "stepw" in st and time.time() - tw + \
                    cfg["PROJ"] * (H - 1) * st["stepw"] * len(batch) > cfg["ABORT_DL"]:
                break       # this round could not finish before the abort deadline: do not start it
            nw = time.time()
            nb = len(batch)
            while len(pool) < nb:
                pool.append(FastCarbon())
            sims, fs = [], []
            for j, (c, i) in enumerate(batch):
                s_ = pool[j]; s_.load_observation(full)
                s_._tid_counter = [cc + 500 for cc in s_._tid_counter]
                if me == 0:
                    s_.step(cand[c], roots[i])
                else:
                    s_.step(roots[i], cand[c])
                sims.append(s_); fs.append((feat_copy(f), feat_copy(fo)))
            depth = 1; aborted = False
            for hh in range(1, H):
                if cfg["ABORT_DL"] is not None:
                    est = st.get("stepw", 0.0) * nb
                    if time.time() - tw + cfg["PROJ"] * est > cfg["ABORT_DL"]:
                        aborted = True
                        break
                ts_ = time.time()
                xs, auxs, tch = [], [], []
                for s_, (fm, fop) in zip(sims, fs):
                    xm = fm.observe(s_.obs(me)); xs.append(xm); auxs.append(fm.aux); tch.append(st.get("T_now", T))
                for (c, i), s_, (fm, fop) in zip(batch, sims, fs):
                    xq = fop.observe(s_.obs(1 - me)); xs.append(xq); auxs.append(fop.aux); tch.append(int(ks[i]))
                nz = np.concatenate([np.zeros((nb, F.OWN)), np.stack([rnoise[i, hh - 1] for c, i in batch])])
                tp = np.concatenate([np.zeros(nb), np.full(nb, tau)])
                X = stack_x(xs)
                tk, cx = encode_b(rpol, X, np.asarray(tch))
                lab = decode_b(rpol, X, auxs, tk, cx, noise=nz, temp=tp)
                for j, (s_, (fm, fop)) in enumerate(zip(sims, fs)):
                    am = fm.to_actions(lab[j]); ao = fop.to_actions(lab[nb + j])
                    if me == 0:
                        s_.step(am, ao)
                    else:
                        s_.step(ao, am)
                stats["branch_steps"] += nb
                depth = hh + 1
                sw = (time.time() - ts_) / nb
                st["stepw"] = sw if "stepw" not in st else max(sw, 0.7 * st["stepw"] + 0.3 * sw)
            if aborted:
                stats["aborted"] = stats.get("aborted", 0) + 1
                break
            lft = max(0, left - depth)
            for (c, i), s_ in zip(batch, sims):
                vals[c].append((i, leaf(s_, me, lft))); n[c] += 1
            sims_done += nb; rounds += 1
            per_sim = (clk() - now) / nb; per_wall = (time.time() - nw) / nb
        stats["sims"] = stats.get("sims", 0) + sims_done
        stats["rounds"] = stats.get("rounds", 0) + rounds
        stats["branches"] += sims_done
        if rounds == 0:
            stats["shallow"] += 1
            return acts
        d0 = dict(vals[0])
        best, bj = -np.inf, 0
        for c in range(1, nc):
            pr = [v - d0[i] for i, v in vals[c] if i in d0]
            if len(pr) < cfg["MIN_N"]:
                continue
            if cfg["RISK_TOL"] is not None:
                # risk-aware: the chosen move's worst sampled outcome may not be much worse than the default's
                wc = min(v for i, v in vals[c] if i in d0); w0 = min(d0[i] for i, _ in vals[c] if i in d0)
                if wc < w0 - cfg["RISK_TOL"]:
                    continue
            if cfg["LEAF_Q"] is not None:
                # robust leaf: lower quantile of the candidate's outcomes minus that of the default (common indices)
                vc = [v for i, v in vals[c] if i in d0]; v0 = [d0[i] for i, _ in vals[c] if i in d0]
                m_ = float(np.quantile(vc, cfg["LEAF_Q"]) - np.quantile(v0, cfg["LEAF_Q"]))
            else:
                m_ = float(np.mean(pr)) - cfg["SE_K"] * (float(np.std(pr)) / max(len(pr) - 1, 1) ** 0.5 if len(pr) > 1 else 0.0)
            if m_ > best:
                best, bj = m_, c
        if cfg["LOG"]:
            log.append((step, cmeta, [[round(v) for _, v in vv] for vv in vals], round(best, 1) if bj else None))
        # record for distillation: root visit counts per move of the searched unit
        try:
            ku = keys[0][1]; ks_ = uid2slot.get(ku)
            if ks_ is not None and ks_ < 10:
                vis = np.zeros(5); qv = np.full(5, np.nan)
                for c in range(nc):
                    m_ = y[ks_] if cmeta[c] is None else (cmeta[c][1] if cmeta[c][0] == ku else None)
                    if m_ is not None and vals[c]:
                        vis[int(m_)] += n[c]; qv[int(m_)] = float(np.mean([b_ for _, b_ in vals[c]]))
                st["rec"] = {"slot": int(ks_), "visits": vis, "q": qv, "changed": bool(bj and best > cfg["MARGIN"])}
        except Exception:
            pass
        if bj and best > cfg["MARGIN"]:
            stats["changed"] += 1; stats["gain"] += best
            return cand[bj]
        return acts

    def _search(obs, x, y, acts, xo, left, t0, tw):
        me = obs["player"]; step = obs["step"]
        f, fo = st["f"], st["fo"]
        keys = key_units(obs, cfg)
        if not keys:
            return acts
        stats["searched"] += 1
        tok, ctx = pol.last_enc
        uid2slot = {u: s for s, u in enumerate(f.own_uid) if u is not None}
        # ---- root candidate joint actions (b128 re-decode, prefix forced to the default)
        forced, meta = [], []
        for _, uid in keys:
            s = uid2slot.get(uid)
            if s is None or s >= 10:
                continue
            for m in range(5):
                if m == y[s]:
                    continue
                fz = np.full(F.OWN, -1, np.int64)
                fz[:s] = y[:s]; fz[s] = m
                forced.append(fz); meta.append((uid, m))
        forced = forced[:cfg["MAX_BR"] - 1]; meta = meta[:cfg["MAX_BR"] - 1]
        if not forced:
            return acts
        B0 = len(forced)
        X1 = {k: np.repeat(x[k][None], B0, 0) for k in ("tok_pos", "tok_kind", "tok_mask", "tok_cargo")}
        X1["cash"] = [x["cash"]] * B0; X1["slots"] = [x["slots"]] * B0; X1["left"] = [x["left"]] * B0
        ys = decode_b(pol, X1, [f.aux] * B0, np.repeat(tok[None], B0, 0), np.repeat(ctx[None], B0, 0),
                      np.stack(forced))
        cand = [acts]; seen = {tuple(y)}; cmeta = [None]
        for yy, mm in zip(ys, meta):
            k = tuple(yy)
            if k in seen:
                continue
            seen.add(k); cand.append(f.to_actions(yy)); cmeta.append(mm)
        if len(cand) < 2:
            return acts
        # ---- opponent root action (b128 from its seat)
        if st.get("oenc") is not None:
            Xo, toko, preo = st["oenc"]
            yo = decode_b(pol, Xo, [fo.aux], toko, ctx_b(pol, preo, np.array([st["ot"]])))[0]
        else:
            yo, _, _ = pol.act(xo, fo.aux, teacher=st["ot"])
        acts_o = fo.to_actions(yo)
        # ---- branch sims
        full = {"step": step, "players": obs["players"], "trees": obs.get("trees") or {},
                "full_carbon": [float(v) for v in f.est], "carbon": obs["carbon"], "remainingOverageTime": 60}
        models = [acts_o]
        if cfg["ROBUST"]:
            ah = hunter_acts(obs, me, acts_o, cfg)
            if ah != acts_o:
                models.append(ah)
        nc = len(cand)
        bl = [(mi, ci) for mi in range(len(models)) for ci in range(nc)]
        nb = len(bl)
        while len(pool) < nb:
            pool.append(FastCarbon())
        sims, fs = [], []
        for i, (mi, ci) in enumerate(bl):
            a = cand[ci]; ao_ = models[mi]
            s_ = pool[i]; s_.load_observation(full)
            s_._tid_counter = [c + 500 for c in s_._tid_counter]
            if me == 0:
                s_.step(a, ao_)
            else:
                s_.step(ao_, a)
            sims.append(s_); fs.append((feat_copy(f), feat_copy(fo)))
        stats["branches"] += nb
        local = None
        if cfg["LOCAL_R"]:
            kp = [obs["players"][me][2][u][0] for _, u in keys]
            local = {w_ for p in (0, 1) for w_, w in obs["players"][p][2].items()
                     if min(F.DIST[w[0], c] for c in kp) <= cfg["LOCAL_R"]}
        H = min(cfg["H"], left + 1)
        depth = 1; last_dt = None
        for hh in range(1, H):
            now = clk()
            if last_dt is not None and now - t0 + cfg["PROJ"] * last_dt > cfg["BUDGET"]:
                break
            if last_dt is None and now - t0 > 0.45 * cfg["BUDGET"]:
                break
            if last_dt is not None and time.time() - tw + cfg["PROJ"] * last_wall > cfg["HARD"]:
                break
            nw = time.time()
            xs, auxs, tch = [], [], []
            for s_, (fm, fop) in zip(sims, fs):
                xm = fm.observe(s_.obs(me)); xs.append(xm); auxs.append(fm.aux); tch.append(st.get("T_now", T))
            for s_, (fm, fop) in zip(sims, fs):
                xq = fop.observe(s_.obs(1 - me)); xs.append(xq); auxs.append(fop.aux); tch.append(st["ot"])
            lab = act_b(rpol, xs, auxs, tch)
            ref = {}
            for i, (s_, (fm, fop)) in enumerate(zip(sims, fs)):
                am = fm.to_actions(lab[i]); ao = fop.to_actions(lab[nb + i])
                if local is not None:
                    if bl[i][1] == 0:
                        ref[bl[i][0]] = (am, ao)
                    else:
                        am0, ao0 = ref[bl[i][0]]
                        # far units / bases copy the default branch (common actions -> less rollout noise)
                        for d_, d0 in ((am, am0), (ao, ao0)):
                            for k_ in [k_ for k_ in d_ if k_ not in local]:
                                del d_[k_]
                            for k_, v_ in d0.items():
                                if k_ not in local:
                                    d_[k_] = v_
                if me == 0:
                    s_.step(am, ao)
                else:
                    s_.step(ao, am)
            stats["branch_steps"] += nb
            depth = hh + 1
            last_dt = clk() - now
            last_wall = time.time() - nw
        stats["depth_sum"] += depth
        if depth < min(cfg["MIN_DEPTH"], H):
            stats["shallow"] += 1
            return acts
        lft = max(0, left - depth)
        sc = np.array([leaf(s_, me, lft) for s_ in sims]).reshape(len(models), nc)
        if len(models) > 1:
            w_ = cfg["ROBUST"]
            sc = (1 - w_) * sc[0] + w_ * sc.min(0)
        else:
            sc = sc[0]
        j = int(np.argmax(sc))
        if cfg["LOG"]:
            log.append((step, cmeta, [round(float(v)) for v in sc], depth))
        if j != 0 and sc[j] > sc[0] + cfg["MARGIN"]:
            stats["changed"] += 1
            stats["gain"] += float(sc[j] - sc[0])
            return cand[j]
        return acts

    _inner = agent

    def agent(obs, config=None):            # root fallback: any exception -> empty action, never a failed step
        try:
            return _inner(obs, config)
        except Exception:
            stats["root_errors"] = stats.get("root_errors", 0) + 1
            return {}
    agent.stats = stats
    agent.state = st
    agent.log = log
    agent.cfg = cfg
    return agent
