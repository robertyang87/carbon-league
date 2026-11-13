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
    """NP.Policy that keeps the last encode() output (same numerics as v12).

    R12: the cache also records WHAT was encoded. The consumers (bankrupt guard,
    _mcts/_search root decodes) pair the cached (tok, ctx) with the ORIGINAL-frame
    x/aux, but after NP.act_tta the last encode is of a symmetry VIEW, and after an
    epol/opol decode this policy was not encoded at all (stale cache) -- in both
    cases last_enc is the wrong frame or a previous step. enc_for() restores the
    invariant: it returns the encoding of exactly (x, teacher), re-encoding on any
    mismatch. On the plain act() path the signature always matches, so the common
    case costs nothing."""

    @staticmethod
    def _sig(x, teacher):
        # The signature must cover EVERY field encode() consumes (trunk_b /
        # tok_features_b read all nine x fields). The old signature missed
        # tok_kind/tok_mask/tok_cargo/hist_cargo/hist_move: two states that
        # differed only in per-token cargo -- (1000, 800) vs (1100, 700) at
        # equal total, maps saturated -- collided, and enc_for() returned
        # the other state's encoding (review measurement: encode diff
        # 0.061544). The standard-TTA part of R12 was already at 0 and is
        # not touched by this.
        return (teacher, x["maps"].tobytes(), x["tok_pos"].tobytes(),
                x["tok_kind"].tobytes(), x["tok_mask"].tobytes(),
                x["tok_cargo"].tobytes(), x["hist_cell"].tobytes(),
                x["hist_cargo"].tobytes(), x["hist_move"].tobytes(),
                np.asarray(x["scal"]).tobytes())

    def encode(self, x, teacher):
        r = NP.Policy.encode(self, x, teacher)
        self.last_enc = r
        self.last_sig = self._sig(x, teacher)
        return r

    def enc_for(self, x, teacher):
        if getattr(self, "last_sig", None) != self._sig(x, teacher):
            return self.encode(x, teacher)
        return self.last_enc


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


def _cell_attn_b(pol, x, j):
    """cnet2_model.CellAttn (attention over the board cells inside the trunk, cfg TA) on x [B, N, C]."""
    n = f"tattn.{j}"; B, N_, C = x.shape; NH = pol.TNH; dh = C // NH
    qkv = _lin(pol, _ln(pol, x, n + ".n1"), n + ".qkv").reshape(B, N_, 3, NH, dh).transpose(2, 0, 3, 1, 4)
    sc = qkv[0] @ qkv[1].transpose(0, 1, 3, 2) / np.float32(dh ** 0.5) + pol.tbias[j]
    sc = np.exp(sc - sc.max(-1, keepdims=True))
    sc /= sc.sum(-1, keepdims=True)
    x = x + _lin(pol, (sc @ qkv[2]).transpose(0, 2, 1, 3).reshape(B, N_, C), n + ".out")
    return (x + _lin(pol, np.maximum(_lin(pol, _ln(pol, x, n + ".n2"), n + ".f1"), 0), n + ".f2")).astype(np.float32)


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
        if i in pol.ta:
            h = _cell_attn_b(pol, h, pol.ta[i])
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
    """every field of f (shallow), then private copies of the state that observe() mutates in place; a field added to
    Feat later is carried automatically (the 2026-11-10 INFER bug was a field this function did not know)."""
    n = F.Feat.__new__(F.Feat)
    n.__dict__.update(f.__dict__)
    n.est = f.est.copy(); n.last = f.last.copy(); n.last_step = f.last_step.copy()
    n.hist = {k: list(v) for k, v in f.hist.items()}
    n.own_uid = list(getattr(f, "own_uid", []))
    if isinstance(getattr(f, "inf", None), dict):
        n.inf = dict(f.inf)
    n.infer_mirror = False; n.infer_tree = False   # branch copies never run the belief experiments
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
    if cfg.get("THREAT_FIRST"):
        # lexicographic: a threatened loaded car (certain loss) outranks every hunting chance (uncertain gain)
        thr = []
        for uid, w in P[me][2].items():
            pos, cargo, kind = w
            if kind != "COLLECTOR" or cargo < cfg.get("THREAT_MIN", 100.0):
                continue
            if any(F.DIST[pos, e[0]] <= cfg["KEY_R"] and e[1] <= cargo for e in en):
                thr.append((cargo, uid))
        if thr:
            thr.sort(reverse=True)
            return thr[:cfg["MAX_KEY"]]
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
    if cfg.get("KEY_LATE_HOME") and obs["step"] >= cfg.get("KLH_STEP", 270):
        # endgame: the search budget goes to loaded cars whose way home is tight (left - d in [0, KLH_SLACK]);
        # the search (with the endgame leaf) decides whether to go home -- nothing is forced
        bases = [int(b) for b in P[me][1].values()]; left = 299 - obs["step"]; tight = []
        for uid, w in P[me][2].items():
            pos, cargo, kind = w
            if kind != "COLLECTOR" or cargo < cfg.get("KLH_MIN", 100.0) or not bases:
                continue
            d = min(F.DIST[pos, b] for b in bases)
            if 0 < d and 0 <= left - d <= cfg.get("KLH_SLACK", 4):
                tight.append((cargo, uid))
        # share the key slot with the normal selection: KLH_SHARE=1 -> tight cars on even steps only
        if tight and (not cfg.get("KLH_SHARE") or obs["step"] % 2 == 0):
            tight.sort(reverse=True)
            return tight[:cfg["MAX_KEY"]]
    mk = cfg["MAX_KEY"]
    if cfg.get("MAX_KEY_LATE") is not None and obs["step"] >= cfg.get("KEY_LATE_STEP", 999):
        mk = cfg["MAX_KEY_LATE"]          # endgame: several loaded units run home at once -> search more of them
    if cfg.get("KEY_LATE_CARGO") and obs["step"] >= cfg.get("KEY_LATE_STEP", 999):
        # endgame short trips: every collector carrying >= KLC_MIN joins the search, heaviest first (the leaf only
        # counts cargo that can still reach a base, so the search itself prefers trips it can finish) -- no threat rule
        have = {u for _, u in keys}
        loaded = sorted(((float(w[1]), u) for u, w in P[me][2].items()
                         if w[2] == "COLLECTOR" and float(w[1]) >= cfg.get("KLC_MIN", 10.0) and u not in have), reverse=True)
        keys = keys + loaded
    keys = keys[:mk]
    if cfg.get("PLANTER_KEY") and (obs["step"] <= cfg.get("PK_STEP", 100) or obs["step"] >= cfg.get("PK_LATE", 999)):
        # opening: up to PK_MAX planters join the search (on top of the collector keys) -- where to plant is decided
        # by the leaf's carbon-based tree value. Planters within PK_R of an enemy tree (the seize temptation) first,
        # the rest in rotation by step, so every planter gets searched now and then.
        et = [int(pos) for pos, _ in P[1 - me][3].values()]
        pl = sorted(uid for uid, w in P[me][2].items() if w[2] == "PLANTER")
        if cfg.get("PK_CASH"):     # no cash to plant now -> searching where to plant is idle churn: skip the planters
            own_ = len(P[me][3]); live_ = sum(1 for q in (0, 1) for _, a_ in P[q][3].values() if a_ < 50)
            price_ = 20.0 if own_ == 0 else 20.0 + 5.0 * 1.235 ** live_
            if float(P[me][0]) < price_:
                pl = []
        if pl:
            near = [u for u in pl if et and min(F.DIST[P[me][2][u][0], t] for t in et) <= cfg.get("PK_R", 2)]
            k0 = obs["step"] % len(pl); rot = pl[k0:] + pl[:k0]
            order = near + [u for u in rot if u not in near]
            keys = keys + [(0.0, u) for u in order[:cfg.get("PK_MAX", 1)]]
    if cfg.get("BASE_KEY") and obs["step"] >= cfg.get("BK_STEP", 200) and float(P[me][0]) >= 30.0 and len(P[me][2]) < 10:
        # late game: recruiting is searched too (none / collector / planter); the typed leaf values decide whether a
        # new worker is still worth its 30 -- no hard cut-off
        keys = keys + [(0.0, b) for b in sorted(P[me][1])[:cfg.get("BK_MAX", 1)]]
    return keys


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


class BankruptcyGuard:
    """Stateful anti-bankruptcy guard (A3: extracted verbatim from
    make_search_agent's bankrupt_guard closure; behaviour identical, see the
    parity test in automation/acceptance_r5/ -- same positions, same cfg,
    same outputs).

    State (per game, per seat): lostw = own workers lost since the last
    game reset; prevw = own worker uids at the last guarded step;
    guard_base = base slots this guard forced to recruit a collector
    (per step; cleared by reset_step, which the agent calls once per step
    exactly where it used to rebind st["guard_base"]).
    cfg keys read: LOWW, GUARD_NARROW, GUARD_CASH, GUARD_LOG.
    NOTE on base_fix: it is NOT guard state -- it is per-step rule-layer
    state written by the MACRO/LATE_PLANT blocks and reconciled by the
    agent after the guard call, so it stays in the agent's st dict; the
    guard only ever reads it indirectly through the caller's post-guard
    loop. Each candidate arm / scenario branch holds its own copy
    (snapshot()/restore()/copy()); never share one instance across
    branches, or one arm's lostw leaks into another.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.lostw = 0
        self.prevw = set()
        self.guard_base = set()

    def reset_game(self):
        """Per-game reset (was: st["lostw"]=0; st["prevw"]=set())."""
        self.lostw = 0
        self.prevw = set()
        self.guard_base.clear()

    def reset_step(self):
        """Per-step reset (was: st["guard_base"]=set()). Clears in place so
        any alias (e.g. the agent's st["guard_base"]) stays bound."""
        self.guard_base.clear()

    def snapshot(self):
        return {"lostw": self.lostw, "prevw": set(self.prevw),
                "guard_base": set(self.guard_base)}

    def restore(self, snap):
        self.lostw = snap["lostw"]
        self.prevw = set(snap["prevw"])
        self.guard_base = set(snap["guard_base"])

    def copy(self):
        g = BankruptcyGuard(self.cfg)
        g.restore(self.snapshot())
        return g

    def apply(self, x, aux, y, lg0, pol, own_uid, teacher, stats=None):
        """Guarded labels for one step. own_uid = this seat's slot-indexed
        worker uids (feat.own_uid); teacher = teacher slot for the re-decode
        path (pol.enc_for); stats is an optional dict receiving guard_moves /
        guard_recruit / guard_log (None = throwaway)."""
        stats = stats if stats is not None else {}
        cfg = self.cfg
        m = x["tok_mask"]; kind = x["tok_kind"]; pos = x["tok_pos"]; cargo = x["tok_cargo"]
        nw = int(sum(1 for s_ in range(10) if m[s_]))
        ncol = int(sum(1 for s_ in range(10) if m[s_] and kind[s_] == 0))
        fz = np.full(F.OWN, -1, np.int64); first = None
        cur = {u for u in own_uid[:10] if u is not None}
        self.lostw = self.lostw + len(self.prevw - cur)
        self.prevw = cur
        risky = nw <= cfg["LOWW"]
        if cfg["GUARD_NARROW"]:
            # only a real bankruptcy risk: we already lost a unit and a further death could not be re-bought
            risky = risky and self.lostw > 0 and x["cash"] < cfg["GUARD_CASH"]
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
                            "step": 299 - int(x["left"]), "nw": nw, "cash": round(float(x["cash"])), "lost": int(self.lostw),
                            "uid": own_uid[s_], "kind": "C" if kind[s_] == 0 else "P", "cargo": round(float(cargo[s_])),
                            "dob": int(aux["dob"][p_]), "nearest_enemy": min(en) if en else None,
                            "net_move": int(y[s_]), "chosen": int(ok[0])})
        if ncol == 0 and x["cash"] >= 30 and x["slots"] > 0 and x["left"] > 10:
            bs = [s_ for s_ in (10, 11) if m[s_]]
            if bs and not any(y[s_] == 1 for s_ in bs):
                fz[bs[0]] = 1; first = bs[0] if first is None else min(first, bs[0]); self.guard_base.add(bs[0])
                stats["guard_recruit"] = stats.get("guard_recruit", 0) + 1
        if first is None:
            return y
        fz[:first] = y[:first]                  # the guard has priority over rule-edited bases: no base_fix here
        tok, ctx = pol.enc_for(x, teacher)      # R12: original-frame encoding of THIS x (last_enc may be a TTA view)
        X1 = {k: x[k][None] for k in ("tok_pos", "tok_kind", "tok_mask", "tok_cargo")}
        X1["cash"] = [x["cash"]]; X1["slots"] = [x["slots"]]; X1["left"] = [x["left"]]
        y2 = decode_b(pol, X1, [aux], tok[None], ctx[None], fz[None])[0]
        return y2


def make_search_agent(conf=None, weights=None, rweights=None, oweights=None):
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
    opol = CachePolicy(load_weights(oweights)) if oweights is not None else None   # opening net (steps < OPEN_NET_STEP, no search)
    # ENS: extra policy weights (paths or dicts); the net's own move (before any rule / search layer) is then the mean
    # of the members' logits (cnet2_np.Ensemble). One extra trunk per member per step; rollouts keep using `pol`.
    ens = cfg.get("ENS")
    epol = NP.Ensemble([pol] + [NP.Policy(load_weights(e_)) for e_ in ens]) if ens else None
    st = {"f": None, "fo": None}
    guard = BankruptcyGuard(cfg)   # A3: the agent's one guard instance (per game, per seat)
    st["guard_base"] = guard.guard_base   # search reads this alias; reset_step() clears it in place
    pool = []
    stats = {"steps": 0, "searched": 0, "branches": 0, "changed": 0, "branch_steps": 0, "worst_ms": 0.0,
             "gain": 0.0, "errors": 0, "shallow": 0, "depth_sum": 0}
    log = []
    T, OT = cfg["TEACHER"], cfg["OPP_TEACHER"]

    def _extinct(sim, p):
        """original env (carbon.py compute_next_board): a side with no workers, no trees and
        (no conversion centre, or cash below the cheapest recruit) is DONE and the game ends
        on the spot. FastCarbon deliberately does not model this (see its header), so the
        search layer must recognise the terminal itself (R02)."""
        if any(w.player == p for w in sim.workers.values()):
            return False
        if any(t.player == p for t in sim.trees.values()):
            return False
        if not sim.bases[p]:
            return True
        cheapest = min(sim.config.get("recCollectorCost", 30), sim.config.get("recPlanterCost", 30))
        return float(sim.cash[p]) < cheapest

    def _terminal(sim):
        return _extinct(sim, 0) or _extinct(sim, 1)

    def leaf(sim, me, left, prev_cash=None):
        if _terminal(sim):
            # F01: original terminal scoring (carbon.py compute_next_board), NOT the cash
            # difference: an extinct side scores (step - episodeSteps - 1); a survivor keeps
            # the reward it carried INTO the terminal step -- its cash in the pre-step state --
            # because the original marks it DONE without a reward update. Callers that saw
            # the terminal transition pass prev_cash; for a state that was already terminal
            # when handed over, its own cash is the only value the state carries.
            pen = float(sim.step_count) - float(sim.config.get("episodeSteps", 300)) - 1.0
            pc = prev_cash if prev_cash is not None else sim.cash
            rew = [pen if _extinct(sim, p) else float(pc[p]) for p in (0, 1)]
            return rew[me] - rew[1 - me]
        m = [float(sim.cash[0]), float(sim.cash[1])]
        wv = cfg["WORKER_V"] * min(1.0, left / 100.0)
        bases = {p: [b for _, b in sim.bases[p]] for p in (0, 1)}
        lt = cfg.get("LEAF_THREAT") is not None and left <= cfg.get("LT_LEFT", 30)
        if lt:     # endgame leaf: cargo next to a lighter enemy collector is worth LEAF_THREAT of itself
            cols = {p: [(w.pos, w.carbon) for w in sim.workers.values() if w.player == p and w.is_collector] for p in (0, 1)}
        wt = cfg.get("WT_LEFT") is not None and left <= cfg["WT_LEFT"]
        if wt:     # late: a car is worth its remaining haul, a planter its best potential tree net of the plant price
            cv = cfg.get("WC_RATE", 4.4) * cfg.get("WC_UTIL", 0.5) * max(0, left - cfg.get("WC_TRIP", 0))
            cbm = np.asarray(sim.carbon, dtype=np.float64).reshape(15, 15)
            nbm = (np.roll(cbm, 1, 0) + np.roll(cbm, -1, 0) + np.roll(cbm, 1, 1) + np.roll(cbm, -1, 1)).ravel()
            tvm = nbm * (1.0 - 0.9625 ** max(0, min(left, 30))) * cfg.get("TSURV", 1.0)
            freem = np.ones(225, bool)
            for t in sim.trees.values():
                freem[int(t.pos)] = False
            for p_ in (0, 1):
                for b_ in bases[p_]:
                    freem[int(b_)] = False
            livem = sum(1 for t in sim.trees.values() if t.age < 50)
            pval = {}
            for p_ in (0, 1):
                own = sum(1 for t in sim.trees.values() if t.player == p_)
                price = 20.0 if own == 0 else 20.0 + 5.0 * 1.235 ** livem
                pval[p_] = np.where(freem, tvm - price, -1e9)
        for w in sim.workers.values():
            v = wv
            if wt and (not cfg.get("WT_NEW_ONLY") or w.id not in st.get("root_wids", ())):
                if w.is_collector:
                    v = cv
                else:
                    d_ = F.DIST[w.pos]; ok_ = d_ <= 3
                    v = max(0.0, float(np.max(pval[w.player][ok_] * (0.85 ** d_[ok_])))) if ok_.any() else 0.0
            if w.carbon > 0:
                bs = bases[w.player]
                d = min(F.DIST[w.pos, b] for b in bs) if bs else 99
                if d <= left:
                    f_ = 1.0
                    if lt:
                        # threat = an enemy collector with cargo <= ours (equal cargo: no winner, all deleted -> our
                        # cargo is lost too) at torus distance <= 1 (can collide next step). Road: next to the car.
                        # Base mouth (LT_BASE): next to the base this car goes home to, only when that base is <= 5
                        # steps away. Either trigger -> ONE discount, never multiplied.
                        en_ = cols[1 - w.player]
                        road = any(F.DIST[w.pos, ep] <= 1 and ec <= w.carbon for ep, ec in en_)
                        mouth = False
                        if cfg.get("LT_BASE") and not road:
                            hb = min(bs, key=lambda b: F.DIST[w.pos, b])
                            mouth = F.DIST[w.pos, hb] <= 5 and any(F.DIST[hb, ep] <= 1 and ec <= w.carbon for ep, ec in en_)
                        if (road and cfg.get("LT_ROAD", True)) or mouth:
                            f_ = cfg["LEAF_THREAT"]
                    v += cfg["CARGO_W"] * w.carbon * f_
            m[w.player] += v
        if cfg.get("TREE_C") is not None:
            # tree = expected income: TREE_C * carbon on its 4 neighbours * remaining life (0.0375 ~ the per-step
            # share a tree draws from each neighbour). TC_ONCE: a tree that changed owner since the root counts once
            # (for the new owner), not as +v for us and -v for them -- a seized tree is worth its own income only.
            cb = sim.carbon; ro = st.get("root_tree_owner", {})
            for t in sim.trees.values():
                q_ = int(t.pos); r_, c_ = divmod(q_, 15)
                nb = cb[((r_ - 1) % 15) * 15 + c_] + cb[((r_ + 1) % 15) * 15 + c_] + cb[r_ * 15 + (c_ - 1) % 15] + cb[r_ * 15 + (c_ + 1) % 15]
                if cfg.get("TREE_SAT"):      # saturating: neighbours drained 3.75%/step, life capped at TREE_SAT (30)
                    L_ = max(0, min(50 - t.age, left, cfg["TREE_SAT"]))
                    v_ = cfg["TREE_C"] * nb * (1.0 - 0.9625 ** L_)
                    if cfg.get("TSURV") is not None and cfg.get("WT_LEFT") is not None and left <= cfg["WT_LEFT"] \
                            and t.id not in st.get("root_tids", ()):
                        v_ *= cfg["TSURV"]   # late, trees planted after the root only: same survival as potential trees
                else:
                    v_ = cfg["TREE_C"] * nb * max(0, min(50 - t.age, left))
                m[t.player] += v_
                if cfg.get("TC_ONCE") and ro.get(q_, t.player) != t.player:
                    m[1 - t.player] += v_
        elif cfg["TREE_V"]:
            for t in sim.trees.values():
                m[t.player] += cfg["TREE_V"] * max(0, min(50 - t.age, left))
        if cfg.get("PD_POT") and left > cfg.get("PD_LEFT", 200):
            # opening: a planter is worth its best "potential tree" within PD_K steps: max over free cells of
            # (saturating tree value - current plant price) * 0.85^d -- same scale as the real trees in this leaf
            cb = np.asarray(sim.carbon, dtype=np.float64).reshape(15, 15)
            nb = (np.roll(cb, 1, 0) + np.roll(cb, -1, 0) + np.roll(cb, 1, 1) + np.roll(cb, -1, 1)).ravel()
            Lp = max(0, min(left, cfg.get("TREE_SAT", 30)))
            tv = cfg.get("TREE_C", 1.0) * nb * (1.0 - 0.9625 ** Lp)
            free = np.ones(225, bool)
            for t in sim.trees.values():
                free[int(t.pos)] = False
            for p_ in (0, 1):
                for b_ in bases[p_]:
                    free[int(b_)] = False
            live = sum(1 for t in sim.trees.values() if t.age < 50)
            K_ = cfg.get("PD_K", 3)
            for p_ in (0, 1):
                own = sum(1 for t in sim.trees.values() if t.player == p_)
                price = 20.0 if own == 0 else 20.0 + 5.0 * 1.235 ** live
                gain = np.where(free, tv - price, -1e9)
                for w in sim.workers.values():
                    if w.player == p_ and not w.is_collector:
                        d_ = F.DIST[w.pos]
                        ok_ = d_ <= K_
                        if ok_.any():
                            best_ = float(np.max(gain[ok_] * (cfg.get("PD_DISC", 0.85) ** d_[ok_])))
                            if best_ > 0:
                                m[p_] += best_
        if cfg.get("PD_W") and left > cfg.get("PD_LEFT", 200):
            # opening: a planter is worth more the closer it stands to a free cell with >= PG_MIN neighbour carbon
            cb = np.asarray(sim.carbon, dtype=np.float64).reshape(15, 15)
            nb = (np.roll(cb, 1, 0) + np.roll(cb, -1, 0) + np.roll(cb, 1, 1) + np.roll(cb, -1, 1)).ravel()
            free = np.ones(225, bool)
            for t in sim.trees.values():
                free[int(t.pos)] = False
            for p_ in (0, 1):
                for b_ in bases[p_]:
                    free[int(b_)] = False
            good = np.nonzero(free & (nb >= cfg.get("PG_MIN", 100.0)))[0]
            if len(good):
                for w in sim.workers.values():
                    if not w.is_collector:
                        m[w.player] -= cfg["PD_W"] * float(F.DIST[w.pos][good].min())
        return m[me] - m[1 - me]

    clk = time.process_time if cfg["CLOCK"] == "cpu" else time.time

    def agent(obs, config=None):
        t0 = clk(); tw = time.time()
        me = obs["player"]
        if st.get("pk_pend"):       # planters whose move the search replaced: planted (our tree under it) within 3 steps?
            P_ = obs["players"][me]; tp_ = {int(pos) for pos, _ in P_[3].values()}
            for u_, s0 in list(st["pk_pend"].items()):
                w_ = P_[2].get(u_)
                if w_ is not None and int(w_[0]) in tp_:
                    stats["pk_rep_plant3"] = stats.get("pk_rep_plant3", 0) + 1; del st["pk_pend"][u_]
                elif w_ is None or obs["step"] - s0 > 3:
                    del st["pk_pend"][u_]
        if cfg.get("TSURV") is not None:   # trees standing at the root keep full value; only new late trees get TSURV
            st["root_tids"] = {t for p_ in (0, 1) for t in obs["players"][p_][3]}
        if cfg.get("WT_NEW_ONLY"):  # workers present at the root keep the flat leaf value; only new ones are typed
            st["root_wids"] = {u for p_ in (0, 1) for u in obs["players"][p_][2]}
        if cfg.get("TC_ONCE"):      # tree owners at the root (by cell): the leaf counts a seized tree once
            st["root_tree_owner"] = {int(pos): p_ for p_ in (0, 1) for pos, _ in obs["players"][p_][3].values()}
        if st["f"] is None or obs["step"] == 0 or st["f"].seat != me:
            st["f"] = F.Feat(me); st["fo"] = F.Feat(1 - me)
            st["f"].infer_mirror = bool(cfg.get("INFER_MIRROR"))
            st["f"].infer_tree = bool(cfg.get("INFER_TREE"))
        f, fo = st["f"], st["fo"]
        if obs["step"] == 0 or "lpost" not in st:
            st["lpost"] = np.array([cfg["ID_PRIOR"] if k == OT else 0.0 for k in cfg["STYLES"]])
            st["ot"] = OT; st["oprev"] = None; st["pw"] = None; st["ntrk"] = 0; guard.reset_game()
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
            # F03: this branch decodes with teacher T -- record it as this step's teacher,
            # like the greedy branch does below. The enc_for consumers (bankruptcy guard,
            # search roots) read st["T_now"]; leaving it unset made the guard raise
            # KeyError('T_now'), fall back and change the sampled opening's actions.
            st["T_now"] = T
            # opening diversity (data generation): sample the net's moves at temperature OPEN_T
            y, lg0, _ = pol.act(x, f.aux, teacher=T, sample_rng=st["orng"], temp=cfg["OPEN_T"],
                                sample_kinds=cfg.get("OPEN_KINDS"))     # e.g. ("planter", "base"): cars stay greedy
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
            if cfg.get("TEACHER_LATE2") is not None and obs["step"] >= cfg.get("LATE_STEP2", 999):
                st["T_now"] = cfg["TEACHER_LATE2"]          # optional third phase (e.g. 7 from 240, then 2 from 280)
            if opol is not None and obs["step"] < cfg.get("OPEN_NET_STEP", 0):
                y, lg0, _ = opol.act(x, f.aux, teacher=cfg.get("OPEN_NET_T", 2))
            else:
                if cfg.get("TTA", 1) > 1 and epol is None:
                    # symmetry averaging: the same net on TTA of the 4 board symmetries that keep both sides' bases
                    # (identity, rot180, transpose, anti-transpose), log-probs averaged. No extra parameters.
                    y, lg0 = NP.act_tta(pol, x, f.aux, teacher=st["T_now"], k=int(cfg["TTA"]))
                else:
                    y, lg0, _ = (epol if epol is not None else pol).act(x, f.aux, teacher=st["T_now"])
        st["lg0"] = lg0; st["y0"] = y.copy()
        st["base_fix"] = {}      # base slots set by a rule layer this step: every search candidate / guard re-decode keeps them
        guard.reset_step()       # A3: clears guard_base in place (st["guard_base"] stays aliased)
        # car-curve rule (both opponent lines, BEFORE the bankruptcy guard so the guard still sees the final plan):
        # while below the Master2s collector curve (target = 2 + #{CC_STEPS <= t}, 8 after the last step, off after
        # CC_END) ONE base switches to a collector: a planter recruit -> collector (only with >= 2 planters in play),
        # or no recruit -> collector. Cash must cover 30 on top of this step's other planned spending (other base
        # recruits, 30 per staying planter as a plant/seize upper bound). Nothing when 10 workers are in play.
        ccs = cfg.get("CAR_CURVE")
        if ccs and ccs[0] <= obs["step"] <= cfg.get("CC_END", 60):
            t_ = obs["step"]; mine = list(obs["players"][me][2].values())
            ncol = sum(1 for w in mine if w[2] == "COLLECTOR"); npl = sum(1 for w in mine if w[2] == "PLANTER")
            target = 2 + sum(1 for th in ccs if th <= t_)
            slots = [s_ for s_ in range(10, F.OWN) if x["tok_mask"][s_] and y[s_] >= 0]
            planned_col = sum(1 for s_ in slots if y[s_] == 1)                 # a collector the net already recruits
            if ncol + planned_col < target and len(mine) + sum(1 for s_ in slots if y[s_] in (1, 2)) < 10 and slots:
                stay_pl = sum(1 for s_ in range(10) if x["tok_mask"][s_] and int(x["tok_kind"][s_]) == 1 and y[s_] == 0)
                for s_ in slots:
                    if y[s_] == 2 and npl < 2:
                        continue
                    if y[s_] in (0, 2):
                        other = 30.0 * sum(1 for o_ in slots if o_ != s_ and y[o_] in (1, 2))
                        if float(obs["players"][me][0]) >= 30.0 + other + plant_price(obs, me) * stay_pl:
                            k = "cc_2to1" if y[s_] == 2 else "cc_0to1"
                            stats[k] = stats.get(k, 0) + 1; y[s_] = 1; st["base_fix"][s_] = 1
                        break
        # PLANTER_CASH: no planter recruit the base could not also pay a tree for -- cash after this step's recruits
        # (30 each) must still cover the current planting price, else the base recruits nothing (stats pb_skip).
        if cfg.get("PLANTER_CASH"):
            slots = [s_ for s_ in range(10, F.OWN) if x["tok_mask"][s_] and y[s_] >= 0]
            pp = plant_price(obs, me)
            for s_ in slots:
                if y[s_] == 2:
                    spend = 30.0 * sum(1 for o_ in slots if y[o_] in (1, 2))
                    if float(obs["players"][me][0]) - spend < pp:
                        y[s_] = 0; stats["pb_skip"] = stats.get("pb_skip", 0) + 1; st["base_fix"][s_] = 0
        # MACRO knobs (reports/plan_macro_training.md, stage A): permission-type limits on recruiting and planting,
        # found by data/knob_search.py. Units keep the net's moves; a knob only vetoes a planter recruit (-> collector
        # or nothing) or a plant at a price above the limit (-> the net's best move). Phases: E < PL_T1 <= M < PL_T2 <= L.
        #   PL_CAP_E/M/L  planters allowed in play (None = no limit); + REACT_T * max(0, enemy trees - our trees)
        #   REC_STOP_P    no planter recruits from this step on
        #   PRICE_M/L     plant only at a price <= limit (None = no limit); - REACT_S * our trees seized in the last 50 steps
        #   PL_MIN_E/M/L  (the one knob that ADDS planters; ladder 2026-11-11: Echo1 loses the games in which the
        #                 opponent out-plants it, and M6 beat Master3s holding ~6 planters) -- while fewer planters than
        #                 this are in play, a collector recruit the net asks for becomes a planter recruit (stats
        #                 mac_push); + PL_MIN_REACT * max(0, enemy trees - our trees). Never adds a recruit of its own.
        mac = cfg.get("MACRO")
        if mac:
            try:
                t_ = obs["step"]; Pm = obs["players"][me]; Po = obs["players"][1 - me]
                ph = "E" if t_ < mac.get("PL_T1", 60) else ("M" if t_ < mac.get("PL_T2", 240) else "L")
                # our trees seized recently (cell now holds an enemy tree)
                cur = {int(pos) for pos, _ in Pm[3].values()}; ecur = {int(pos) for pos, _ in Po[3].values()}
                if t_ == 0 or "mac_prev" not in st:
                    st["mac_seized"] = []
                else:
                    st["mac_seized"] += [t_] * len((st["mac_prev"] - cur) & ecur)
                st["mac_prev"] = cur
                st["mac_seized"] = [s0 for s0 in st["mac_seized"] if t_ - s0 < 50]
                npl = sum(1 for w in Pm[2].values() if w[2] == "PLANTER"); nwk = len(Pm[2])
                slots = [s_ for s_ in range(10, F.OWN) if x["tok_mask"][s_] and y[s_] >= 0]
                cap = mac.get("PL_CAP_" + ph)
                if cap is not None:
                    cap = cap + int(round(mac.get("REACT_T", 0.0) * max(0, len(Po[3]) - len(Pm[3]))))
                stop = t_ >= mac.get("REC_STOP_P", 999)
                for s_ in slots:
                    if y[s_] != 2:
                        continue
                    planned = sum(1 for o_ in slots if y[o_] == 2)
                    if stop:
                        y[s_] = 0; st["base_fix"][s_] = 0; stats["mac_stop"] = stats.get("mac_stop", 0) + 1
                    elif cap is not None and npl + planned > cap:
                        to = 1 if mac.get("CAP_TO_CAR", True) and t_ < mac.get("CAR_STOP", 276) else 0
                        y[s_] = to; st["base_fix"][s_] = to; stats["mac_cap"] = stats.get("mac_cap", 0) + 1
                pmin = mac.get("PL_MIN_" + ph)
                if pmin is not None and t_ < mac.get("PUSH_STOP", 250):
                    pmin = pmin + int(round(mac.get("PL_MIN_REACT", 0.0) * max(0, len(Po[3]) - len(Pm[3]))))
                    for s_ in slots:
                        if y[s_] == 1 and npl + sum(1 for o_ in slots if y[o_] == 2) < pmin:
                            y[s_] = 2; st["base_fix"][s_] = 2; stats["mac_push"] = stats.get("mac_push", 0) + 1
                lim = mac.get("PRICE_" + ph)
                if lim is not None:
                    lim = lim - mac.get("REACT_S", 0.0) * len(st["mac_seized"])
                    pp = plant_price(obs, me)
                    if pp > lim and float(Pm[0]) >= pp:
                        blocked = cur | ecur | {int(b) for q_ in (0, 1) for b in obs["players"][q_][1].values()}
                        for s_ in range(10):
                            if not x["tok_mask"][s_] or int(x["tok_kind"][s_]) != 1 or y[s_] != 0:
                                continue
                            uid = f.own_uid[s_]; w_ = Pm[2].get(uid)
                            if w_ is None or int(w_[0]) in blocked:
                                continue          # standing on a tree (seize / own tree) or a base: not a plant
                            mv = 1 + int(np.argmax(lg0[s_][1:5]))
                            y[s_] = mv; st["base_fix"][s_] = mv; stats["mac_price"] = stats.get("mac_price", 0) + 1
            except Exception as e:
                stats["errors"] += 1; stats["last_error"] = repr(e)[:300]
        # LATE_PLANT {FROM, TO, RATIO, K}: late in the game a planter that stands on a free cell plants when the tree pays:
        # believed carbon on the 4 neighbours * (1 - 0.9625 ** steps left) * K  >=  RATIO * current price. Ladder
        # 2026-11-11: all four games Master3s lost were lost in the last 50 steps to opponents holding 9-11 trees at step
        # 290 while it held none (their tree income in the last third 1158-2711, its own 0-1070); with one side treeless
        # the price is low. The net learned from Master3s to stop planting. Only the move of a planter is changed (stay =
        # plant); no recruit is added (see MACRO PL_MIN_L / PUSH_STOP for keeping planters late). stats: lp.
        lpc = cfg.get("LATE_PLANT")
        if lpc and lpc.get("FROM", 220) <= obs["step"] <= lpc.get("TO", 292):
            try:
                Pm = obs["players"][me]; Po = obs["players"][1 - me]; left = 299 - obs["step"]
                taken = {int(v[0]) for q_ in (Pm, Po) for v in q_[3].values()} | {int(b) for q_ in (Pm, Po) for b in q_[1].values()}
                cash = float(Pm[0]) - 30.0 * sum(1 for s_ in range(10, F.OWN) if x["tok_mask"][s_] and y[s_] > 0)
                price = plant_price(obs, me); ntree = len(Pm[3])
                for s_ in range(10):
                    if not x["tok_mask"][s_] or int(x["tok_kind"][s_]) != 1 or y[s_] == 0:
                        continue
                    w_ = Pm[2].get(f.own_uid[s_])
                    if w_ is None or int(w_[0]) in taken or cash < price:
                        continue
                    nb = float(sum(f.est[n_] for n_ in F.DEST1_L[int(w_[0])] if n_ not in taken))
                    if nb * (1.0 - 0.9625 ** left) * lpc.get("K", 0.7) >= lpc.get("RATIO", 1.5) * price:
                        y[s_] = 0; stats["lp"] = stats.get("lp", 0) + 1
                        cash -= price; taken.add(int(w_[0])); ntree += 1
                        price = 20.0 + 5.0 * 1.235 ** (sum(1 for q_ in (Pm, Po) for v in q_[3].values() if int(v[1]) < 50) + ntree - len(Pm[3]))
            except Exception as e:
                stats["errors"] += 1; stats["last_error"] = repr(e)[:300]
        if cfg["GUARD"]:
            try:
                y = bankrupt_guard(x, f.aux, y, lg0)
                for s_b in [k for k, v in st.get("base_fix", {}).items() if int(y[k]) != v]:
                    st["base_fix"].pop(s_b); stats["fix_dropped"] = stats.get("fix_dropped", 0) + 1   # guard overrode
            except Exception as e:
                stats["errors"] += 1; stats["last_error"] = repr(e)[:300]
        # vs the Master line (opponent's first recruit was a collector): from step MNP_STEP no planter recruits
        # until we have MNP_COLS collectors -- the base recruits a collector instead (bases are decoded last)
        if cfg.get("MASTER_NO_PLANTER") and st.get("opp_first") == "COLLECTOR" and obs["step"] >= cfg.get("MNP_STEP", 40):
            ncol = sum(1 for w in obs["players"][me][2].values() if w[2] == "COLLECTOR")
            if ncol < cfg.get("MNP_COLS", 10):
                for s_ in range(10, F.OWN):
                    if y[s_] == 2:
                        y[s_] = 1; stats["mnp"] = stats.get("mnp", 0) + 1; st["base_fix"][s_] = 1
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
            if not cfg["SEARCH"] or left < cfg["ENDGAME_STOP"] or (opol is not None and obs["step"] < cfg.get("OPEN_NET_STEP", 0)):
                agent.last_stats = dict(stats)
                return acts
            if cfg["MODE"] == "mcts":
                st["seq_last"] = None
                out = _mcts(obs, x, y, acts, xo, left, t0, tw)
                if int(cfg.get("SEQ_KEYS") or 1) > 1:
                    # measurement (fixed-sims clock): after the first searched car, search the next car in a capture
                    # situation with the first result as the new default, and so on -- up to SEQ_KEYS cars per step.
                    # The normal search changes at most ONE unit per step while ~3 cars are in such a situation.
                    try:
                        done_ = set(); f_ = st["f"]
                        for _ in range(int(cfg["SEQ_KEYS"]) - 1):
                            if st.get("seq_last") is None:
                                break
                            done_.add(st["seq_last"]); st["seq_skip"] = set(done_); st["seq_last"] = None
                            y2 = y.copy()
                            for s_, u_ in enumerate(f_.own_uid):
                                if u_ is None:
                                    continue
                                a_ = out.get(u_, "NONE")
                                if s_ < 10:
                                    y2[s_] = next(k_ for k_ in range(5) if F.MOVES[F.CANON_MOVE[k_] if f_.seat else k_] == a_)
                                else:
                                    y2[s_] = F.BASE_ACTS.index(a_)
                            c0_ = stats["changed"]
                            out = _mcts(obs, x, y2, out, xo, left, t0, tw)
                            stats["seq_changed"] = stats.get("seq_changed", 0) + (stats["changed"] - c0_)
                    except Exception:
                        stats["errors"] += 1
                    finally:
                        st["seq_skip"] = None
            else:
                out = _search(obs, x, y, acts, xo, left, t0, tw)
            stats["worst_ms"] = max(stats["worst_ms"], (clk() - t0) * 1000)
            stats["worst_wall_ms"] = max(stats.get("worst_wall_ms", 0.0), (time.time() - tw) * 1000)
            stats.setdefault("ms_steps", []).append(round((time.time() - tw) * 1000))   # wall ms of every searched step
            agent.last_stats = dict(stats)
            return out
        except Exception as e:  # never let the search layer lose a game
            stats["errors"] += 1; stats["last_error"] = repr(e)[:300]
            agent.last_stats = dict(stats)
            return acts

    def bankrupt_guard(x, aux, y, lg0):
        """Thin wrapper kept for the single call site below; the logic lives
        in BankruptcyGuard.apply (A3). st["guard_base"] stays aliased to
        guard.guard_base (same set object; reset_step clears in place)."""
        return guard.apply(x, aux, y, lg0, pol, st["f"].own_uid, st["T_now"], stats)

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
        if st.get("seq_skip"):                   # SEQ_KEYS pass >= 2: units searched earlier this step are not searched again
            keys = [k_ for k_ in key_units(obs, dict(cfg, MAX_KEY=10)) if k_[1] not in st["seq_skip"]][:cfg["MAX_KEY"]]
        if not keys:
            return acts
        st["seq_last"] = keys[0][1]
        stats["searched"] += 1
        tok, ctx = pol.enc_for(x, st["T_now"])     # R12: original-frame encoding of THIS x (last_enc may be a TTA view)
        lg0 = st["lg0"]
        uid2slot = {u: s for s, u in enumerate(f.own_uid) if u is not None}
        forced, meta, pri = [], [], []
        pdef = 0.0
        for _, uid in keys:
            s = uid2slot.get(uid)
            if s is None or (s >= 10 and (not cfg.get("BASE_KEY") or s in st.get("base_fix", {}) or s in st.get("guard_base", ()))):
                continue
            z = lg0[s].astype(np.float64); pz = np.exp(z - z.max()); pz /= pz.sum()
            pdef = max(pdef, pz[y[s]])
            for m in range(3 if s >= 10 else 5):
                if m == y[s]:
                    continue
                fz = np.full(F.OWN, -1, np.int64)
                fz[:s] = y[:s]; fz[s] = m
                for s_b, v_b in st.get("base_fix", {}).items():
                    if s_b != s:
                        fz[s_b] = v_b
                forced.append(fz); meta.append((uid, m)); pri.append(pz[m])
        if cfg.get("BASE_KEY"):      # base keys: got candidates / all cut by MAX_BR
            bk_u = {u for _, u in keys if u in uid2slot and uid2slot[u] >= 10}
            if bk_u:
                kept_b = {u for u, _ in meta[:cfg["MAX_BR"] - 1]}
                stats["bk_in"] = stats.get("bk_in", 0) + (1 if bk_u & kept_b else 0)
                stats["bk_cut"] = stats.get("bk_cut", 0) + (1 if bk_u - kept_b else 0)
        if cfg.get("PLANTER_KEY"):   # did a planter key get any candidate, or did MAX_BR cut all of them?
            pk_u = {u for _, u in keys if u in uid2slot and obs["players"][me][2].get(u, (0, 0, ""))[2] == "PLANTER"}
            if pk_u:
                kept = {u for u, _ in meta[:cfg["MAX_BR"] - 1]}
                stats["pk_in"] = stats.get("pk_in", 0) + (1 if pk_u & kept else 0)
                stats["pk_cut"] = stats.get("pk_cut", 0) + (1 if pk_u - kept else 0)
        mbr = cfg["MAX_BR"] if not (cfg.get("MAX_BR_LATE") and obs["step"] >= cfg.get("KEY_LATE_STEP", 999)) else cfg["MAX_BR_LATE"]
        forced = forced[:mbr - 1]; meta = meta[:mbr - 1]
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
        H = min(cfg["H"], left)          # R01: left = 299 - step already counts the remaining transfers (root step included); left+1 simulated one step past the end of the game
        if cfg.get("PK_H2") and all(obs["players"][me][2].get(u, (0, 0, ""))[2] == "PLANTER" for _, u in keys):
            H = min(H, 2)          # planter-only step: the root plant shows in a depth-2 leaf; depth 3 adds little
        rnoise = rng.random((RS, max(H - 1, 1), F.OWN))
        full = {"step": step, "players": obs["players"], "trees": obs.get("trees") or {},
                "full_carbon": [float(v) for v in f.est], "carbon": obs["carbon"], "remainingOverageTime": 60}
        vals = [[] for _ in range(nc)]
        n = np.zeros(nc)
        rounds = 0; per_sim = None; sims_done = 0
        fixed = cfg["CLOCK"] == "sims"
        fix_n = cfg["FIX_SIMS"][min(step // 50, len(cfg["FIX_SIMS"]) - 1)] if fixed else 0
        while True:
            # ---- choose this round's (child, index) pairs
            if rounds == 0:
                batch = [(c, 0) for c in range(nc)]
            else:
                nv = n.copy(); batch = []
                q = np.array([np.mean([b for _, b in v]) if v else 0.0 for v in vals])
                for _ in range(int(cfg.get("ROUND_K", 1)) * nc):     # ROUND_K: more sims per round (bigger, fewer batches)
                    sc = (q - q[0]) / cfg["VS"] + cfg["C_PUCT"] * P * np.sqrt(nv.sum()) / (1 + nv)
                    if cfg.get("ALLOC") == "uniform":         # round-robin: least-sampled candidate next (balanced SE)
                        sc = -nv.astype(float) - 1e-6 * np.arange(nc)
                    sc[nv >= RS] = -np.inf
                    c = int(np.argmax(sc))
                    if not np.isfinite(sc[c]):
                        break
                    batch.append((c, int(nv[c]))); nv[c] += 1
                # the default must keep pace (paired comparison)
                need = int(max((nv[1:]).max(), 1)) if nc > 1 else 1
                while nv[0] < min(need, RS) and len(batch) < 2 * int(cfg.get("ROUND_K", 1)) * nc:
                    batch.append((0, int(nv[0]))); nv[0] += 1
            if not batch:
                break
            now = clk()
            if fixed:       # exam clock: a fixed simulation budget per step (by 50-step phase), no time checks,
                if rounds > 0 and sims_done + len(batch) > fix_n:   # so paired games stay identical until they fork
                    if not cfg.get("FIX_TRUNC") or sims_done >= fix_n:
                        break
                    batch = batch[:fix_n - sims_done]       # FIX_TRUNC: use the budget exactly (fair across ROUND_K)
            elif per_sim is not None and now - t0 + cfg["PROJ"] * per_sim * len(batch) > cfg["BUDGET"]:
                break
            elif rounds == 0 and now - t0 > 0.4 * cfg["BUDGET"]:
                break
            elif per_sim is not None and time.time() - tw + cfg["PROJ"] * per_wall * len(batch) > cfg["HARD"]:
                break
            elif cfg["START_DL"] is not None and time.time() - tw > cfg["START_DL"]:
                break
            elif cfg["ABORT_DL"] is not None and "stepw" in st and time.time() - tw + \
                    cfg["PROJ"] * (H - 1) * st["stepw"] * len(batch) > cfg["ABORT_DL"]:
                break       # this round could not finish before the abort deadline: do not start it
            nw = time.time()
            nb = len(batch)
            while len(pool) < nb:
                pool.append(FastCarbon())
            sims, fs = [], []
            term_cash = []
            for j, (c, i) in enumerate(batch):
                s_ = pool[j]; s_.load_observation(full)
                s_._tid_counter = [cc + 500 for cc in s_._tid_counter]
                pc_ = list(s_.cash)     # F01: cash carried into the root step (survivor scoring if it ends the game)
                if me == 0:
                    s_.step(cand[c], roots[i])
                else:
                    s_.step(roots[i], cand[c])
                sims.append(s_); fs.append((feat_copy(f), feat_copy(fo)))
                term_cash.append(pc_)
            term = [_terminal(s_) for s_ in sims]     # R02: sims the root step already ended by extinction stay frozen
            depth = 1; aborted = False
            for hh in range(1, H):
                if all(term):
                    break
                if cfg["ABORT_DL"] is not None and not fixed:
                    est = st.get("stepw", 0.0) * nb
                    if time.time() - tw + cfg["PROJ"] * est > cfg["ABORT_DL"]:
                        aborted = True
                        break
                ts_ = time.time()
                xs, auxs, tch = [], [], []
                for s_, (fm, fop) in zip(sims, fs):
                    xm = fm.observe(s_.obs(me)); xs.append(xm); auxs.append(fm.aux); tch.append(st.get("T_now", T))
                # SHARE_OPP: on the first rollout step the opponent's view barely depends on which candidate we
                # played, so its net runs once per simulation index i and every candidate with that i reuses it
                share = bool(cfg.get("SHARE_OPP")) and hh == 1
                orow = {}                                   # sim j -> row of its opponent label
                no = 0; reps = []
                order = list(range(nb))
                if share:                                   # the default branch (c == 0) is the source of each index's reply
                    order.sort(key=lambda j: (batch[j][0] != 0, j))
                obs_q = {}
                if share and int(cfg.get("SHARE_OPP")) >= 2:
                    # level 2: the reusing branches also take the source branch's opponent features (copy, no observe)
                    src = {}
                    for j in order:
                        c_, i_ = batch[j]
                        if i_ in src:
                            fs[j] = (fs[j][0], feat_copy(fs[src[i_]][1])); obs_q[j] = None
                        else:
                            obs_q[j] = fs[j][1].observe(sims[j].obs(1 - me)); src[i_] = j
                else:
                    for j in range(nb):
                        obs_q[j] = fs[j][1].observe(sims[j].obs(1 - me))   # every copy still observes (later steps need it)
                for j in order:
                    (c, i), s_, (fm, fop) = batch[j], sims[j], fs[j]
                    xq = obs_q[j]
                    if share and i in orow.get("_first", {}):
                        orow[j] = orow["_first"][i]; continue
                    xs.append(xq); auxs.append(fop.aux); tch.append(int(ks[i])); orow[j] = nb + no; reps.append((c, i))
                    if share:
                        orow.setdefault("_first", {})[i] = nb + no
                    no += 1
                nz = np.concatenate([np.zeros((nb, F.OWN)), np.stack([rnoise[i, hh - 1] for c, i in reps])])
                tp = np.concatenate([np.zeros(nb), np.full(no, tau)])
                X = stack_x(xs)
                tk, cx = encode_b(rpol, X, np.asarray(tch))
                lab = decode_b(rpol, X, auxs, tk, cx, noise=nz, temp=tp)
                if share:
                    stats["share_saved"] = stats.get("share_saved", 0) + (nb - no)
                for j, (s_, (fm, fop)) in enumerate(zip(sims, fs)):
                    if term[j]:
                        continue     # R02: frozen at extinction -- neither the sim nor its cash may evolve further (leaf pins the terminal value)
                    am = fm.to_actions(lab[j]); ao = fop.to_actions(lab[orow[j]])
                    pc_ = list(s_.cash)     # F01: cash carried into this step (survivor scoring if it ends the game)
                    if me == 0:
                        s_.step(am, ao)
                    else:
                        s_.step(ao, am)
                    if _terminal(s_):
                        term[j] = True; term_cash[j] = pc_; stats["term"] = stats.get("term", 0) + 1
                stats["branch_steps"] += nb
                depth = hh + 1
                sw = (time.time() - ts_) / nb
                st["stepw"] = sw if "stepw" not in st else max(sw, 0.7 * st["stepw"] + 0.3 * sw)
            if aborted:
                stats["aborted"] = stats.get("aborted", 0) + 1
                break
            lft = max(0, left - depth)
            for j, ((c, i), s_) in enumerate(zip(batch, sims)):
                vals[c].append((i, leaf(s_, me, lft, term_cash[j]))); n[c] += 1
            sims_done += nb; rounds += 1
            per_sim = (clk() - now) / nb; per_wall = (time.time() - nw) / nb
        stats["sims"] = stats.get("sims", 0) + sims_done
        stats.setdefault("sims_steps", []).append((step, sims_done))   # per-step budget actually used (calibrates FIX_SIMS)
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
            if cfg.get("PLANTER_KEY") and cmeta[bj] is not None:
                u_ = cmeta[bj][0]; w_ = obs["players"][me][2].get(u_)
                if w_ is not None and w_[2] == "PLANTER":      # a planter's move replaced: did it plant within 3 steps?
                    stats["pk_rep"] = stats.get("pk_rep", 0) + 1
                    P_ = obs["players"][me]; own_ = len(P_[3]); live_ = sum(1 for q in (0, 1) for _, a_ in obs["players"][q][3].values() if a_ < 50)
                    if float(P_[0]) >= (20.0 if own_ == 0 else 20.0 + 5.0 * 1.235 ** live_):
                        stats["pk_rep_cash"] = stats.get("pk_rep_cash", 0) + 1
                    st.setdefault("pk_pend", {})[u_] = step
            return cand[bj]
        return acts

    def _search(obs, x, y, acts, xo, left, t0, tw):
        me = obs["player"]; step = obs["step"]
        f, fo = st["f"], st["fo"]
        keys = key_units(obs, cfg)
        if not keys:
            return acts
        stats["searched"] += 1
        tok, ctx = pol.enc_for(x, st["T_now"])     # R12: original-frame encoding of THIS x (last_enc may be a TTA view)
        uid2slot = {u: s for s, u in enumerate(f.own_uid) if u is not None}
        # ---- root candidate joint actions (b128 re-decode, prefix forced to the default)
        forced, meta = [], []
        for _, uid in keys:
            s = uid2slot.get(uid)
            if s is None or (s >= 10 and (not cfg.get("BASE_KEY") or s in st.get("base_fix", {}) or s in st.get("guard_base", ()))):
                continue
            for m in range(3 if s >= 10 else 5):
                if m == y[s]:
                    continue
                fz = np.full(F.OWN, -1, np.int64)
                fz[:s] = y[:s]; fz[s] = m
                for s_b, v_b in st.get("base_fix", {}).items():
                    if s_b != s:
                        fz[s_b] = v_b
                forced.append(fz); meta.append((uid, m))
        if cfg.get("BASE_KEY"):      # base keys: got candidates / all cut by MAX_BR
            bk_u = {u for _, u in keys if u in uid2slot and uid2slot[u] >= 10}
            if bk_u:
                kept_b = {u for u, _ in meta[:cfg["MAX_BR"] - 1]}
                stats["bk_in"] = stats.get("bk_in", 0) + (1 if bk_u & kept_b else 0)
                stats["bk_cut"] = stats.get("bk_cut", 0) + (1 if bk_u - kept_b else 0)
        if cfg.get("PLANTER_KEY"):   # did a planter key get any candidate, or did MAX_BR cut all of them?
            pk_u = {u for _, u in keys if u in uid2slot and obs["players"][me][2].get(u, (0, 0, ""))[2] == "PLANTER"}
            if pk_u:
                kept = {u for u, _ in meta[:cfg["MAX_BR"] - 1]}
                stats["pk_in"] = stats.get("pk_in", 0) + (1 if pk_u & kept else 0)
                stats["pk_cut"] = stats.get("pk_cut", 0) + (1 if pk_u - kept else 0)
        mbr = cfg["MAX_BR"] if not (cfg.get("MAX_BR_LATE") and obs["step"] >= cfg.get("KEY_LATE_STEP", 999)) else cfg["MAX_BR_LATE"]
        forced = forced[:mbr - 1]; meta = meta[:mbr - 1]
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
        term_cash = []
        for i, (mi, ci) in enumerate(bl):
            a = cand[ci]; ao_ = models[mi]
            s_ = pool[i]; s_.load_observation(full)
            s_._tid_counter = [c + 500 for c in s_._tid_counter]
            pc_ = list(s_.cash)     # F01: cash carried into the root step (survivor scoring if it ends the game)
            if me == 0:
                s_.step(a, ao_)
            else:
                s_.step(ao_, a)
            sims.append(s_); fs.append((feat_copy(f), feat_copy(fo)))
            term_cash.append(pc_)
        term = [_terminal(s_) for s_ in sims]     # R02: sims the root step already ended by extinction stay frozen
        stats["branches"] += nb
        local = None
        if cfg["LOCAL_R"]:
            kp = [obs["players"][me][2][u][0] for _, u in keys]
            local = {w_ for p in (0, 1) for w_, w in obs["players"][p][2].items()
                     if min(F.DIST[w[0], c] for c in kp) <= cfg["LOCAL_R"]}
        H = min(cfg["H"], left)          # R01: see _mcts -- left already counts the remaining transfers
        depth = 1; last_dt = None
        for hh in range(1, H):
            if all(term):
                break     # R02: every branch sim already ended by extinction
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
                if not term[i]:     # R02: frozen at extinction -- neither the sim nor its cash may evolve further
                    pc_ = list(s_.cash)     # F01: cash carried into this step (survivor scoring if it ends the game)
                    if me == 0:
                        s_.step(am, ao)
                    else:
                        s_.step(ao, am)
                    if _terminal(s_):
                        term[i] = True; term_cash[i] = pc_; stats["term"] = stats.get("term", 0) + 1
            stats["branch_steps"] += nb
            depth = hh + 1
            last_dt = clk() - now
            last_wall = time.time() - nw
        stats["depth_sum"] += depth
        if depth < min(cfg["MIN_DEPTH"], H) and not all(term):
            # C01: the MIN_DEPTH guard distrusts shallow HEURISTIC leaves. When every branch
            # sim has reached an exact terminal (all(term)), the leaf values below are the true
            # terminal scores under F01 semantics, not estimates -- the guard must not discard
            # them and fall back to the default action. Any unresolved branch keeps the guard,
            # including an expansion cut short by the budget checks above.
            stats["shallow"] += 1
            return acts
        lft = max(0, left - depth)
        sc = np.array([leaf(s_, me, lft, term_cash[i]) for i, s_ in enumerate(sims)]).reshape(len(models), nc)
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

    MV = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

    def _nxt(pos, mv):
        r, c = divmod(int(pos), F.S); dr, dc = MV[mv]
        return ((r + dr) % F.S) * F.S + (c + dc) % F.S

    def plant_price(obs, me):
        """the engine's planting price: 20 while we own <= 0 trees, else 20 + 5 * 1.235 ** (living trees, both sides)"""
        P = obs["players"]
        mine = len(P[me][3]) if isinstance(P[me][3], dict) else 0
        if mine <= 0:
            return 20.0
        alive = sum(1 for q in (0, 1) if isinstance(P[q][3], dict) for v in P[q][3].values() if int(v[1]) < 50)
        return 20.0 + 5.0 * (1.235 ** alive)

    def tree_gate(obs, acts):
        """TG: no planting when our price is >= TG_MAX, or (from TG_LATE_STEP) above TG_LATE_MAX. Planting happens
        whenever a planter stays on a treeless, non-base cell with the cash, so a gated planter that planned to stay
        there takes one step instead (toward the nearest enemy tree: seizing is unaffected; else any free cell)."""
        me = obs["player"]; P = obs["players"]; step = obs["step"]
        price = plant_price(obs, me)
        if not (price >= cfg.get("TG_MAX", 1e9) or (step >= cfg.get("TG_LATE_STEP", 999) and price > cfg.get("TG_LATE_MAX", 1e9))):
            return acts
        tcells = {int(v[0]) for q in (0, 1) if isinstance(P[q][3], dict) for v in P[q][3].values()}
        etrees = [int(v[0]) for v in P[1 - me][3].values()] if isinstance(P[1 - me][3], dict) else []
        bcells = {int(b) for q in (0, 1) for b in P[q][1].values()}
        out = dict(acts)
        claimed = {}
        for u2, w2 in P[me][2].items():
            mv2 = out.get(u2); c2 = _nxt(w2[0], mv2) if mv2 in MV else int(w2[0])
            claimed[c2] = claimed.get(c2, 0) + 1
        for uid, w in P[me][2].items():
            if w[2] != "PLANTER" or out.get(uid) in MV:
                continue
            pos = int(w[0])
            if pos in tcells or pos in bcells:
                continue                                  # staying here seizes / does nothing: not a plant
            opts = [m for m in MV if not claimed.get(_nxt(pos, m)) and _nxt(pos, m) not in bcells]
            if not opts:
                continue
            key = (lambda m: min(F.DIST[_nxt(pos, m), e] for e in etrees)) if etrees else (lambda m: 0)
            pick = min(opts, key=lambda m: (key(m), m))
            claimed[pos] -= 1; claimed[_nxt(pos, pick)] = claimed.get(_nxt(pos, pick), 0) + 1
            out[uid] = pick; stats["tg"] = stats.get("tg", 0) + 1
        return out

    def recruit_gate(obs, acts):
        """RG: from RG_STEP, with >= RG_PL planters in play and >= RG_W workers, a base that would recruit a planter
        recruits a collector instead (the engine skips it if the cash is short)."""
        me = obs["player"]; P = obs["players"]
        mine = list(P[me][2].values())
        if obs["step"] < cfg.get("RG_STEP", 60) or len(mine) < cfg.get("RG_W", 8) or \
                sum(1 for w in mine if w[2] == "PLANTER") < cfg.get("RG_PL", 2):
            return acts
        out = dict(acts)
        for b in P[me][1]:
            if out.get(b) == "RECPLANTER":
                out[b] = "RECCOLLECTOR"; stats["rg"] = stats.get("rg", 0) + 1
        return out

    def planter_explore(obs, acts):
        """opening information: PX_MODE 'greedy' -- a planter that has not planted yet (step < PX_END) whose planned move
        reveals no never-seen cell (mirrors included: the start map is 4-fold symmetric) takes the move that reveals the
        most, if that is >= PX_MIN cells; a planned plant (stay) is never overridden. PX_MODE 'snake' -- the first planter
        follows a fixed snake through its base quadrant from PX_S0 for 15 steps, then the net plays."""
        me = obs["player"]; P = obs["players"]; f = st["f"]
        if f is None:
            return acts
        seen = f.last_step >= 0
        unseen = ~seen[F.MIRROR].any(1)                 # no copy of this cell's symmetry class seen yet
        out = dict(acts); step = obs["step"]
        pls = sorted((u, w) for u, w in P[me][2].items() if w[2] == "PLANTER")
        if cfg.get("PX_MODE") == "snake":
            s0 = cfg.get("PX_S0", 8)
            if pls and s0 <= step < s0 + 15 and P[me][1]:
                b0 = int(sorted(P[me][1].values())[0]); br, bc = divmod(b0, F.S); dr = 1 if br < 7 else -1; dc = 1 if bc < 7 else -1
                i = step - s0; leg, j = divmod(i, 5); j = j if leg % 2 == 0 else 4 - j
                tgt = ((br + dr * (1 + 3 * leg)) % F.S) * F.S + (bc + dc * j) % F.S
                u, w = pls[0]; pos = int(w[0])
                if pos != tgt:
                    mv = min(MV, key=lambda m: (F.DIST[_nxt(pos, m), tgt], m))
                    if F.DIST[_nxt(pos, mv), tgt] < F.DIST[pos, tgt]:
                        out[u] = mv; stats["px"] = stats.get("px", 0) + 1
            return out
        planted = st.setdefault("px_planted", set())
        tpos = {int(t[0]) for t in P[me][3].values()}
        for u, w in pls:
            pos = int(w[0])
            if pos in tpos:
                planted.add(u)
            if (u in planted and not cfg.get("PX_ANY")) or step >= cfg.get("PX_END", 30):
                continue
            cur = out.get(u)
            if cur not in MV:                            # staying = planting here: leave it
                continue
            gain = lambda m: int(unseen[F.DEST[_nxt(pos, m), 1:]].sum())
            g0 = gain(cur); best = max(MV, key=lambda m: (gain(m), m == cur))
            if (gain(best) - g0 >= cfg["PX_DIFF"]) if cfg.get("PX_DIFF") else (g0 == 0 and gain(best) >= cfg.get("PX_MIN", 2)):
                out[u] = best; stats["px"] = stats.get("px", 0) + 1
        return out

    def planter_avoid(obs, acts):
        """PLANTER_AVOID: a planter that stays put this step WITHOUT planting or seizing (own tree under it, or no
        cash for the plant / seize) and has an enemy planter in its 8-neighbourhood moves to a cell with no enemy
        planter or collector within one step (planter vs planter collisions delete both). Stays if no such cell."""
        me = obs["player"]; P = obs["players"]
        obases = {int(b) for b in P[me][1].values()}
        cash = float(P[me][0]) + sum(float(w[1]) for u_, w in P[me][2].items()     # deposits land before planting
                                     if w[2] == "COLLECTOR" and acts.get(u_) in MV and _nxt(w[0], acts.get(u_)) in obases)
        en = [(int(w[0]), w[2]) for w in P[1 - me][2].values()]
        epl = [p_ for p_, k in en if k == "PLANTER"]
        if not epl:
            return acts
        own_t = {int(v[0]) for v in P[me][3].values()} if isinstance(P[me][3], dict) else set()
        opp_t = {int(v[0]) for v in P[1 - me][3].values()} if isinstance(P[1 - me][3], dict) else set()
        ebase = {int(b) for b in P[1 - me][1].values()}
        cheb = lambda a, b: max(min(abs(a // 15 - b // 15), 15 - abs(a // 15 - b // 15)), min(abs(a % 15 - b % 15), 15 - abs(a % 15 - b % 15)))
        out = dict(acts); pp = plant_price(obs, me)
        claimed = {}
        for u2, w2 in P[me][2].items():
            mv2 = out.get(u2); c2 = _nxt(w2[0], mv2) if mv2 in MV else int(w2[0]); claimed[c2] = claimed.get(c2, 0) + 1
        for uid, w in P[me][2].items():
            if w[2] != "PLANTER" or out.get(uid) in MV:
                continue
            pos = int(w[0])
            busy = (pos not in own_t) and ((pos in opp_t and cash >= 20.0) or (pos not in opp_t and cash >= pp))
            if cfg.get("PA_ALL"):
                busy = False          # collisions resolve before planting: a planter about to plant / seize dies too
            near = (lambda e: F.DIST[pos, e] <= 1) if cfg.get("PA_MANH") else (lambda e: cheb(pos, e) <= 1)
            if busy or not any(near(e) for e in epl):
                continue
            safe = [mv for mv in MV if not claimed.get(_nxt(pos, mv)) and _nxt(pos, mv) not in ebase
                    and not any(cheb(_nxt(pos, mv), e) <= 1 for e, _ in en)]
            if safe:
                mv = safe[0]; out[uid] = mv; claimed[pos] -= 1; claimed[_nxt(pos, mv)] = claimed.get(_nxt(pos, mv), 0) + 1
                stats["pa_moves"] = stats.get("pa_moves", 0) + 1
        return out

    def endgame_home(obs, acts):
        """t >= EH_STEP: a loaded collector that can only just reach a base in time must step toward it"""
        me = obs["player"]; left = 299 - obs["step"]; P = obs["players"]
        bases = [int(b) for b in P[me][1].values()]
        if not bases:
            return acts
        en = [e for e in P[1 - me][2].values() if e[2] == "COLLECTOR"]
        out = dict(acts)
        # cells our other units are about to enter (or stay on): a forced car never steps into them (no self-collisions)
        claimed = {}                                                      # cell -> how many own units plan to be there
        for u2, w2 in P[me][2].items():
            mv2 = out.get(u2); c2 = _nxt(w2[0], mv2) if mv2 in MV else int(w2[0])
            claimed[c2] = claimed.get(c2, 0) + 1
        for uid, w in sorted(P[me][2].items(), key=lambda kv: -kv[1][1]):      # heaviest car chooses first
            pos, cargo, kind = w
            if kind != "COLLECTOR" or cargo < cfg["MINC"]:
                continue
            d = min(F.DIST[pos, b] for b in bases)
            slack = cfg.get("EH_HEAVY_SLACK", 1) if cargo >= cfg.get("EH_HEAVY", 1e9) else 1
            if cfg.get("EH_HEAVY_STEP") is not None and obs["step"] >= cfg["EH_HEAVY_STEP"] and cargo >= cfg.get("EH_HEAVY", 1e9):
                slack = left                     # heavy car from this step on: go home as soon as it is reachable at all
            if d == 0 or not (left - slack <= d <= left):
                continue
            stats.setdefault("eh_act", []).append((obs["step"], uid))      # car is under the go-home rule this step
            if cfg.get("EH_SHADOW"):                                          # record only, never change a move
                continue
            tgt = bases; d0 = d                                              # d0: trigger distance (nearest base)
            if cfg.get("EH_BASE_PICK") and len(bases) > 1:
                # of the bases still reachable in time, head for the one with the fewest lighter enemy cars within 2
                # cells of it (ambush at the base mouth); ties -> nearer. No lookahead: just a choice of target.
                reach = [b for b in bases if F.DIST[pos, b] <= left]
                amb = lambda b: sum(1 for e in en if F.DIST[b, e[0]] <= 2 and e[1] <= cargo)
                bt = min(reach, key=lambda b: (amb(b), F.DIST[pos, b]))
                if bt != min(bases, key=lambda b: F.DIST[pos, b]):
                    stats["eh_alt"] = stats.get("eh_alt", 0) + 1
                tgt = [bt]; d = F.DIST[pos, bt]
            cur = out.get(uid)
            c0 = _nxt(pos, cur) if cur in MV else int(pos)                    # its own planned cell: remove ONE claim only
            claimed[c0] -= 1
            if cur in MV and min(F.DIST[_nxt(pos, cur), b] for b in tgt) < d and (not claimed.get(_nxt(pos, cur)) or _nxt(pos, cur) in bases):
                claimed[_nxt(pos, cur)] = claimed.get(_nxt(pos, cur), 0) + 1
                k = "eh_edge_ok" if d0 >= left - 1 else "eh_heavy_ok"  # rule fired but the planned move already heads home
                stats[k] = stats.get(k, 0) + 1; continue
            # only directions that bring it closer; inside those prefer no lighter enemy car adjacent; never give up home
            opts = [mv for mv in MV if min(F.DIST[_nxt(pos, mv), b] for b in tgt) < d and (not claimed.get(_nxt(pos, mv)) or _nxt(pos, mv) in bases)]
            safe = [mv for mv in opts if not any(F.DIST[_nxt(pos, mv), e[0]] <= 1 and e[1] <= cargo for e in en)]
            pick = (safe or opts or [None])[0]
            if not pick:
                claimed[c0] += 1                                              # keeps its planned cell
            if pick:
                out[uid] = pick; claimed[_nxt(pos, pick)] = claimed.get(_nxt(pos, pick), 0) + 1; stats["eh"] = stats.get("eh", 0) + 1
                k = "eh_edge" if d0 >= left - 1 else "eh_heavy"       # heavy = fired only because of the wider slack
                stats[k] = stats.get(k, 0) + 1
                stats.setdefault("eh_list", []).append((obs["step"], uid))
        return out

    def safe_entry(obs, acts):
        """a loaded collector (>= SE_MINC) within SE_R of our base, with a lighter-or-equal enemy collector within
        SE_NEAR: step only onto cells no such enemy can reach next step (our own base cells are always safe: cargo
        delivered there is ours whatever collides), the one nearest our base; keep the planned move if it is safe
        and no farther. Provably safe for one step; no search, no estimate."""
        me = obs["player"]; P = obs["players"]
        bases = [int(b) for b in P[me][1].values()]
        if not bases:
            return acts
        ebases = {int(b) for b in P[1 - me][1].values()}
        en = [e for e in P[1 - me][2].values() if e[2] == "COLLECTOR"]
        out = dict(acts)
        claimed = {}
        for u2, w2 in P[me][2].items():
            mv2 = out.get(u2); c2 = _nxt(w2[0], mv2) if mv2 in MV else int(w2[0])
            claimed[c2] = claimed.get(c2, 0) + 1
        bset = set(bases)
        wait = cfg.get("SE_MODE") == "wait"       # stall trigger: only a car that has sat loaded near home SE_WAIT steps
        if wait:
            wc = st.setdefault("se_wait", {})
            for u_, w_ in P[me][2].items():
                near_ = w_[2] == "COLLECTOR" and w_[1] >= cfg.get("SE_MINC", 300.0) and 1 <= min(F.DIST[w_[0], b] for b in bases) <= cfg.get("SE_R", 6)
                wc[u_] = wc.get(u_, 0) + 1 if near_ else 0
            # team stall: >= SE_TEAM on board in total and no own collector reached a base cell (no delivery) for SE_WAIT steps
            tot_ = sum(w_[1] for w_ in P[me][2].values() if w_[2] == "COLLECTOR")
            home_ = any(w_[2] == "COLLECTOR" and int(w_[0]) in bset for w_ in P[me][2].values())
            st["se_team"] = st.get("se_team", 0) + 1 if cfg.get("SE_TEAM") and tot_ >= cfg["SE_TEAM"] and not home_ else 0
            if cfg.get("SE_TEAM"):         # latency: steps from the trigger to the first delivery (or None if the game ended first)
                if st["se_team"] == cfg.get("SE_WAIT", 8) and st.get("se_t0") is None:
                    st["se_t0"] = obs["step"]
                if home_ and st.get("se_t0") is not None:
                    stats.setdefault("se_lat", []).append(obs["step"] - st["se_t0"]); st["se_t0"] = None
                if obs["step"] >= 298 and st.get("se_t0") is not None:
                    stats.setdefault("se_lat", []).append(None); st["se_t0"] = None
        for uid, w in sorted(P[me][2].items(), key=lambda kv: -kv[1][1]):       # heaviest first
            pos, cargo, kind = w
            if kind != "COLLECTOR" or cargo < cfg.get("SE_MINC", 150.0):
                continue
            dpos = min(F.DIST[pos, b] for b in bases)
            if not (1 <= dpos <= cfg.get("SE_R", 3)):
                continue
            th = [int(e[0]) for e in en if e[1] <= cargo and F.DIST[pos, e[0]] <= cfg.get("SE_NEAR", 4)]
            if wait:
                if (st["se_team"] if cfg.get("SE_TEAM") else st["se_wait"].get(uid, 0)) < cfg.get("SE_WAIT", 8):
                    continue
            elif not th:
                continue
            stats["se_seen"] = stats.get("se_seen", 0) + 1
            danger = {int(c) for e in th for c in F.DEST[e]}                    # enemy stays or steps one cell
            cur = out.get(uid); c0 = _nxt(pos, cur) if cur in MV else int(pos)
            claimed[c0] -= 1
            def ok(c):
                return (c in bset or c not in danger) and c not in ebases and (c in bset or not claimed.get(c))
            opts = [(min(F.DIST[c, b] for b in bases), i, mv, c) for i, (mv, c) in
                    enumerate([(None, int(pos))] + [(mv, _nxt(pos, mv)) for mv in MV]) if ok(c)]
            if not opts:
                claimed[c0] += 1; stats["se_none"] = stats.get("se_none", 0) + 1
                continue
            dbest = min(o[0] for o in opts)
            if ok(c0) and (cfg.get("SE_MODE") == "veto" or min(F.DIST[c0, b] for b in bases) <= dbest):   # veto: a safe planned move stands
                claimed[c0] += 1; stats["se_ok"] = stats.get("se_ok", 0) + 1
                continue
            _, _, mv, c = min(opts)
            if mv is None:
                out.pop(uid, None)
            else:
                out[uid] = mv
            claimed[c] = claimed.get(c, 0) + 1; stats["se"] = stats.get("se", 0) + 1
        return out

    def camp_bust(obs, acts):
        """CB_STEP..: an enemy collector parked within CB_ER (1) of our base while our loaded cars wait nearby -> the nearest
        light own collector (cargo <= CB_MAXC and <= the camper's; least cargo wins a collision, equal = both die)
        steps toward it. Re-aimed every step, so the chase follows a camper that shifts.
        The wide setting (from step 100 / 150, CB_R 9) was examined on 2026-11-11 and is negative: 18-27 chases a game.
        Narrow, endgame-only setting modelled on Master3s (2.3 own cars a game traded within 2 cells of its bases in the
        last 25 steps, each followed by a loaded car going in):
            CB_STEP 270, CB_ER 2, CB_R 4, CB_MAXC 5, CB_LOAD 100, CB_D 4"""
        me = obs["player"]; P = obs["players"]
        bases = [int(b) for b in P[me][1].values()]
        if not bases:
            return acts
        own = [(u, w) for u, w in P[me][2].items() if w[2] == "COLLECTOR"]
        out = dict(acts); used = set()
        claimed = {}
        for u2, w2 in P[me][2].items():
            mv2 = out.get(u2); c2 = _nxt(w2[0], mv2) if mv2 in MV else int(w2[0])
            claimed[c2] = claimed.get(c2, 0) + 1
        for e in P[1 - me][2].values():
            if e[2] != "COLLECTOR" or min(F.DIST[e[0], b] for b in bases) > cfg.get("CB_ER", 1):
                continue
            b0 = min(bases, key=lambda b: F.DIST[e[0], b])
            if not any(w[1] >= cfg.get("CB_LOAD", 200.0) and F.DIST[w[0], b0] <= cfg.get("CB_R", 9) for _, w in own):
                continue
            stats["cb_seen"] = stats.get("cb_seen", 0) + 1
            cands = [(F.DIST[w[0], e[0]], w[1], u, w) for u, w in own if u not in used and w[1] <= min(cfg.get("CB_MAXC", 50.0), e[1])
                     and 0 < F.DIST[w[0], e[0]] <= cfg.get("CB_D", 6)]
            if not cands:
                continue
            d, _, u, w = min(cands)
            cur = out.get(u); c0 = _nxt(w[0], cur) if cur in MV else int(w[0])
            opts = [mv for mv in MV if F.DIST[_nxt(w[0], mv), e[0]] < d and (_nxt(w[0], mv) == int(e[0]) or not claimed.get(_nxt(w[0], mv)))]
            if not opts:
                continue
            pick = opts[0]; used.add(u)
            if pick != cur:
                claimed[c0] -= 1; claimed[_nxt(w[0], pick)] = claimed.get(_nxt(w[0], pick), 0) + 1
                out[u] = pick; stats["cb"] = stats.get("cb", 0) + 1
            else:
                stats["cb_ok"] = stats.get("cb_ok", 0) + 1
        return out

    def agent(obs, config=None):            # root fallback: any exception -> empty action, never a failed step
        try:
            a = _inner(obs, config)
        except Exception:
            stats["root_errors"] = stats.get("root_errors", 0) + 1
            return {}
        if cfg.get("REC_CUT_ALL") is not None or cfg.get("REC_CUT_PLANTER") is not None:
            # measurement only (upper bound of a late recruiting discipline): no planter recruits from REC_CUT_PLANTER,
            # no recruits at all from REC_CUT_ALL; nothing else changes
            try:
                bids = set(obs["players"][obs["player"]][1])
                for b_ in [k for k in list(a) if k in bids]:
                    if obs["step"] >= cfg.get("REC_CUT_ALL", 999) or (obs["step"] >= cfg.get("REC_CUT_PLANTER", 999) and a[b_] == "RECPLANTER"):
                        a.pop(b_); stats["rec_cut"] = stats.get("rec_cut", 0) + 1
            except Exception:
                stats["errors"] += 1
        if cfg.get("OPEN_PL") and obs["step"] <= cfg.get("OPEN_PL_END", 15):
            try:                                 # opening sensors: A = first recruit is a planter; B = >= 2 planters by OPEN_PL_END
                me_ = obs["player"]; P_ = obs["players"][me_]; mine = list(P_[2].values())
                npl = sum(1 for w in mine if w[2] == "PLANTER"); bids = sorted(P_[1]); cash = float(P_[0])
                rec_pl = sum(1 for b in bids if a.get(b) == "RECPLANTER")
                if "A" in cfg["OPEN_PL"] and not st.get("opl_first"):
                    recs = [b for b in bids if a.get(b) in ("RECPLANTER", "RECCOLLECTOR")]
                    if recs:
                        st["opl_first"] = True
                        if not npl and not rec_pl:
                            a[recs[0]] = "RECPLANTER"; rec_pl = 1; stats["opl"] = stats.get("opl", 0) + 1
                if "B" in cfg["OPEN_PL"] and npl + rec_pl < 2 and len(mine) < 10:
                    spend = 30.0 * sum(1 for b in bids if a.get(b) in ("RECPLANTER", "RECCOLLECTOR"))
                    occ = {int(w[0]) for w in mine}
                    for b in bids:
                        if npl + rec_pl >= 2 or cash - spend < 30.0:
                            break
                        if a.get(b) == "RECCOLLECTOR":
                            a[b] = "RECPLANTER"; rec_pl += 1; stats["opl"] = stats.get("opl", 0) + 1
                        elif a.get(b) is None and int(P_[1][b]) not in occ:
                            a[b] = "RECPLANTER"; rec_pl += 1; spend += 30.0; stats["opl"] = stats.get("opl", 0) + 1
            except Exception:
                stats["opl_errors"] = stats.get("opl_errors", 0) + 1
        if cfg.get("TG_MAX") is not None or cfg.get("TG_LATE_STEP") is not None:
            try:
                a = tree_gate(obs, a)
            except Exception:
                stats["tg_errors"] = stats.get("tg_errors", 0) + 1
        if cfg.get("RG_STEP") is not None:
            try:
                a = recruit_gate(obs, a)
            except Exception:
                stats["rg_errors"] = stats.get("rg_errors", 0) + 1
        if cfg.get("PX_MODE") and obs["step"] < max(cfg.get("PX_END", 30), cfg.get("PX_S0", 8) + 15):
            try:
                a = planter_explore(obs, a)
            except Exception:
                stats["px_errors"] = stats.get("px_errors", 0) + 1
        if cfg.get("PLANTER_AVOID") and obs["step"] <= cfg.get("PA_STEP", 999):
            try:
                a = planter_avoid(obs, a)
            except Exception:
                stats["pa_errors"] = stats.get("pa_errors", 0) + 1
        if cfg.get("SE_STEP") is not None and obs["step"] >= cfg["SE_STEP"]:
            try:
                a = safe_entry(obs, a)
            except Exception:
                stats["se_errors"] = stats.get("se_errors", 0) + 1
        if cfg.get("CB_STEP") is not None and obs["step"] >= cfg["CB_STEP"]:
            try:
                a = camp_bust(obs, a)
            except Exception:
                stats["cb_errors"] = stats.get("cb_errors", 0) + 1
        if cfg.get("EH_STEP") is not None and obs["step"] >= cfg["EH_STEP"]:
            try:                                # rule layer has its own fallback: a bug here keeps the searched actions
                a = endgame_home(obs, a)
            except Exception:
                stats["eh_errors"] = stats.get("eh_errors", 0) + 1
        return a
    agent.stats = stats
    agent.state = st
    agent.log = log
    agent.cfg = cfg
    return agent
