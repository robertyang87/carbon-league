"""cnet2_np -- pure NumPy inference of cnet2_model.CNet2 (single observation).

Policy(weights dict).act(x, aux, teacher, greedy=True) -> (labels [OWN], logits [OWN,5], value)
Autoregressive decoding over own slots with cnet2_feat.Prefix features.
"""

import json
import math

import numpy as np

import cnet2_feat as F

NEG = np.float32(-1e9)


def _wmat(w):
    """[O,C,3,3] -> [9*C, O] for channels-last im2col with shift order (dr,dc) in -1..1."""
    O, C = w.shape[:2]
    return np.ascontiguousarray(w.transpose(2, 3, 1, 0).reshape(9 * C, O))


SHIFTS = [(ki - 1, kj - 1) for ki in range(3) for kj in range(3)]
# im2col gather index: for output cell i and shift k, input cell
_r, _c = F.ROW, F.COL
IM2COL = np.stack([((_r + dr) % F.S) * F.S + (_c + dc) % F.S for dr, dc in SHIFTS], 1)   # [N,9]



def _tk(t, s, x):
    """temperature for unit slot s: a scalar, or (car, planter, base) for per-kind sampling (exam diagnostics)"""
    if not isinstance(t, (tuple, list)):
        return t
    return t[2] if s >= 10 else (t[1] if int(x["tok_kind"][s]) == 1 else t[0])

class Policy:
    def __init__(self, w):
        meta = w.get("meta")
        if meta is not None and not isinstance(meta, dict):
            meta = json.loads(str(meta))
        self.cfg = meta["cfg"] if meta else None
        self.w = {k: np.asarray(v, np.float32) for k, v in w.items() if k != "meta"}
        w = self.w
        self.NB = len({k.split(".")[1] for k in w if k.startswith("blocks.")})
        self.L = len({k.split(".")[1] for k in w if k.startswith("attn.")})
        self.TA = len({k.split(".")[1] for k in w if k.startswith("tattn.")})     # cell attention in the trunk
        if self.TA:
            step = max(self.NB // self.TA, 1)
            self.ta = {min(step * (j + 1), self.NB) - 1: j for j in range(self.TA)}
            self.TNH = w["tattn.0.rel.weight"].shape[1]
            r_, c_ = np.arange(F.N) // F.S, np.arange(F.N) % F.S
            self.rel_cell = ((r_[:, None] - r_[None, :]) % F.S) * F.S + (c_[:, None] - c_[None, :]) % F.S
            self.tbias = [np.ascontiguousarray(w[f"tattn.{j}.rel.weight"][self.rel_cell].transpose(2, 0, 1))
                          for j in range(self.TA)]
        else:
            self.ta = {}
        self.wm = {k[:-7]: _wmat(v) for k, v in w.items() if k.endswith(".weight") and v.ndim == 4 and v.shape[-1] == 3}
        self.C = w["stem.weight"].shape[0]
        self.G = self.cfg["G"] if self.cfg else 8
        self.NH = w["attn.0.rel.weight"].shape[1] if self.L else 1
        for k in [k for k in w if k.endswith(".weight") and w[k].ndim == 2]:
            if not k.startswith(("attn.", "teach", "aemb")) or k.endswith(("qkv.weight", "out.weight", "f1.weight", "f2.weight")):
                self.w[k + "T"] = np.ascontiguousarray(w[k].T)
        self.gih = np.ascontiguousarray(w["gru.weight_ih"].T)
        self.ghh = np.ascontiguousarray(w["gru.weight_hh"].T)

    def lin(self, x, n):
        return x @ self.w[n + ".weightT"] + self.w[n + ".bias"]

    def conv(self, x, n):
        """x [N, C] channels-last -> [N, O]"""
        cols = x[IM2COL].reshape(F.N, -1)
        return cols @ self.wm[n] + self.w[n + ".bias"]

    def gn(self, x, n):
        N_, C = x.shape
        g = x.reshape(N_, self.G, C // self.G)
        m = g.mean(axis=(0, 2), keepdims=True)
        v = ((g - m) ** 2).mean(axis=(0, 2), keepdims=True)
        y = ((g - m) / np.sqrt(v + 1e-5)).reshape(N_, C)
        return y * self.w[n + ".weight"] + self.w[n + ".bias"]

    def ln(self, x, n):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * self.w[n + ".weight"] + self.w[n + ".bias"]

    @staticmethod
    def tok_features(x):
        kind = np.eye(6, dtype=np.float32)[x["tok_kind"]]
        lc = np.minimum(np.log1p(np.maximum(x["tok_cargo"], 0)) / F.LOGC, 1.5).astype(np.float32)
        pos = x["tok_pos"]
        r = (pos // F.S) / 14.0; c = (pos % F.S) / 14.0
        hc = x["hist_cell"]
        valid = (hc >= 0).astype(np.float32)
        hcc = np.maximum(hc, 0)
        dr = ((hcc // F.S) - (pos // F.S)[:, None]) % F.S
        dc = ((hcc % F.S) - (pos % F.S)[:, None]) % F.S
        dr = np.where(dr > 7, dr - F.S, dr) / 7.0
        dc = np.where(dc > 7, dc - F.S, dc) / 7.0
        hl = np.minimum(np.log1p(np.maximum(x["hist_cargo"], 0)) / F.LOGC, 1.5)
        mv = np.eye(6, dtype=np.float32)[x["hist_move"]]
        h = np.concatenate([valid[..., None], (dr * valid)[..., None], (dc * valid)[..., None],
                            (hl * valid)[..., None], mv * valid[..., None]], -1)
        return np.concatenate([kind, lc[:, None], r[:, None], c[:, None], h.reshape(F.T, -1)], -1).astype(np.float32)

    def encode(self, x, teacher):
        w = self.w
        m = (x["maps"].astype(np.float32) / np.float32(255.0)).T.copy()       # [N,P]
        h = np.maximum(self.gn(self.conv(m, "stem"), "stem_n"), 0)
        for i in range(self.NB):
            n = f"blocks.{i}"
            t = np.maximum(self.gn(self.conv(h, n + ".c1"), n + ".n1"), 0)
            h = np.maximum(h + self.gn(self.conv(t, n + ".c2"), n + ".n2"), 0)
            if i in self.ta:
                h = self.cell_attn(h, self.ta[i])
        self.trunk = h
        pos = x["tok_pos"]
        tf = self.tok_features(x)
        t = self.lin(np.maximum(self.lin(tf, "tok1"), 0), "tok2")
        tok = self.lin(np.concatenate([h[pos], t], -1), "fuse")
        mask = x["tok_mask"]
        mf_ = mask.astype(np.float32)[:, None]
        tok = tok * mf_
        r = pos // F.S; c = pos % F.S
        rel = ((r[:, None] - r[None, :]) % F.S) * F.S + (c[:, None] - c[None, :]) % F.S
        D = tok.shape[1]; NH = self.NH; dh = D // NH
        for li in range(self.L):
            n = f"attn.{li}"
            qkv = self.lin(tok, n + ".qkv").reshape(F.T, 3, NH, dh).transpose(1, 2, 0, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            sc = q @ k.transpose(0, 2, 1) / np.float32(dh ** 0.5) + w[n + ".rel.weight"][rel].transpose(2, 0, 1)
            sc = np.where(mask[None, None, :], sc, NEG)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            a = (sc @ v).transpose(1, 0, 2).reshape(F.T, D)
            tok = self.ln(tok + self.lin(a, n + ".out"), n + ".n1")
            tok = self.ln(tok + self.lin(np.maximum(self.lin(tok, n + ".f1"), 0), n + ".f2"), n + ".n2")
            tok = tok * mf_
        tmean = tok.sum(0) / max(mask.sum(), 1)
        s = self.lin(np.maximum(self.lin(x["scal"], "sc1"), 0), "sc2")
        ctx = np.maximum(self.lin(np.concatenate([h.mean(0), tmean, s, w["teach.weight"][teacher]]), "ctx"), 0)
        return tok, ctx

    def cell_attn(self, x, j):
        """cnet2_model.CellAttn on x [N, C] (channels-last)."""
        n = f"tattn.{j}"; C = x.shape[1]; NH = self.TNH; dh = C // NH
        qkv = self.lin(self.ln(x, n + ".n1"), n + ".qkv").reshape(F.N, 3, NH, dh).transpose(1, 2, 0, 3)
        sc = qkv[0] @ qkv[1].transpose(0, 2, 1) / np.float32(dh ** 0.5) + self.tbias[j]
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        sc /= sc.sum(-1, keepdims=True)
        x = x + self.lin((sc @ qkv[2]).transpose(1, 0, 2).reshape(F.N, C), n + ".out")
        return x + self.lin(np.maximum(self.lin(self.ln(x, n + ".n2"), n + ".f1"), 0), n + ".f2")

    def gru(self, x, h):
        w = self.w
        gi = x @ self.gih + w["gru.bias_ih"]
        gh = h @ self.ghh + w["gru.bias_hh"]
        D = h.shape[0]
        r = 1 / (1 + np.exp(-(gi[:D] + gh[:D])))
        z = 1 / (1 + np.exp(-(gi[D:2 * D] + gh[D:2 * D])))
        n = np.tanh(gi[2 * D:] + r * gh[2 * D:])
        return (1 - z) * n + z * h

    def act(self, x, aux, teacher=0, forced=None, sample_rng=None, temp=1.0, hook=None, sample_kinds=None):
        """forced: optional labels to teacher-force (parity tests).
        hook(s, logits, prefix) -> optional action override (rule layer)."""
        w = self.w
        tok, ctx = self.encode(x, teacher)
        value = math.tanh(float(ctx @ w["value.weightT"][:, 0] + w["value.bias"][0]))
        h = self.lin(ctx, "h0")
        prev = 8
        pf = F.Prefix(x, aux)
        labels = np.full(F.OWN, -1, np.int64)
        logits = np.full((F.OWN, 5), NEG, np.float32)
        for s in range(F.OWN):
            if not x["tok_mask"][s]:
                continue
            uf, mf, legal = pf.feats(s)
            inp = self.lin(np.concatenate([tok[s], ctx, uf, mf.reshape(-1)]), "din") + w["aemb.weight"][prev]
            h = self.gru(inp, h)
            base = self.lin(h, "whead") if s < 10 else np.concatenate([self.lin(h, "bhead"), np.zeros(2, np.float32)])
            lg = base + self.lin(mf, "V") @ self.lin(h, "U")
            lg = np.where(legal, lg, NEG).astype(np.float32)
            logits[s] = lg
            if forced is not None:
                a = int(forced[s])
            elif sample_rng is not None and (sample_kinds is None
                                             or (s >= 10 and "base" in sample_kinds)
                                             or (s < 10 and int(x["tok_kind"][s]) == 1 and "planter" in sample_kinds)):
                z = lg / _tk(temp, s, x); p = np.exp(z - z.max()); p /= p.sum()   # sample_kinds: only these units are sampled
                a = int(sample_rng.choice(5, p=p))
            else:
                a = int(lg.argmax())
            if hook is not None:
                o = hook(s, lg, pf, a)
                if o is not None:
                    a = int(o)
            labels[s] = a
            pf.advance(s, a)
            prev = a + (0 if s < 10 else 5)
        return labels, logits, value


class Ensemble:
    """several Policy nets as one: every unit's move logits are the mean over the members, the chosen action is fed
    back to all of them (same autoregressive prefix). Drop-in for Policy.act; costs one trunk per member."""

    def __init__(self, pols):
        self.pols = list(pols)
        self.w = self.pols[0].w; self.cfg = self.pols[0].cfg

    def act(self, x, aux, teacher=0, forced=None, sample_rng=None, temp=1.0, hook=None, sample_kinds=None):
        P = self.pols; K = len(P)
        enc = [p.encode(x, teacher) for p in P]
        value = float(np.mean([math.tanh(float(c @ p.w["value.weightT"][:, 0] + p.w["value.bias"][0])) for p, (_, c) in zip(P, enc)]))
        hs = [p.lin(c, "h0") for p, (_, c) in zip(P, enc)]
        prev = 8
        pf = F.Prefix(x, aux)
        labels = np.full(F.OWN, -1, np.int64)
        logits = np.full((F.OWN, 5), NEG, np.float32)
        for s in range(F.OWN):
            if not x["tok_mask"][s]:
                continue
            uf, mf, legal = pf.feats(s)
            acc = np.zeros(5, np.float32)
            for k, p in enumerate(P):
                tok, ctx = enc[k]
                inp = p.lin(np.concatenate([tok[s], ctx, uf, mf.reshape(-1)]), "din") + p.w["aemb.weight"][prev]
                hs[k] = p.gru(inp, hs[k])
                base = p.lin(hs[k], "whead") if s < 10 else np.concatenate([p.lin(hs[k], "bhead"), np.zeros(2, np.float32)])
                acc += base + p.lin(mf, "V") @ p.lin(hs[k], "U")
            lg = np.where(legal, acc / K, NEG).astype(np.float32)
            logits[s] = lg
            if forced is not None:
                a = int(forced[s])
            elif sample_rng is not None and (sample_kinds is None
                                             or (s >= 10 and "base" in sample_kinds)
                                             or (s < 10 and int(x["tok_kind"][s]) == 1 and "planter" in sample_kinds)):
                z = lg / _tk(temp, s, x); pr = np.exp(z - z.max()); pr /= pr.sum()
                a = int(sample_rng.choice(5, p=pr))
            else:
                a = int(lg.argmax())
            if hook is not None:
                o = hook(s, lg, pf, a)
                if o is not None:
                    a = int(o)
            labels[s] = a
            pf.advance(s, a)
            prev = a + (0 if s < 10 else 5)
        return labels, logits, value


def sym_view(x, g):
    """x in symmetry frame g (cnet2_feat.SYM_CELL/SYM_MOVE; all are involutions)."""
    cm, mm = F.SYM_CELL[g], F.SYM_MOVE[g]
    mm6 = np.concatenate([mm, [5]])
    y = dict(x)
    y["maps"] = x["maps"][:, cm]
    y["tok_pos"] = cm[x["tok_pos"]]
    hc = x["hist_cell"]
    y["hist_cell"] = np.where(hc >= 0, cm[np.maximum(hc, 0)], hc)
    y["hist_move"] = mm6[x["hist_move"]]
    return y


def act_tta(pol, x, aux, teacher=0, k=2, hook=None):
    """Symmetry ensemble: average move logits over k frames (identity first), shared autoregressive choices."""
    w = pol.w
    views = []
    for g in range(k):
        tok, ctx = pol.encode(sym_view(x, g) if g else x, teacher)
        views.append([tok, ctx, pol.lin(ctx, "h0")])
    pf = F.Prefix(x, aux)
    prev = [8] * k
    labels = np.full(F.OWN, -1, np.int64)
    logits = np.full((F.OWN, 5), NEG, np.float32)
    for s in range(F.OWN):
        if not x["tok_mask"][s]:
            continue
        uf, mf, legal = pf.feats(s)
        acc = np.zeros(5, np.float32)
        for g in range(k):
            tok, ctx, h = views[g]
            mm = F.SYM_MOVE[g]
            mfg = mf[mm] if s < 10 else mf
            inp = pol.lin(np.concatenate([tok[s], ctx, uf, mfg.reshape(-1)]), "din") + w["aemb.weight"][prev[g]]
            h = pol.gru(inp, h)
            views[g][2] = h
            base = pol.lin(h, "whead") if s < 10 else np.concatenate([pol.lin(h, "bhead"), np.zeros(2, np.float32)])
            lg = base + pol.lin(mfg, "V") @ pol.lin(h, "U")
            if s < 10:
                lg = lg[mm]
            z = np.where(legal, lg, NEG)
            z = z - z.max()
            acc += z - np.log(np.exp(z).sum())
        lg = np.where(legal, acc / k, NEG).astype(np.float32)
        logits[s] = lg
        a = int(lg.argmax())
        if hook is not None:
            o = hook(s, lg, pf, a)
            if o is not None:
                a = int(o)
        labels[s] = a
        pf.advance(s, a)
        for g in range(k):
            prev[g] = (int(F.SYM_MOVE[g][a]) if s < 10 else a + 5)
    return labels, logits
