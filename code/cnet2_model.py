"""cnet2_model -- torch definition of the cnet2 policy (mirrored by cnet2_np.py).

  trunk:   conv3x3 circular P->C, GN, ReLU; NB x [conv-GN-ReLU-conv-GN +skip, ReLU]
  aux:     1x1 conv -> true carbon (training signal only)
  tokens:  trunk[pos] ++ MLP(static kind/cargo/coords ++ 8-step history) -> D
  attn:    L post-norm transformer layers, NH heads, learned torus relative-position bias
  context: [mean trunk, masked-mean tokens, MLP(scalars), teacher embedding] -> D
  value:   tanh(Linear(context))
  decoder: GRUCell over own slots 0..11 (workers then bases); input =
           Linear(token, context, uf, mf.flatten) + emb(previous action);
           logits[m] = head(h)[m] + <U h, V mf[m]>, illegal masked.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn

import cnet2_feat as F

DEFAULT = dict(C=64, NB=4, D=128, NH=4, L=2, E=32, G=8)
TOKF = 9 + F.H * 10
NEG = -1e9


def tok_features(tok_kind, tok_cargo, tok_pos, hist_cell, hist_cargo, hist_move):
    """[B,T] ints/floats -> [B,T,TOKF] (torch); mirrored in cnet2_np."""
    B, T_ = tok_kind.shape
    kind = Fn.one_hot(tok_kind, 6).float()
    lc = torch.clamp(torch.log1p(torch.clamp(tok_cargo, min=0)) / F.LOGC, max=1.5)
    r = (tok_pos // F.S).float() / 14.0
    c = (tok_pos % F.S).float() / 14.0
    valid = (hist_cell >= 0)
    hc = torch.clamp(hist_cell, min=0)
    dr = ((hc // F.S) - (tok_pos // F.S)[..., None]) % F.S
    dc = ((hc % F.S) - (tok_pos % F.S)[..., None]) % F.S
    dr = torch.where(dr > 7, dr - F.S, dr).float() / 7.0
    dc = torch.where(dc > 7, dc - F.S, dc).float() / 7.0
    vf = valid.float()
    hl = torch.clamp(torch.log1p(torch.clamp(hist_cargo, min=0)) / F.LOGC, max=1.5)
    mv = Fn.one_hot(hist_move.long(), 6).float()
    h = torch.cat([vf[..., None], (dr * vf)[..., None], (dc * vf)[..., None], (hl * vf)[..., None],
                   mv * vf[..., None]], -1)
    return torch.cat([kind, lc[..., None], r[..., None], c[..., None], h.reshape(B, T_, -1)], -1)


_r = (torch.arange(17) - 1) % F.S
WRAP_IDX = (_r[:, None] * F.S + _r[None, :]).reshape(-1)


def wrap(h):
    """circular pad by 1 via a single gather (much faster than padding_mode='circular' on MPS; on the GB10 the
    native circular pad is slower too: its slice copies cost ~1.7x per training step, measured 2026-11-09)."""
    B, C = h.shape[:2]
    idx = WRAP_IDX.to(h.device)
    return h.reshape(B, C, F.N).index_select(2, idx).reshape(B, C, 17, 17)


class Block(nn.Module):
    def __init__(self, C, G):
        super().__init__()
        self.c1 = nn.Conv2d(C, C, 3)
        self.n1 = nn.GroupNorm(G, C)
        self.c2 = nn.Conv2d(C, C, 3)
        self.n2 = nn.GroupNorm(G, C)

    def forward(self, x):
        h = Fn.relu(self.n1(self.c1(wrap(x))))
        return Fn.relu(x + self.n2(self.c2(wrap(h))))


class Attn(nn.Module):
    def __init__(self, D, NH):
        super().__init__()
        self.NH = NH
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)
        self.rel = nn.Embedding(F.N, NH)
        self.n1 = nn.LayerNorm(D)
        self.f1 = nn.Linear(D, 2 * D)
        self.f2 = nn.Linear(2 * D, D)
        self.n2 = nn.LayerNorm(D)

    def forward(self, x, rel, mask):
        B, T_, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T_, 3, self.NH, D // self.NH).permute(2, 0, 3, 1, 4)
        sc = q @ k.transpose(-1, -2) / (D // self.NH) ** 0.5 + self.rel(rel).permute(0, 3, 1, 2)
        sc = sc.masked_fill(~mask[:, None, None, :], NEG)
        a = torch.softmax(sc, -1) @ v
        a = a.transpose(1, 2).reshape(B, T_, D)
        x = self.n1(x + self.out(a))
        x = self.n2(x + self.f2(Fn.relu(self.f1(x))))
        return x * mask[..., None].float()


class CNet2(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        c = dict(DEFAULT); c.update(cfg); self.cfg = c
        C, D, E, G = c["C"], c["D"], c["E"], c["G"]
        self.stem = nn.Conv2d(F.P, C, 3)
        self.stem_n = nn.GroupNorm(G, C)
        self.blocks = nn.ModuleList([Block(C, G) for _ in range(c["NB"])])
        if c.get("ZI"):                      # zero-init residual branch (blocks start as identity) for deep/wide stability
            for blk in self.blocks:
                nn.init.zeros_(blk.n2.weight)
        self.aux = nn.Conv2d(C, 1, 1)
        self.tok1 = nn.Linear(TOKF, D)
        self.tok2 = nn.Linear(D, D)
        self.fuse = nn.Linear(C + D, D)
        self.attn = nn.ModuleList([Attn(D, c["NH"]) for _ in range(c["L"])])
        self.sc1 = nn.Linear(F.NS, 64)
        self.sc2 = nn.Linear(64, 64)
        self.teach = nn.Embedding(F.N_TEACH, 16)
        self.ctx = nn.Linear(C + D + 64 + 16, D)
        self.value = nn.Linear(D, 1)
        self.h0 = nn.Linear(D, D)
        self.din = nn.Linear(2 * D + F.UF + 5 * F.MF, D)
        self.aemb = nn.Embedding(9, D)
        self.gru = nn.GRUCell(D, D)
        self.whead = nn.Linear(D, 5)
        self.bhead = nn.Linear(D, 3)
        self.U = nn.Linear(D, E)
        self.V = nn.Linear(F.MF, E)

    def encode(self, b):
        x = b["maps"]                                   # [B,P,15,15] float
        B = x.shape[0]
        h = Fn.relu(self.stem_n(self.stem(wrap(x))))
        for blk in self.blocks:
            h = blk(h)
        aux = self.aux(h).reshape(B, -1)
        flat = h.reshape(B, h.shape[1], -1)             # [B,C,N]
        pos = b["tok_pos"]
        g = torch.gather(flat, 2, pos[:, None, :].expand(-1, flat.shape[1], -1)).transpose(1, 2)
        tf = tok_features(b["tok_kind"], b["tok_cargo"], pos, b["hist_cell"], b["hist_cargo"], b["hist_move"])
        t = self.tok2(Fn.relu(self.tok1(tf)))
        tok = self.fuse(torch.cat([g, t], -1))
        mask = b["tok_mask"]
        r = pos // F.S; cc = pos % F.S
        rel = ((r[:, :, None] - r[:, None, :]) % F.S) * F.S + (cc[:, :, None] - cc[:, None, :]) % F.S
        tok = tok * mask[..., None].float()
        for a in self.attn:
            tok = a(tok, rel, mask)
        mf_ = mask.float()
        tmean = tok.sum(1) / mf_.sum(1, keepdim=True).clamp(min=1)
        s = self.sc2(Fn.relu(self.sc1(b["scal"])))
        pre = torch.cat([flat.mean(2), tmean, s], -1)
        if b.get("_want_pre"):
            return tok, pre, aux
        ctx = self.ctx_from(pre, b["teacher"])
        return tok, ctx, aux

    def ctx_from(self, pre, teacher):
        """style enters only here: trunk summary [B, C+D+64] + teach[teacher] -> ctx."""
        return Fn.relu(self.ctx(torch.cat([pre, self.teach(teacher)], -1)))

    def decode(self, b, tok, ctx):
        """teacher-forced decoder from given trunk outputs (tok [B,>=OWN,D], ctx [B,D]) -> logits [B,OWN,5]."""
        B = tok.shape[0]
        h = self.h0(ctx)
        y = b["y"].clamp(min=0)
        prev = torch.full((B,), 8, dtype=torch.long, device=tok.device)
        outs = []
        mask = b["tok_mask"]
        for s in range(F.OWN):
            mf = b["mf"][:, s]
            inp = self.din(torch.cat([tok[:, s], ctx, b["uf"][:, s], mf.reshape(B, -1)], -1)) + self.aemb(prev)
            hn = self.gru(inp, h)
            m = mask[:, s][:, None]
            h = torch.where(m, hn, h)
            base = self.whead(h) if s < 10 else Fn.pad(self.bhead(h), (0, 2))
            lg = base + (self.V(mf) * self.U(h)[:, None, :]).sum(-1)
            lg = lg.masked_fill(~b["legal"][:, s], NEG)
            outs.append(lg)
            a = y[:, s] + (0 if s < 10 else 5)
            prev = torch.where(mask[:, s], a, prev)
        return torch.stack(outs, 1)

    def forward_seq(self, b):
        """Same as forward (teacher-forced) but the 12-slot GRU recurrence runs as ONE packed nn.GRU call:
        every slot input depends only on data (the previous action is the teacher label), so no Python loop.
        Masked slots keep the previous hidden state, exactly like the loop: valid slots are packed to the
        front of each row, run as a variable-length sequence, then scattered back."""
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from torch.func import functional_call
        tok, ctx, aux = self.encode(b)
        B, S = tok.shape[0], F.OWN
        dev = tok.device
        mask = b["tok_mask"][:, :S]
        y = b["y"].clamp(min=0)
        off = torch.tensor([0] * 10 + [5] * (S - 10), device=dev)
        act = y[:, :S] + off                                              # action fed to the next valid slot
        pos = torch.arange(S, device=dev).expand(B, S)
        lv = torch.where(mask, pos, torch.full_like(pos, -1)).cummax(1).values      # last valid slot <= s
        lvb = torch.cat([torch.full((B, 1), -1, device=dev, dtype=lv.dtype), lv[:, :-1]], 1)  # last valid < s
        prev = torch.where(lvb >= 0, act.gather(1, lvb.clamp(min=0)), torch.full_like(lvb, 8))
        mf = b["mf"][:, :S]
        inp = self.din(torch.cat([tok[:, :S], ctx[:, None].expand(-1, S, -1), b["uf"][:, :S],
                                  mf.reshape(B, S, -1)], -1)) + self.aemb(prev)
        h0 = self.h0(ctx)
        n = mask.sum(1)
        order = torch.argsort((~mask).to(torch.int64) * S + pos, dim=1)              # valid slots first, in order
        inp_c = inp.gather(1, order[:, :, None].expand(-1, -1, inp.shape[-1]))
        g = self.__dict__.get("_seq_gru")
        if g is None or g.weight_ih_l0.device != dev:
            g = nn.GRU(inp.shape[-1], h0.shape[-1], batch_first=True).to(dev)
            self.__dict__["_seq_gru"] = g                                        # not a registered submodule
        params = {"weight_ih_l0": self.gru.weight_ih, "weight_hh_l0": self.gru.weight_hh,
                  "bias_ih_l0": self.gru.bias_ih, "bias_hh_l0": self.gru.bias_hh}
        pk = pack_padded_sequence(inp_c, n.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        out, _ = functional_call(g, params, (pk, h0[None].contiguous()))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=S)
        cnt = mask.to(torch.int64).cumsum(1)                                       # valid slots <= s
        hs = out.gather(1, (cnt - 1).clamp(min=0)[:, :, None].expand(-1, -1, out.shape[-1]))
        hs = torch.where((cnt > 0)[:, :, None], hs, h0[:, None].expand(-1, S, -1))
        base = torch.cat([self.whead(hs[:, :10]), Fn.pad(self.bhead(hs[:, 10:]), (0, 2))], 1)
        lg = base + (self.V(mf) * self.U(hs)[:, :, None, :]).sum(-1)
        lg = lg.masked_fill(~b["legal"][:, :S], NEG)
        return lg, torch.tanh(self.value(ctx)).squeeze(1), aux

    def forward(self, b):
        """teacher-forced. returns logits [B,OWN,5], value [B], aux [B,N]."""
        tok, ctx, aux = self.encode(b)
        B = tok.shape[0]
        h = self.h0(ctx)
        y = b["y"].clamp(min=0)
        prev = torch.full((B,), 8, dtype=torch.long, device=tok.device)
        outs = []
        mask = b["tok_mask"]
        for s in range(F.OWN):
            mf = b["mf"][:, s]                          # [B,5,MF]
            inp = self.din(torch.cat([tok[:, s], ctx, b["uf"][:, s], mf.reshape(B, -1)], -1)) + self.aemb(prev)
            hn = self.gru(inp, h)
            m = mask[:, s][:, None]
            h = torch.where(m, hn, h)
            if s < 10:
                base = self.whead(h)
            else:
                base = Fn.pad(self.bhead(h), (0, 2))
            lg = base + (self.V(mf) * self.U(h)[:, None, :]).sum(-1)
            lg = lg.masked_fill(~b["legal"][:, s], NEG)
            outs.append(lg)
            a = y[:, s] + (0 if s < 10 else 5)
            prev = torch.where(mask[:, s], a, prev)
        return torch.stack(outs, 1), torch.tanh(self.value(ctx)).squeeze(1), aux
