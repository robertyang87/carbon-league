"""Plan stages 0-3: the trainable PyTorch mirror of master3s2's network.

MainNet reproduces the source's _TorchNet math exactly (circular-pad 3x3
convs, per-block global-mean gate g, hw/hb 1x1-conv heads reshaped to
(n, 225, 5) / (n, 225, 3), and the critic: v1 1x1 conv -> mean+amax pool ->
concat the (n, 5) scalars -> v2 -> v3, output = sc[:, 0] + residual).
Weights load from / export to the source's exact npz key layout (see
layout.py), so a checkpoint produced here drops straight into
pack_submission.py and into the chassis' _Policy/_TorchNet.

Training-only additions (never exported -- export_source_dict writes only
the source keys, so stripping them is structural, not a runtime step):
  * aux_short_delta: Linear on the pooled trunk, predicts the net-carbon
    (cash-margin) delta at +15/+40/+80 steps;
  * aux_fog_carbon:  1x1 conv on the trunk map, predicts the true 15x15
    carbon map;
  * aux_win_logit:   Linear on the pooled trunk, win probability logit.
Optional capacity growth: n_extra residual blocks appended after the
inherited ones, their c2 projection zero-initialized so they are the
identity at init (the plan's 6 -> 8 block widening); export then writes
meta = total block count, which _Policy/_TorchNet read generically.

RollNet is the small rollout net for the chassis' _ROLL slot: same block
structure (the chassis builds rollouts with the same _TorchNet class, so
the key layout -- including the g gate -- is identical), 2-3 blocks,
32-48 channels, policy heads only, no critic.  Note: the default fix3
chassis has no _ROLL wiring (see pack_submission.py); RollNet matters for
the master3s2 chassis and for fast rollout evaluation inside training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cconv(x, conv):
    """3x3 conv with circular padding, exactly the source's cconv."""
    return conv(F.pad(x, (1, 1, 1, 1), mode='circular'))


class ResBlock(nn.Module):
    """t = relu(conv3(h)); t += g(mean(t)); t = conv3(t); out = relu(h + t)."""

    def __init__(self, ch, zero_init_out=False):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3)
        self.g = nn.Linear(ch, ch)
        self.c2 = nn.Conv2d(ch, ch, 3)
        if zero_init_out:
            nn.init.zeros_(self.c2.weight)
            nn.init.zeros_(self.c2.bias)

    def forward(self, h):
        t = F.relu(_cconv(h, self.c1))
        t = t + self.g(t.mean(dim=(2, 3)))[:, :, None, None]
        t = _cconv(t, self.c2)
        return F.relu(h + t)


class _Trunk(nn.Module):
    """stem + strategy embedding + residual blocks + hw/hb policy heads."""

    def __init__(self, in_ch, ch, nb, n_sid_rows):
        super().__init__()
        self.ch = ch
        self.stem = nn.Conv2d(in_ch, ch, 3)
        self.emb = nn.Parameter(torch.zeros(n_sid_rows, ch))
        self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(nb)])
        self.hw = nn.Conv2d(ch, 5, 1)
        self.hb = nn.Conv2d(ch, 3, 1)

    def trunk(self, x, sid):
        x = x[:, :self.stem.weight.shape[1]]
        e = self.emb[sid]
        if e.dim() == 1:
            e = e[None, :, None, None]
        else:
            e = e[:, :, None, None]
        h = F.relu(_cconv(x, self.stem) + e)
        for blk in self.blocks:
            h = blk(h)
        return h

    def heads(self, h):
        n, c = h.shape[0], h.shape[1]
        hf = h.reshape(n, c, -1)
        lw = torch.einsum('kc,ncp->npk', self.hw.weight[:, :, 0, 0], hf) + self.hw.bias
        lb = torch.einsum('kc,ncp->npk', self.hb.weight[:, :, 0, 0], hf) + self.hb.bias
        return lw, lb, hf


class MainNet(_Trunk):
    """The full master3s2 net: trunk + critic + training-only aux heads."""

    def __init__(self, in_ch=54, ch=96, nb=6, n_sid_rows=121, critic_k=32,
                 critic_h=128, n_extra=0, aux=True):
        super().__init__(in_ch, ch, nb + n_extra, n_sid_rows)
        for blk in list(self.blocks)[nb:]:
            nn.init.zeros_(blk.c2.weight)
            nn.init.zeros_(blk.c2.bias)
        self.nb_inherited = nb
        self.v1 = nn.Conv2d(ch, critic_k, 1)
        self.v2 = nn.Linear(2 * critic_k + 5, critic_h)
        self.v3 = nn.Linear(critic_h, 1)
        self.has_aux = aux
        if aux:
            self.aux_delta = nn.Linear(ch, 3)      # +15/+40/+80 margin deltas
            self.aux_fog = nn.Conv2d(ch, 1, 1)     # true carbon map (15x15)
            self.aux_win = nn.Linear(ch, 1)        # win logit

    def forward(self, x, sid, sc=None, want_aux=False):
        h = self.trunk(x, sid)
        lw, lb, hf = self.heads(h)
        v = None
        if sc is not None:
            z = F.relu(self.v1(h))
            z = torch.cat([z.mean(dim=(2, 3)), z.amax(dim=(2, 3)), sc], dim=1)
            v = sc[:, 0] + self.v3(F.relu(self.v2(z)))[:, 0]
        aux = None
        if want_aux and self.has_aux:
            pooled = hf.mean(dim=2)
            aux = {'delta': self.aux_delta(pooled),
                   'fog': self.aux_fog(h)[:, 0],
                   'win': self.aux_win(pooled)[:, 0]}
        return lw, lb, v, aux

    # -- source-layout (de)serialization ------------------------------------
    @classmethod
    def from_source(cls, policy, value, nb, n_extra=0, aux=True):
        """Build a MainNet with master3s2's weights (dicts of float32 arrays)."""
        ch = policy['stem.w'].shape[0]
        net = cls(in_ch=policy['stem.w'].shape[1], ch=ch, nb=nb,
                  n_sid_rows=policy['emb'].shape[0],
                  critic_k=value['v1.weight'].shape[0],
                  critic_h=value['v2.weight'].shape[0],
                  n_extra=n_extra, aux=aux)
        missing, unexpected = net.load_state_dict(
            source_to_state_dict(policy, value, nb), strict=False)
        # Only the zero-init extra blocks and the (random-init) aux heads
        # may be missing; anything else means a layout drift -- fail loudly.
        allowed = tuple('blocks.%d.' % i for i in range(nb, nb + n_extra))
        bad = [k for k in missing
               if not (k.startswith(allowed) or k.startswith('aux_'))]
        if bad or unexpected:
            raise ValueError('state_dict mismatch: missing=%s unexpected=%s'
                             % (bad, unexpected))
        return net

    def export_source_dict(self, snames):
        """{key: float32 ndarray} in the source layout (policy + value keys).

        Aux heads are not part of the layout and are therefore stripped by
        construction.  meta[0] = total block count (inherited + extra).
        """
        return state_dict_to_source(self.state_dict(), snames)


class RollNet(_Trunk):
    """Small policy-only rollout net (the _ROLL role): no critic, no aux."""

    def __init__(self, in_ch=54, ch=40, nb=3, n_sid_rows=121):
        super().__init__(in_ch, ch, nb, n_sid_rows)
        self.nb_inherited = nb

    def forward(self, x, sid):
        lw, lb, _ = self.heads(self.trunk(x, sid))
        return lw, lb

    def export_source_dict(self, snames):
        """Policy keys only -- a _ROLL payload must not carry value keys."""
        return state_dict_to_source(self.state_dict(), snames)


def source_to_state_dict(policy, value, nb):
    """Source-layout arrays -> MainNet/RollNet state_dict tensors."""
    sd = {'stem.weight': policy['stem.w'], 'stem.bias': policy['stem.b'],
          'emb': policy['emb']}
    for i in range(nb):
        p = 'b%d.' % i
        sd['blocks.%d.c1.weight' % i] = policy[p + 'c1.w']
        sd['blocks.%d.c1.bias' % i] = policy[p + 'c1.b']
        sd['blocks.%d.g.weight' % i] = policy[p + 'g.w']
        sd['blocks.%d.g.bias' % i] = policy[p + 'g.b']
        sd['blocks.%d.c2.weight' % i] = policy[p + 'c2.w']
        sd['blocks.%d.c2.bias' % i] = policy[p + 'c2.b']
    sd['hw.weight'] = policy['hw.w']
    sd['hw.bias'] = policy['hw.b']
    sd['hb.weight'] = policy['hb.w']
    sd['hb.bias'] = policy['hb.b']
    if value is not None:
        sd['v1.weight'] = value['v1.weight']
        sd['v1.bias'] = value['v1.bias']
        sd['v2.weight'] = value['v2.weight']
        sd['v2.bias'] = value['v2.bias']
        sd['v3.weight'] = value['v3.weight']
        sd['v3.bias'] = value['v3.bias']
    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            for k, v in sd.items()}


def state_dict_to_source(sd, snames):
    """MainNet/RollNet state_dict -> source-layout float32 arrays.

    Includes value keys only when the state dict has them (RollNet does
    not).  'snames' and 'meta' are included so the result is a complete
    checkpoint dict for pack_submission.py / npz storage.
    """
    nb = 0
    while ('blocks.%d.c1.weight' % nb) in sd:
        nb += 1

    def npy(k):
        return sd[k].detach().cpu().numpy().astype(np.float32)

    out = {'stem.w': npy('stem.weight'), 'stem.b': npy('stem.bias'),
           'emb': npy('emb')}
    for i in range(nb):
        p = 'b%d.' % i
        out[p + 'c1.w'] = npy('blocks.%d.c1.weight' % i)
        out[p + 'c1.b'] = npy('blocks.%d.c1.bias' % i)
        out[p + 'g.w'] = npy('blocks.%d.g.weight' % i)
        out[p + 'g.b'] = npy('blocks.%d.g.bias' % i)
        out[p + 'c2.w'] = npy('blocks.%d.c2.weight' % i)
        out[p + 'c2.b'] = npy('blocks.%d.c2.bias' % i)
    out['hw.w'] = npy('hw.weight')
    out['hw.b'] = npy('hw.bias')
    out['hb.w'] = npy('hb.weight')
    out['hb.b'] = npy('hb.bias')
    if 'v1.weight' in sd:
        out['v1.weight'] = npy('v1.weight')
        out['v1.bias'] = npy('v1.bias')
        out['v2.weight'] = npy('v2.weight')
        out['v2.bias'] = npy('v2.bias')
        out['v3.weight'] = npy('v3.weight')
        out['v3.bias'] = npy('v3.bias')
    out['snames'] = np.array(list(snames))
    out['meta'] = np.array([nb], dtype=np.int32)
    return out


def load_main_npz(path, n_extra=0, aux=True):
    """Load a source-layout checkpoint npz (policy+value keys, snames, meta)
    into a MainNet.  Returns (net, snames)."""
    z = np.load(path, allow_pickle=False)
    arrs = {k: z[k] for k in z.files}
    snames = [str(s) for s in arrs.pop('snames')]
    nb = int(arrs.pop('meta')[0])
    policy = {k: v.astype(np.float32) for k, v in arrs.items()
              if k in _policy_key_set(nb)}
    value = {k: arrs[k].astype(np.float32) for k in
             ('v1.weight', 'v1.bias', 'v2.weight', 'v2.bias', 'v3.weight', 'v3.bias')}
    net = MainNet.from_source(policy, value, nb, n_extra=n_extra, aux=aux)
    return net, snames


def _policy_key_set(nb):
    ks = {'stem.w', 'stem.b', 'emb', 'hw.w', 'hw.b', 'hb.w', 'hb.b'}
    for i in range(nb):
        ks |= {'b%d.%s' % (i, s) for s in
               ('c1.w', 'c1.b', 'g.w', 'g.b', 'c2.w', 'c2.b')}
    return ks
