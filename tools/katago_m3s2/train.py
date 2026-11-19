"""Plan stage 2: joint training on self-play replay shards.

Dataset: the .npz shards written by vec_selfplay.py, taken from a sliding
window over the last 3 generations (the caller passes those generations'
replay dirs; run_loop.sh does exactly that).

Augmentation: V4 symmetry (the Klein four group {identity, rot180, flipH,
flipV}) with the exact tables the fix2 chassis builds for its TTA
(_v4_coords / V4_FWD_CELL / V4_INV_CELL / V4_FWD_ACT / V4_PLANE_PERM):
feature planes move by the inverse cell map, direction-danger planes
25-34 additionally permute by the inverse action map; worker targets move
by the forward cell/action pushforward (the same transform in pullback
form), base targets and the carbon map by the cell map alone; scalars,
margin and cash-delta targets are symmetry invariants and stay put.

Losses (brief-prescribed weights):
  L = L_policy_w + L_policy_b + 1.5 * L_value + 0.3 * L_aux_delta
      + 0.2 * L_aux_fog + 0.2 * L_aux_win
Policy losses are masked cross-entropies against the marginalized targets
(worker cells only / base cells only).  Rows whose targets came from the
fast plan carry the net's own probabilities, so their policy weight is 0
by default (--fast-policy-weight to change); value/aux train on all rows.
Value target: the Score Utility U(d) = 0.6*tanh(d/600) + 0.4*clip(d/2500,
-1, 1) of the final margin.  The scale tension with the chassis' search
(whose arithmetic runs on _terminal(m) = clip(m/3000, -2.5, 2.5) +
0.5*sign(m)) is RESOLVED by Strategy ruling
reports/RULING_KATAGO_VALUE_SCALE_BRIDGE_20261119.md: the value head
keeps training toward U (this file is unchanged by the ruling), and the
exact bridge B(u) = _terminal(U^{-1}(u)) -- bridge.py in this directory
-- is applied at the single interface where the value head's output
enters search leaf evaluation: pack_submission.py injects it into the
packed chassis' look() for trained checkpoints (--value-scale u), and
vec_selfplay.py applies it to deep-plan leaf values under the same
flag.  Raw final margins stay in the replay shards, so both scales
remain exactly recoverable from the same records.

aux targets: delta = (margin change at +15/+40/+80) / 1000; fog = true
carbon / 100; win = the shard's win label (BCE on the logit).

Optimizer: AdamW, lr 1.5e-4 cosine-decayed to 3e-5 over the run, batch
512, BF16 autocast.  Checkpoints are saved in the source npz key layout
(policy + value keys, float16) so pack_submission.py consumes them
directly; aux heads never enter the export (see model.py).
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout

SIZE = 15
N = 225
NPL = 54
DR = (0, -1, 0, 1, 0)
DC = (0, 0, 1, 0, -1)


def _v4_coords(r, c, idx):
    if idx == 0:
        return r, c
    if idx == 1:
        return SIZE - 1 - r, SIZE - 1 - c
    if idx == 2:
        return r, SIZE - 1 - c
    return SIZE - 1 - r, c


def build_v4_tables():
    """Identical construction to the fix2 chassis' V4 tables."""
    fwd_cell = np.zeros((4, N), dtype=np.intp)
    inv_cell = np.zeros((4, N), dtype=np.intp)
    fwd_act = np.zeros((4, 5), dtype=np.intp)
    inv_act = np.zeros((4, 5), dtype=np.intp)
    plane_perm = np.zeros((4, NPL), dtype=np.intp)
    for g in range(4):
        for i in range(N):
            r, c = divmod(i, SIZE)
            r2, c2 = _v4_coords(r, c, g)
            j = r2 * SIZE + c2
            fwd_cell[g, i] = j
            inv_cell[g, j] = i
        r0, c0 = _v4_coords(7, 7, g)
        ia = [0] * 5
        for a in range(5):
            r1, c1 = _v4_coords(7 + DR[a], 7 + DC[a], g)
            dr, dc = r1 - r0, c1 - c0
            for a2 in range(5):
                if DR[a2] == dr and DC[a2] == dc:
                    fwd_act[g, a] = a2
                    ia[a2] = a
                    break
        inv_act[g] = ia
        perm = np.arange(NPL, dtype=np.intp)
        for an in range(5):
            perm[25 + an] = 25 + ia[an]
            perm[30 + an] = 30 + ia[an]
        plane_perm[g] = perm
    return fwd_cell, inv_cell, fwd_act, inv_act, plane_perm


V4 = build_v4_tables()


def augment(row, g):
    """Apply V4 element g to one replay row (dict of numpy arrays)."""
    if g == 0:
        return row
    _, inv_cell, _, inv_act, plane_perm = V4
    out = dict(row)
    x = row['x'].reshape(NPL, N)
    out['x'] = x[:, inv_cell[g]][plane_perm[g]].reshape(NPL, SIZE, SIZE)
    out['tw'] = row['tw'][inv_cell[g]][:, inv_act[g]]
    out['tb'] = row['tb'][inv_cell[g]]
    out['wmask'] = row['wmask'][inv_cell[g]]
    out['bmask'] = row['bmask'][inv_cell[g]]
    out['carbon'] = row['carbon'][inv_cell[g]]
    return out


def score_utility(d):
    """U(d) = 0.6*tanh(d/600) + 0.4*clip(d/2500, -1, 1), elementwise."""
    return 0.6 * np.tanh(d / 600.0) + 0.4 * np.clip(d / 2500.0, -1.0, 1.0)


def load_rows(dirs):
    rows = []
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, '*.npz')))
    if not files:
        raise SystemExit('no replay shards found in %s' % (dirs,))
    for path in files:
        z = np.load(path)
        n = z['x'].shape[0]
        for i in range(n):
            rows.append({k: z[k][i] for k in z.files})
    print('loaded %d rows from %d shards' % (len(rows), len(files)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', required=True,
                    help='comma-separated replay dirs (the last-3-generations window)')
    ap.add_argument('--init', required=True, help='init checkpoint npz (source layout)')
    ap.add_argument('--out', required=True, help='output checkpoint npz (source layout)')
    ap.add_argument('--out-pt', default=None, help='optional torch state_dict sidecar')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--lr-final', type=float, default=3e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--fast-policy-weight', type=float, default=0.0)
    ap.add_argument('--n-extra', type=int, default=0,
                    help='append this many zero-init blocks to a 6-block init')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import model as model_mod

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rows = load_rows([d for d in args.data.split(',') if d])
    net, snames = model_mod.load_main_npz(args.init, n_extra=args.n_extra)
    net.to(args.device)
    sid = snames.index(layout.SID_NAME) if layout.SID_NAME in snames else len(snames)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(rows) / args.batch)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        t = min(1.0, step / max(1, total_steps))
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * t))

    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(rows))
        tot = {}
        nbatches = 0
        for s0 in range(0, len(order), args.batch):
            idx = order[s0:s0 + args.batch]
            batch = [augment(rows[i], int(rng.randint(4))) for i in idx]
            x = torch.from_numpy(np.stack([b['x'] for b in batch]).astype(np.float32) / 100.0)
            sc = torch.from_numpy(np.stack([b['sc'] for b in batch]).astype(np.float32))
            tw = torch.from_numpy(np.stack([b['tw'] for b in batch]).astype(np.float32))
            tb = torch.from_numpy(np.stack([b['tb'] for b in batch]).astype(np.float32))
            wm = torch.from_numpy(np.stack([b['wmask'] for b in batch]).astype(np.float32))
            bm = torch.from_numpy(np.stack([b['bmask'] for b in batch]).astype(np.float32))
            margin = np.stack([b['margin'] for b in batch]).astype(np.float32)
            vt = torch.from_numpy(score_utility(margin).astype(np.float32))
            dt_ = torch.from_numpy(np.stack([b['aux_delta'] for b in batch]).astype(np.float32) / 1000.0)
            fog = torch.from_numpy(np.stack([b['carbon'] for b in batch]).astype(np.float32) / 100.0)
            win = torch.from_numpy(np.stack([b['win'] for b in batch]).astype(np.float32))
            searched = np.stack([b['searched'] for b in batch]).astype(np.float32)
            pw = torch.from_numpy(searched + (1 - searched) * args.fast_policy_weight)
            x, sc = x.to(args.device), sc.to(args.device)
            tw, tb, wm, bm = tw.to(args.device), tb.to(args.device), wm.to(args.device), bm.to(args.device)
            vt, dt_, fog, win, pw = (vt.to(args.device), dt_.to(args.device),
                                     fog.to(args.device), win.to(args.device), pw.to(args.device))
            for gp in opt.param_groups:
                gp['lr'] = lr_at(step)
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                lw, lb, v, aux = net(x, sid, sc, want_aux=True)
                logw = F.log_softmax(lw.float(), dim=-1)
                logb = F.log_softmax(lb.float(), dim=-1)
                ce_w = -(tw * logw).sum(-1)                     # (n, 225)
                ce_b = -(tb * logb).sum(-1)
                lp_w = ((ce_w * wm).sum(1) / wm.sum(1).clamp(min=1) * pw).sum() / pw.sum().clamp(min=1)
                lp_b = ((ce_b * bm).sum(1) / bm.sum(1).clamp(min=1) * pw).sum() / pw.sum().clamp(min=1)
                l_v = F.mse_loss(v.float(), vt)
                l_d = F.mse_loss(aux['delta'].float(), dt_)
                l_f = F.mse_loss(aux['fog'].float(), fog.reshape(-1, SIZE, SIZE))
                l_win = F.binary_cross_entropy_with_logits(aux['win'].float(), win)
                loss = lp_w + lp_b + 1.5 * l_v + 0.3 * l_d + 0.2 * l_f + 0.2 * l_win
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            nbatches += 1
            for k, val in (('loss', loss), ('pw', lp_w), ('pb', lp_b), ('v', l_v),
                           ('delta', l_d), ('fog', l_f), ('win', l_win)):
                tot[k] = tot.get(k, 0.0) + float(val)
        print('epoch %d: %s' % (epoch, {k: round(v / nbatches, 4) for k, v in tot.items()}))

    out = net.export_source_dict(snames)
    with open(args.out, 'wb') as f:
        np.savez_compressed(f, **layout.to_f16(out))
    print('wrote %s' % args.out)
    if args.out_pt:
        torch.save(net.state_dict(), args.out_pt)
        print('wrote %s' % args.out_pt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
