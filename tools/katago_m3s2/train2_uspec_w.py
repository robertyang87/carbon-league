"""train2_uspec_w: Gen-5 combined trainer = U' utility scale + per-row weights.

COMPOSITION (two already-gated variants, disjoint deltas, composed
mechanically 2026-11-20 for the Gen-5 launch):
  * Value-spectrum U' arm: train2_uspec.py md5
    aa1f8596c8101a146b6a230d9bfd3f54 -- train2.py with ONE change:
    score_utility's tanh scale parameterized (module global
    _UTILITY_SCALE, CLI --utility-scale, default 600.0). Its own
    equivalence gate passed 2026-11-20 (design S4; scale 600 bit-identical
    to train2, scale 300 = U').
  * Lossprobe 2 row-weight channel: train2_w.py md5
    cab1d22a20c34bfbe6d4e3b361a021b8 -- train2.main() copied with exactly
    two deltas: (1) per-row wpw/wvw extraction at the plan's indices
    (shards without the keys default to 1.0), (2) policy terms use
    pw_eff = pw * wpw, value term is the wvw-weighted mean. Its
    equivalence gate + independent deep review passed 2026-11-20
    (all-ones bit-identical to train2; x0.5 probe exact).

THIS FILE is train2_w.py verbatim except:
  (a) the base-module import is train2_uspec instead of train2 (all
      reused machinery -- load_dataset, OrderedFeeder, build_epoch_plan,
      _to_device, materialize/score_utility -- comes from train2_uspec,
      so the feeder's vt uses the parameterized score_utility);
  (b) CLI gains --utility-scale (default 600.0), and main() writes it
      into train2_uspec._UTILITY_SCALE before the OrderedFeeder forks
      its workers (they inherit the base module's globals at fork),
      mirroring train2_uspec.main().

IDENTITY CLAIMS (each verified by the combined gate before any use):
  * scale 600 + no weight keys        == train2, bit-identical;
  * scale 600 + wpw/wvw all ones      == train2, bit-identical;
  * scale 300 + wpw/wvw all ones      == train2_uspec --utility-scale 300,
                                         bit-identical;
  * scale 600 + suppressed wpw 0.5    == train2_w's recorded C-run epoch
                                         stats on the same fixture.

Weight semantics (Strategy Group registration 029c3d2b):
  SUPPRESSED games: wpw 0.5, wvw 1.0. Everything else: wpw 1.0, wvw 1.0.
  Correction rows (searched=1) keep wpw 1.0; their effective policy
  weight is already 1 via pw. See attach_row_weights.py (conversion
  side) for how the keys are attached.

This file never edits train2.py, train2_uspec.py or train2_w.py.
CLI is train2's CLI plus --utility-scale.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout
import train2_uspec as train2

SIZE = train2.SIZE
N = train2.N


def main():
    ap = argparse.ArgumentParser(description='train2 + U-prime utility scale + per-row weights (Gen-5 combined).')
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
    ap.add_argument('--feed-workers', type=int, default=8,
                    help='materialization worker processes (<=1 = inline)')
    ap.add_argument('--prefetch', type=int, default=4,
                    help='steps materialized ahead of consumption (in-order)')
    ap.add_argument('--feed-timeout', type=float,
                    default=train2.DEFAULT_FEED_TIMEOUT,
                    help='max seconds to wait for one materialized step '
                         'before dying loudly, naming the step (worker '
                         'hard-death guard, review 2.7; default 600)')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile the training forward (default OFF; '
                         'keep OFF for equivalence-gate runs)')
    ap.add_argument('--utility-scale', type=float, default=600.0,
                    help='tanh scale of score_utility in the base module '
                         '(600 = train2 verbatim U; 300 = value-spectrum '
                         'U-prime arm)')
    args = ap.parse_args()
    if not (args.feed_timeout and args.feed_timeout > 0):
        ap.error('--feed-timeout must be > 0 seconds, got %r'
                 % (args.feed_timeout,))

    # Utility scale must be set on the base module before the
    # OrderedFeeder forks its workers (they inherit the base module's
    # globals at fork time). Mirrors train2_uspec.main().
    train2._UTILITY_SCALE = float(args.utility_scale)
    print('utility scale: %g' % train2._UTILITY_SCALE, flush=True)

    import torch
    import torch.nn.functional as F
    import model as model_mod

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rows = train2.load_dataset([d for d in args.data.split(',') if d])
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

    train_net = torch.compile(net) if args.compile else net

    feeder = train2.OrderedFeeder(rows.arrays, args.fast_policy_weight,
                                  args.feed_workers, args.prefetch,
                                  args.feed_timeout)
    step = 0
    t_train = time.time()
    try:
        for epoch in range(args.epochs):
            # RNG plan for the whole epoch, generated in the main process
            # in train.py's exact consumption order (see train2 docstring).
            plan = train2.build_epoch_plan(rng, len(rows), args.batch)
            tot = {}
            nbatches = 0
            # DELTA (1): zip the plan with the feeder's in-order batches
            # so each batch's row indices are known for weight lookup.
            # The feeder yields strictly in plan order, so the pairing is
            # exact. wpw/wvw are per-row scalars (augment-invariant);
            # shards without the keys default to 1.0.
            for (idx, _gs), np_batch in zip(plan, feeder.run_epoch(plan)):
                b = train2._to_device(np_batch, torch, args.device)
                w_np = train2._to_device(
                    {'wpw': (rows.arrays['wpw'][idx].astype(np.float32)
                             if 'wpw' in rows.arrays
                             else np.ones(len(idx), dtype=np.float32)),
                     'wvw': (rows.arrays['wvw'][idx].astype(np.float32)
                             if 'wvw' in rows.arrays
                             else np.ones(len(idx), dtype=np.float32))},
                    torch, args.device)
                wpw, wvw = w_np['wpw'], w_np['wvw']
                x, sc = b['x'], b['sc']
                tw, tb, wm, bm = b['tw'], b['tb'], b['wm'], b['bm']
                vt, dt_, fog, win, pw = (b['vt'], b['dt'], b['fog'],
                                         b['win'], b['pw'])
                for gp in opt.param_groups:
                    gp['lr'] = lr_at(step)
                with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                    lw, lb, v, aux = train_net(x, sid, sc, want_aux=True)
                    logw = F.log_softmax(lw.float(), dim=-1)
                    logb = F.log_softmax(lb.float(), dim=-1)
                    ce_w = -(tw * logw).sum(-1)                     # (n, 225)
                    ce_b = -(tb * logb).sum(-1)
                    # DELTA (2): policy terms use pw_eff = pw * wpw;
                    # value term is the wvw-weighted mean of per-row
                    # squared errors. Aux terms untouched. With wpw == 1
                    # and wvw == 1 these lines are train2's lines.
                    pw_eff = pw * wpw
                    lp_w = ((ce_w * wm).sum(1) / wm.sum(1).clamp(min=1) * pw_eff).sum() / pw_eff.sum().clamp(min=1)
                    lp_b = ((ce_b * bm).sum(1) / bm.sum(1).clamp(min=1) * pw_eff).sum() / pw_eff.sum().clamp(min=1)
                    l_v = (F.mse_loss(v.float(), vt, reduction='none') * wvw).sum() / wvw.sum().clamp(min=1)
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
                if step % 200 == 0:
                    el = time.time() - t_train
                    print('epoch %d step %d/%d: %.0fs elapsed, %.2f steps/s'
                          % (epoch, step, total_steps, el, step / max(el, 1e-9)),
                          flush=True)
            print('epoch %d: %s' % (epoch, {k: round(v / nbatches, 4) for k, v in tot.items()}))
    finally:
        feeder.close()

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
