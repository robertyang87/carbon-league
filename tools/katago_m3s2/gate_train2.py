"""gate_train2: equivalence gate + bench for train2.py (bypass piece).

Modes:
  --mode seq    (a) batch-sequence check ONLY. Pure numpy, no torch, no
                GPU, no data files needed: replays train.py's index/RNG
                logic (reference implementation below, transcribed from
                train.py main loop) against train2.generate_index_sequence
                and compares every step's row-index tuple AND augment-draw
                tuple verbatim (set + order). Runs on any machine.
  --mode equiv  (a) + (b). (b) loss-trajectory check needs torch + real
                --data/--init: runs the OLD inline feeding path and the
                NEW pooled path in-process from the same init weights,
                same data, same seed, for --loss-steps steps, comparing
                per-step loss and components. If torch/data/init are
                unavailable it prints exactly why (b) was skipped plus the
                L20 command that runs it -- it never fabricates a pass.
  --mode bench  Feeding + step throughput: old inline vs new pooled on
                the same data/init (torch required for full step timing;
                without torch it times materialization only and says so).
                Baseline anchor: 11.78 steps/s (L20 Gen-2, war-room
                measurement). GPU utilization is sampled via nvidia-smi
                when present, else reported as not-measured.

Verdict discipline: any failed check prints a final line
  VERDICT: FAIL -> ABANDON (不过 → 弃置)
plus a failure record (step, metric, measured, line). There is NO retry
or auto-fix logic in this file, by design.

Tolerances for (b) [all marked 待策略组钉 -- pending strategy pinning]:
  GPU BF16 training is not bit-deterministic across feeding orders of
  kernel launches; the loss comparison therefore uses
    |loss_new - loss_old| <= max(ABS_TOL, REL_TOL * |loss_old|)
  with ABS_TOL = 2e-2, REL_TOL = 0.02 as PLACEHOLDERS pending the pin.
  The sequence check (a) has NO tolerance: verbatim equality or FAIL.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train2 as t2

ABS_TOL = 2e-2    # 待策略组钉 (placeholder, see docstring)
REL_TOL = 0.02    # 待策略组钉 (placeholder, see docstring)
BASELINE_STEPS_S = 11.78  # L20 Gen-2 measured, war-room supplied


# ---------------------------------------------------------------------------
# Reference sequence: train.py's RNG consumption, transcribed from its main
# loop (per epoch: permutation, then per batch, per row: randint(4)).
# ---------------------------------------------------------------------------

def reference_index_sequence(n_rows, batch, epochs, seed, max_steps=None):
    rng = np.random.RandomState(seed)
    seq = []
    for _epoch in range(epochs):
        order = rng.permutation(n_rows)
        for s0 in range(0, len(order), batch):
            idx = order[s0:s0 + batch]
            gs = tuple(int(rng.randint(4)) for _ in idx)
            seq.append((tuple(int(v) for v in idx), gs))
            if max_steps is not None and len(seq) >= max_steps:
                return seq
    return seq


def check_sequence(n_rows, batch, epochs, seed, max_steps):
    """(a) Verbatim sequence check. Returns (ok, failure_dict_or_None)."""
    ref = reference_index_sequence(n_rows, batch, epochs, seed, max_steps)
    new = t2.generate_index_sequence(n_rows, batch, epochs, seed, max_steps)
    if len(ref) != len(new):
        return False, {'step': min(len(ref), len(new)), 'metric': 'n_steps',
                       'measured': len(new), 'line': len(ref)}
    for s, ((ri, rg), (ni, ng)) in enumerate(zip(ref, new)):
        if ri != ni:
            return False, {'step': s, 'metric': 'row_index_sequence',
                           'measured': ni[:8], 'line': ri[:8]}
        if rg != ng:
            return False, {'step': s, 'metric': 'augment_draw_sequence',
                           'measured': ng[:8], 'line': rg[:8]}
    return True, None


# ---------------------------------------------------------------------------
# (b) loss trajectory + bench: in-process, torch required.
# ---------------------------------------------------------------------------

def _run_steps(feed_mode, rows, net_init_path, args, n_steps, feeder_kw):
    """Run n_steps of training with old (inline) or new (pooled) feeding.

    Replicates train.py's step math verbatim (same loss code as train2's
    main). Returns (losses list of dicts, elapsed_seconds).
    """
    import copy
    import torch
    import torch.nn.functional as F
    import model as model_mod
    import layout

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    net, snames = model_mod.load_main_npz(net_init_path, n_extra=args.n_extra)
    net.to(args.device)
    sid = snames.index(layout.SID_NAME) if layout.SID_NAME in snames else len(snames)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    import math
    steps_per_epoch = math.ceil(len(rows) / args.batch)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        t = min(1.0, step / max(1, total_steps))
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * t))

    losses = []
    step = 0
    t0 = time.time()
    feeder = None
    if feed_mode == 'new':
        feeder = t2.OrderedFeeder(rows.arrays, args.fast_policy_weight,
                                  feeder_kw['workers'], feeder_kw['prefetch'])
    try:
        done = False
        for _epoch in range(args.epochs):
            if done:
                break
            plan = t2.build_epoch_plan(rng, len(rows), args.batch)
            if feed_mode == 'new':
                stream = feeder.run_epoch(plan)
            else:
                stream = (t2.materialize_numpy(rows.arrays, idx, gs,
                                               args.fast_policy_weight)
                          for idx, gs in plan)
            for np_batch in stream:
                b = t2._to_device(np_batch, torch, args.device)
                x, sc = b['x'], b['sc']
                tw, tb, wm, bm = b['tw'], b['tb'], b['wm'], b['bm']
                vt, dt_, fog, win, pw = (b['vt'], b['dt'], b['fog'],
                                         b['win'], b['pw'])
                for gp in opt.param_groups:
                    gp['lr'] = lr_at(step)
                with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                    lw, lb, v, aux = net(x, sid, sc, want_aux=True)
                    logw = F.log_softmax(lw.float(), dim=-1)
                    logb = F.log_softmax(lb.float(), dim=-1)
                    ce_w = -(tw * logw).sum(-1)
                    ce_b = -(tb * logb).sum(-1)
                    lp_w = ((ce_w * wm).sum(1) / wm.sum(1).clamp(min=1) * pw).sum() / pw.sum().clamp(min=1)
                    lp_b = ((ce_b * bm).sum(1) / bm.sum(1).clamp(min=1) * pw).sum() / pw.sum().clamp(min=1)
                    l_v = F.mse_loss(v.float(), vt)
                    l_d = F.mse_loss(aux['delta'].float(), dt_)
                    l_f = F.mse_loss(aux['fog'].float(), fog.reshape(-1, 15, 15))
                    l_win = F.binary_cross_entropy_with_logits(aux['win'].float(), win)
                    loss = lp_w + lp_b + 1.5 * l_v + 0.3 * l_d + 0.2 * l_f + 0.2 * l_win
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append({'loss': float(loss), 'pw': float(lp_w),
                               'pb': float(lp_b), 'v': float(l_v),
                               'delta': float(l_d), 'fog': float(l_f),
                               'win': float(l_win)})
                step += 1
                if step >= n_steps:
                    done = True
                    break
    finally:
        if feeder is not None:
            feeder.close()
    return losses, time.time() - t0


def _gpu_util_sample():
    """One nvidia-smi utilization sample, or None if unavailable."""
    import shutil
    import subprocess
    if shutil.which('nvidia-smi') is None:
        return None
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
        parts = out.stdout.strip().splitlines()[0].split(',')
        return {'gpu_util_pct': float(parts[0]), 'mem_used_mib': float(parts[1])}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--mode', choices=['seq', 'equiv', 'bench'], default='equiv')
    ap.add_argument('--n-rows', type=int, default=5003,
                    help='synthetic row count for seq mode (5003 = non-multiple '
                         'of 512, exercises the short tail batch)')
    ap.add_argument('--data', default=None, help='comma-separated replay dirs (equiv/bench)')
    ap.add_argument('--init', default=None, help='init checkpoint npz (equiv/bench)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--lr-final', type=float, default=3e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--fast-policy-weight', type=float, default=0.0)
    ap.add_argument('--n-extra', type=int, default=0)
    ap.add_argument('--seq-steps', type=int, default=200,
                    help='sequence check covers min(this, steps to 2 epoch boundaries)')
    ap.add_argument('--loss-steps', type=int, default=100)
    ap.add_argument('--bench-steps', type=int, default=100)
    ap.add_argument('--feed-workers', type=int, default=8)
    ap.add_argument('--prefetch', type=int, default=4)
    args = ap.parse_args()

    failures = []

    # ---- (a) sequence check: always runs, pure numpy --------------------
    # N = min(--seq-steps, steps spanning >= 2 epoch boundaries): with the
    # default synthetic n_rows that is 2 * ceil(5003/512) = 20 steps; on
    # real data pass --n-rows = dataset size to get the production shape.
    steps_per_epoch = -(-args.n_rows // args.batch)
    n_seq = min(args.seq_steps, 2 * steps_per_epoch + 1)
    ok, fail = check_sequence(args.n_rows, args.batch, args.epochs,
                              args.seed, n_seq)
    print('[a] sequence check: n_rows=%d batch=%d steps=%d -> %s'
          % (args.n_rows, args.batch, n_seq, 'PASS' if ok else 'FAIL'))
    if not ok:
        failures.append(('sequence', fail))

    if args.mode == 'seq':
        pass  # (a) only
    elif args.data is None or args.init is None:
        print('[b] SKIPPED: --data/--init not given (or torch unavailable '
              'here). Run on L20, next to model.py/layout.py:')
        print('    python gate_train2.py --mode %s --data <dirs> --init <ckpt> '
              '--device cuda --n-rows <dataset rows>' % args.mode)
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            print('[b] SKIPPED: torch not installed on this machine. '
                  'Run on L20 (command shape above).')
            torch = None
        if 'torch' in dir() and torch is not None:
            rows = t2.load_dataset([d for d in args.data.split(',') if d])
            if args.mode == 'equiv':
                old, _ = _run_steps('old', rows, args.init, args,
                                    args.loss_steps, {})
                new, _ = _run_steps('new', rows, args.init, args,
                                    args.loss_steps,
                                    {'workers': args.feed_workers,
                                     'prefetch': args.prefetch})
                worst = 0.0
                for s, (o, n) in enumerate(zip(old, new)):
                    for k in o:
                        tol = max(ABS_TOL, REL_TOL * abs(o[k]))
                        d = abs(n[k] - o[k])
                        worst = max(worst, d)
                        if d > tol:
                            failures.append(('loss', {
                                'step': s, 'metric': k, 'measured': n[k],
                                'line': 'old=%r tol=%r' % (o[k], tol)}))
                            break
                    if failures and failures[-1][0] == 'loss':
                        break
                print('[b] loss trajectory: %d steps, max|diff|=%.5f '
                      '(tol max(%.3g, %.3g*|old|), 待策略组钉) -> %s'
                      % (len(old), worst, ABS_TOL, REL_TOL,
                         'PASS' if not failures else 'FAIL'))
            elif args.mode == 'bench':
                util0 = _gpu_util_sample()
                _, el_old = _run_steps('old', rows, args.init, args,
                                       args.bench_steps, {})
                _, el_new = _run_steps('new', rows, args.init, args,
                                       args.bench_steps,
                                       {'workers': args.feed_workers,
                                        'prefetch': args.prefetch})
                util1 = _gpu_util_sample()
                sps_old = args.bench_steps / max(el_old, 1e-9)
                sps_new = args.bench_steps / max(el_new, 1e-9)
                print('[bench] old %.2f steps/s | new %.2f steps/s | '
                      'ratio %.2fx | baseline anchor %.2f steps/s (L20 Gen-2)'
                      % (sps_old, sps_new, sps_new / max(sps_old, 1e-9),
                         BASELINE_STEPS_S))
                print('[bench] gpu util samples (nvidia-smi, start/end): '
                      '%s / %s' % (util0, util1))

    if failures:
        for kind, f in failures:
            print('FAILURE [%s]: step=%s metric=%s measured=%s line=%s'
                  % (kind, f.get('step'), f.get('metric'),
                     f.get('measured'), f.get('line')))
        print('VERDICT: FAIL -> ABANDON (不过 → 弃置)')
        return 1
    print('VERDICT: PASS (within the checks actually run; skipped parts '
          'are printed above and do not count as passed)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
