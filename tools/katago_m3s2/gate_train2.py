"""gate_train2: equivalence gate + bench for train2_fixed.py (bypass piece).

Modes:
  --mode seq    (a) batch-sequence check ONLY. Pure numpy, no torch, no
                GPU, no data files needed: replays train.py's index/RNG
                logic (reference implementation below, transcribed from
                train.py main loop) against
                train2_fixed.generate_index_sequence and compares every
                step's row-index tuple AND augment-draw tuple verbatim
                (set + order).
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

Dependencies (review 4.3 -- the earlier "Runs on any machine" wording
was overbroad; this is the real list):
  * Python 3 and numpy: required for every mode ((a) is pure numpy).
  * train2_fixed.py: must sit next to this file. It is imported LAZILY,
    inside main, so that an import failure still produces a VERDICT
    line and a report file instead of a bare traceback (review 4.3).
  * layout.py: train2_fixed imports it at module load. In the repo it
    lives in tools/katago_m3s2/ next to train.py; in a bare directory,
    put a copy next to this file or add its directory to PYTHONPATH.
  * model.py and torch: only for --mode equiv / bench WITH
    --data/--init (the (b) / bench arms); seq mode never touches them.
  * nvidia-smi: optional, bench GPU-utilization sampling only.

Report discipline (review 4.6): this gate writes its OWN report file
(--report, default gate_train2_report.json in the cwd) on EVERY path --
pass, fail, skipped arms, invalid arguments, import failure and
unexpected exceptions alike -- and also prints the final VERDICT line
to stdout. The report carries a gate_md5 stamp (this file's own md5,
computed at runtime -- the same self-stamp discipline as gate_vec2's)
plus the md5 of the subject file actually run (train2_fixed.py, next
to this gate), every check's readings, and the failure records.

(b) coverage note (review 4.7 -- read before citing (b)): the "old"
arm below is train2_fixed.materialize_numpy driven inline, i.e. an
inline REPLICATION of train.py's main-loop batch assembly
(train.py L283-296), NOT train.py itself. train.py exposes no
importable batch-assembly function: its assembly is inline glue
inside main(), interleaved with the RNG draws and the training step,
so it cannot be called step-by-step without modifying train.py
(which this fix round must not do). What (b) proves is therefore
inline-vs-pooled equivalence of the SAME materialization function;
the replication-vs-train.py link rests on (a)'s transcribed reference
plus the review's manual line-by-line comparison (REVIEW_TRAIN2
section 5.8), not on executing train.py. Do not cite (b) as a
train.py execution check.

Verdict discipline: any failed check prints a final line
  VERDICT: FAIL -> ABANDON (不过 → 弃置)
plus a failure record (step, metric, measured, line). Any exception on
any path -- argument validation, import, a check raising -- is caught
by the bootstrap layer, recorded the same way, and exits nonzero
(review 4.3 / 4.4): this gate never dies without a VERDICT line and a
report file. There is NO retry or auto-fix logic in this file, by
design.

Tolerances for (b) [all marked 待策略组钉 -- pending strategy pinning]:
  GPU BF16 training is not bit-deterministic across feeding orders of
  kernel launches; the loss comparison therefore uses
    |loss_new - loss_old| <= max(ABS_TOL, REL_TOL * |loss_old|)
  with ABS_TOL = 2e-2, REL_TOL = 0.02 as PLACEHOLDERS pending the pin.
  The sequence check (a) has NO tolerance: verbatim equality or FAIL.
"""

# ---------------------------------------------------------------------------
# Minimal bootstrap layer: STDLIB ONLY. Everything in this section must
# stay importable on a bare machine, because it is what guarantees a
# VERDICT line + report file when the heavy imports below fail
# (review 4.3 / 4.4). numpy and train2_fixed are imported lazily,
# inside main, further down.
# ---------------------------------------------------------------------------

import hashlib
import json
import os
import sys
import time
import traceback

ABS_TOL = 2e-2    # 待策略组钉 (placeholder, see docstring)
REL_TOL = 0.02    # 待策略组钉 (placeholder, see docstring)
BASELINE_STEPS_S = 11.78  # L20 Gen-2 measured, war-room supplied
DEFAULT_REPORT = 'gate_train2_report.json'

np = None   # numpy, bound lazily in main (bootstrap stays stdlib-only)
t2 = None   # train2_fixed module, bound lazily in main


def _self_md5():
    """md5 of this gate file itself, computed at runtime (report stamp)."""
    with open(os.path.abspath(__file__), 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _file_md5(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def _subject_md5():
    """md5 of the subject file actually gated (train2_fixed.py, sibling)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'train2_fixed.py')
    return _file_md5(path)


def _report_path_from_argv(argv):
    """Best-effort --report extraction before argparse exists.

    The bootstrap layer must know where the report goes even when
    argument parsing itself fails, so it scans argv directly.
    """
    for i, a in enumerate(argv):
        if a == '--report' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--report='):
            return a.split('=', 1)[1]
    return DEFAULT_REPORT


class _Reporter:
    """Collects output lines / check readings; always lands a report."""

    def __init__(self, report_path):
        self.path = report_path
        self.lines = []
        self.checks = []
        self.failures = []   # list of (kind, failure-dict)
        self.started = time.time()

    def say(self, line):
        print(line, flush=True)
        self.lines.append(line)

    def check(self, name, status, detail):
        self.checks.append({'name': name, 'status': status,
                            'detail': detail})

    def fail(self, kind, failure):
        self.failures.append((kind, failure))

    def write(self, verdict, exit_code):
        report = {
            'gate_md5': _self_md5(),
            'gate_file': os.path.basename(os.path.abspath(__file__)),
            'subject_file': 'train2_fixed.py',
            'subject_md5': _subject_md5(),
            'verdict': verdict,
            'exit_code': exit_code,
            'checks': self.checks,
            'failures': [{'kind': k, 'detail': f}
                         for k, f in self.failures],
            'output_lines': self.lines,
            'elapsed_s': round(time.time() - self.started, 3),
        }
        try:
            with open(self.path, 'w') as f:
                json.dump(report, f, indent=1, default=str)
            print('report written: %s (gate_md5 %s)'
                  % (self.path, report['gate_md5']), flush=True)
        except Exception as exc:  # report path itself broken: still say so
            print('WARNING: could not write report file %s: %r'
                  % (self.path, exc), flush=True)
        return report


def _finish(rep):
    """Print failure records + the final VERDICT line, write the report."""
    if rep.failures:
        for kind, f in rep.failures:
            rep.say('FAILURE [%s]: step=%s metric=%s measured=%s line=%s'
                    % (kind, f.get('step'), f.get('metric'),
                       f.get('measured'), f.get('line')))
        verdict = 'FAIL -> ABANDON (不过 → 弃置)'
        rep.say('VERDICT: ' + verdict)
        rep.write(verdict, 1)
        return 1
    verdict = ('PASS (within the checks actually run; skipped parts '
               'are printed above and do not count as passed)')
    rep.say('VERDICT: ' + verdict)
    rep.write(verdict, 0)
    return 0


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
# NOTE (review 4.7): feed_mode 'old' is train2_fixed.materialize_numpy
# driven inline -- a replication of train.py's inline assembly, NOT
# train.py itself (see the module docstring's coverage note).
# ---------------------------------------------------------------------------

def _run_steps(feed_mode, rows, net_init_path, args, n_steps, feeder_kw):
    """Run n_steps of training with old (inline) or new (pooled) feeding.

    Replicates train.py's step math verbatim (same loss code as
    train2_fixed's main). Returns (losses list of dicts, elapsed_seconds).
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


# ---------------------------------------------------------------------------
# Argument validation (review 4.4): every rule is checked up front and
# every violation is named; any violation -> FAIL verdict + report.
# ---------------------------------------------------------------------------

def _validate_args(args):
    problems = []
    if args.batch < 1:
        problems.append('--batch must be >= 1, got %r' % (args.batch,))
    if args.epochs < 1:
        problems.append('--epochs must be >= 1, got %r' % (args.epochs,))
    if args.n_rows < 1:
        problems.append('--n-rows must be >= 1, got %r' % (args.n_rows,))
    if args.seq_steps < 1:
        problems.append('--seq-steps must be >= 1, got %r' % (args.seq_steps,))
    if args.loss_steps < 1:
        problems.append('--loss-steps must be >= 1, got %r' % (args.loss_steps,))
    if args.bench_steps < 1:
        problems.append('--bench-steps must be >= 1, got %r' % (args.bench_steps,))
    if args.feed_workers < 0:
        problems.append('--feed-workers must be >= 0, got %r' % (args.feed_workers,))
    if args.prefetch < 1:
        problems.append('--prefetch must be >= 1, got %r' % (args.prefetch,))
    if args.n_extra < 0:
        problems.append('--n-extra must be >= 0, got %r' % (args.n_extra,))
    if not (0 <= args.seed < 2 ** 32):
        problems.append('--seed must be in [0, 2**32), got %r' % (args.seed,))
    if args.lr <= 0:
        problems.append('--lr must be > 0, got %r' % (args.lr,))
    if args.lr_final < 0:
        problems.append('--lr-final must be >= 0, got %r' % (args.lr_final,))
    if args.weight_decay < 0:
        problems.append('--weight-decay must be >= 0, got %r' % (args.weight_decay,))
    if args.fast_policy_weight < 0:
        problems.append('--fast-policy-weight must be >= 0, got %r'
                        % (args.fast_policy_weight,))
    return problems


def _main_inner(rep):
    global np, t2
    import argparse
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
    ap.add_argument('--report', default=DEFAULT_REPORT,
                    help='report file this gate writes on every path '
                         '(default: %(default)s in the cwd)')
    args = ap.parse_args()
    rep.path = args.report

    # ---- entry validation (review 4.4) ----------------------------------
    problems = _validate_args(args)
    for p in problems:
        rep.say('INVALID ARGUMENT: ' + p)
        rep.fail('arguments', {'step': None, 'metric': 'argument',
                               'measured': p,
                               'line': 'entry validation (review 4.4)'})
    if problems:
        rep.check('arguments', 'FAIL', {'problems': problems})
        return _finish(rep)
    rep.check('arguments', 'PASS', {'mode': args.mode, 'batch': args.batch,
                                    'epochs': args.epochs,
                                    'n_rows': args.n_rows, 'seed': args.seed})

    # ---- lazy heavy imports (review 4.3): failure -> verdict, not a bare
    # traceback. numpy first, then the subject module (which itself
    # imports layout at module load).
    try:
        import numpy as _np
        np = _np
    except BaseException as exc:
        rep.fail('import', {'step': None, 'metric': 'import numpy',
                            'measured': repr(exc),
                            'line': 'numpy is required for every mode'})
        rep.check('import', 'FAIL', {'module': 'numpy', 'error': repr(exc)})
        return _finish(rep)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import train2_fixed as _t2
        t2 = _t2
    except BaseException as exc:
        rep.fail('import', {'step': None, 'metric': 'import train2_fixed',
                            'measured': repr(exc),
                            'line': 'train2_fixed.py must sit next to this '
                                    'gate and its own imports (layout.py) '
                                    'must be importable -- see docstring '
                                    'Dependencies'})
        rep.check('import', 'FAIL',
                  {'module': 'train2_fixed', 'error': repr(exc)})
        return _finish(rep)
    rep.check('import', 'PASS', {'subject_md5': _subject_md5()})

    failures = rep.failures

    # ---- (a) sequence check: always runs, pure numpy --------------------
    # N = min(--seq-steps, steps spanning >= 2 epoch boundaries): with the
    # default synthetic n_rows that is 2 * ceil(5003/512) = 20 steps; on
    # real data pass --n-rows = dataset size to get the production shape.
    steps_per_epoch = -(-args.n_rows // args.batch)
    n_seq = min(args.seq_steps, 2 * steps_per_epoch + 1)
    ok, fail = check_sequence(args.n_rows, args.batch, args.epochs,
                              args.seed, n_seq)
    rep.say('[a] sequence check: n_rows=%d batch=%d steps=%d -> %s'
            % (args.n_rows, args.batch, n_seq, 'PASS' if ok else 'FAIL'))
    rep.check('sequence', 'PASS' if ok else 'FAIL',
              {'n_rows': args.n_rows, 'batch': args.batch, 'steps': n_seq,
               'failure': fail})
    if not ok:
        failures.append(('sequence', fail))

    if args.mode == 'seq':
        pass  # (a) only
    elif args.data is None or args.init is None:
        rep.say('[b] SKIPPED: --data/--init not given (or torch unavailable '
                'here). Run on L20, next to model.py/layout.py:')
        rep.say('    python gate_train2.py --mode %s --data <dirs> --init <ckpt> '
                '--device cuda --n-rows <dataset rows>' % args.mode)
        rep.check('loss_or_bench', 'SKIPPED',
                  {'reason': '--data/--init not given'})
    else:
        torch = None
        try:
            import torch as _torch  # noqa: F401
            torch = _torch
        except ImportError:
            rep.say('[b] SKIPPED: torch not installed on this machine. '
                    'Run on L20 (command shape above).')
            rep.check('loss_or_bench', 'SKIPPED',
                      {'reason': 'torch not installed'})
        if torch is not None:
            rows = t2.load_dataset([d for d in args.data.split(',') if d])
            if args.mode == 'equiv':
                rep.say('[b] NOTE (review 4.7): old arm = train2_fixed '
                        'inline materialization, a replication of '
                        'train.py\'s inline assembly -- NOT train.py '
                        'itself; see docstring coverage note.')
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
                rep.say('[b] loss trajectory: %d steps, max|diff|=%.5f '
                        '(tol max(%.3g, %.3g*|old|), 待策略组钉) -> %s'
                        % (len(old), worst, ABS_TOL, REL_TOL,
                           'PASS' if not failures else 'FAIL'))
                rep.check('loss_trajectory',
                          'FAIL' if any(k == 'loss' for k, _ in failures)
                          else 'PASS',
                          {'steps_compared': len(old), 'max_abs_diff': worst,
                           'abs_tol': ABS_TOL, 'rel_tol': REL_TOL})
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
                rep.say('[bench] old %.2f steps/s | new %.2f steps/s | '
                        'ratio %.2fx | baseline anchor %.2f steps/s (L20 Gen-2)'
                        % (sps_old, sps_new, sps_new / max(sps_old, 1e-9),
                           BASELINE_STEPS_S))
                rep.say('[bench] gpu util samples (nvidia-smi, start/end): '
                        '%s / %s' % (util0, util1))
                rep.check('bench', 'MEASURED',
                          {'old_steps_s': sps_old, 'new_steps_s': sps_new,
                           'ratio': sps_new / max(sps_old, 1e-9),
                           'baseline_anchor_steps_s': BASELINE_STEPS_S,
                           'gpu_util_start': util0, 'gpu_util_end': util1})

    return _finish(rep)


def main():
    """Bootstrap wrapper: NO exception path leaves without a verdict.

    _main_inner handles validation / import / check failures itself.
    Anything that still escapes (a check raising, an unexpected error)
    is converted here into a FAIL verdict + report + nonzero exit
    (review 4.4). A SystemExit raised by argparse inside _main_inner
    (usage error, before any verdict existed) is likewise converted
    here; --help's SystemExit(0) is passed through untouched.
    """
    rep = _Reporter(_report_path_from_argv(sys.argv[1:]))
    try:
        return _main_inner(rep)
    except SystemExit as se:
        if se.code in (0, None):
            return 0
        rep.fail('arguments',
                 {'step': None, 'metric': 'SystemExit',
                  'measured': repr(se.code),
                  'line': 'argument parsing failed (argparse); '
                          'see usage error on stderr'})
        rep.check('arguments', 'FAIL', {'system_exit': repr(se.code)})
        return _finish(rep)
    except BaseException as exc:
        traceback.print_exc()
        rep.fail('exception', {'step': None,
                               'metric': type(exc).__name__,
                               'measured': repr(exc),
                               'line': 'unhandled exception in gate main; '
                                       'traceback on stderr'})
        rep.check('main', 'FAIL', {'exception': repr(exc)})
        return _finish(rep)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as _exc:  # last-resort guard, should not happen
        # (SystemExit is not an Exception subclass and propagates as-is:
        # main() has already turned every error path into a verdict.)
        _rep = _Reporter(_report_path_from_argv(sys.argv[1:]))
        _rep.fail('exception', {'step': None,
                                'metric': type(_exc).__name__,
                                'measured': repr(_exc),
                                'line': 'exception escaped main()'})
        _finish(_rep)
        sys.exit(1)
