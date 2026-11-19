"""Equivalence gate for vec_selfplay2.py vs vec_selfplay.py.

Positioning (owner ruling 2026-11-20 03:12): vec_selfplay2 is a BYPASS
artifact.  This gate is a one-shot verdict, not the start of a repair
loop: if the gate fails, vec_selfplay2 is DISCARDED as-is.  The script
therefore contains no retry/repair logic; its output ends with an
explicit verdict line （通过 / 不过 → 弃置） and a failure-point record
({check, metric, measured, line}) for every failed hard check.

Pre-registered checks (the gate must pass before vec_selfplay2 output
may enter production training data):

(a) Forward consistency on a fixed position batch.  The same fixed
    batch of positions (drawn from an existing replay shard, hence real
    game positions) is evaluated two ways with the same nets: row-by-row
    (batch 1 -- the shape vec_selfplay.py's fast path actually produces)
    and as one assembled batch (vec_selfplay2.py's transport).  Report
    max|dlw|, max|dlb|, max|dv| per net tag.
    Lines: HARD max|.| <= 0.25 -- the magnitude of pack_submission.py's
    FORWARD_TOL (the tolerance this line already accepts for float16
    payload storage of the same logits); REPORT line <= 0.02 (stricter
    observation line; exceeding it does not fail the gate but must be
    recorded for 策略组 -- NEW line, 待策略组钉).

(b) Distribution equivalence on a fixed-seed small batch of games
    (default 200).  Both generators are run (or have been run) with the
    same --seed/--games/--deep-frac/--ckpt/--src; the produced shards
    are compared on the row-level distributions that the shards carry:
    final margin, win, searched fraction, aux_delta -- plus total
    rows per game.  Game LENGTH is fixed by construction (the chassis'
    LAST_TURN pins every game to the same number of decision steps and
    both generators share the chassis next_state verbatim), so length
    equivalence is checked through the rows-per-game identity and the
    step semantics, not through a length histogram the shards do not
    carry.
    Lines (all NEW, 待策略组钉, chosen at the scale of run-to-run noise
    for 200 games): two-sample KS statistic D <= 0.05 for margin;
    |mean margin shift| <= 0.05 * pooled SD; |win-rate shift| <= 0.02;
    |searched-fraction shift| <= 0.01; rows-per-game relative shift
    <= 0.01.

(c) Shard schema mechanical check.  Every shard in both directories
    must have exactly SHARD_KEYS in order, with the per-key dtype and
    trailing shape of vec_selfplay.py's write_shard / record
    construction.  Any deviation fails the gate outright.

Usage:
  python3 gate_vec2.py --shard FIXED.npz --ckpt CKPT.npz [--src SRC]
      [--device cuda] [--dtype bf16] [--batch-rows 512]      # (a) + (c)
  python3 gate_vec2.py --old-dir DIR --new-dir DIR           # (b) + (c)
  python3 gate_vec2.py --run-games 200 --seed 1 --ckpt CKPT.npz
      --vec1 tools/katago_m3s2/vec_selfplay.py
      --vec2 vec_selfplay2.py --workdir /tmp/gate_run ...     # runs (b)

Exit code 0 iff every applicable hard check passes.  A JSON report is
written to --report (default: gate_report.json in the cwd).
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHARD_KEYS = ['x', 'sc', 'tw', 'tb', 'wmask', 'bmask', 'carbon', 'margin',
              'aux_delta', 'win', 'searched']

# key -> (dtype, trailing shape) from vec_selfplay.py's record
# construction + write_shard (np.stack adds the leading row axis).
SCHEMA = {
    'x': (np.uint8, (54, 15, 15)),
    'sc': (np.float32, (5,)),
    'tw': (np.float16, (225, 5)),
    'tb': (np.float16, (225, 3)),
    'wmask': (np.uint8, (225,)),
    'bmask': (np.uint8, (225,)),
    'carbon': (np.float16, (225,)),
    'margin': (np.float32, ()),
    'aux_delta': (np.float32, (3,)),
    'win': (np.float32, ()),
    'searched': (np.uint8, ()),
}


# ---------------------------------------------------------------- (c)
def check_schema_dir(d):
    """Return (n_files, n_rows, problems[]) for one shard directory."""
    problems = []
    files = sorted(glob.glob(os.path.join(d, '*.npz')))
    n_rows = 0
    if not files:
        return 0, 0, ['no .npz shards in %s' % d]
    for path in files:
        with np.load(path) as z:
            keys = list(z.keys())
            if keys != SHARD_KEYS:
                problems.append('%s: key list/order %r != SHARD_KEYS'
                                % (os.path.basename(path), keys))
                continue
            n = None
            for k in SHARD_KEYS:
                dt, tail = SCHEMA[k]
                a = z[k]
                if a.dtype != np.dtype(dt):
                    problems.append('%s:%s dtype %s != %s'
                                    % (os.path.basename(path), k, a.dtype, np.dtype(dt)))
                if a.shape[1:] != tail:
                    problems.append('%s:%s shape %s tail != %r'
                                    % (os.path.basename(path), k, a.shape, tail))
                if n is None:
                    n = a.shape[0]
                elif a.shape[0] != n:
                    problems.append('%s:%s row count %d != %d'
                                    % (os.path.basename(path), k, a.shape[0], n))
            n_rows += n or 0
    return len(files), n_rows, problems


# ---------------------------------------------------------------- (b)
def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov D, plain numpy."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    grid = np.concatenate([a, b])
    ea = np.searchsorted(a, grid, side='right') / len(a)
    eb = np.searchsorted(b, grid, side='right') / len(b)
    return float(np.max(np.abs(ea - eb)))


def load_rows(d):
    cols = {k: [] for k in ('margin', 'win', 'searched', 'aux_delta')}
    for path in sorted(glob.glob(os.path.join(d, '*.npz'))):
        with np.load(path) as z:
            for k in cols:
                cols[k].append(z[k].astype(np.float64))
    return {k: (np.concatenate(v) if v else np.empty(0)) for k, v in cols.items()}


def check_distributions(old_dir, new_dir, games, lines):
    old = load_rows(old_dir)
    new = load_rows(new_dir)
    out = {}
    mo, mn = old['margin'], new['margin']
    pooled_sd = float(np.sqrt((mo.var() + mn.var()) / 2.0)) or 1.0
    out['margin'] = {
        'mean_old': float(mo.mean()), 'mean_new': float(mn.mean()),
        'sd_old': float(mo.std()), 'sd_new': float(mn.std()),
        'q_old': [float(np.quantile(mo, q)) for q in (0.1, 0.5, 0.9)],
        'q_new': [float(np.quantile(mn, q)) for q in (0.1, 0.5, 0.9)],
        'ks_D': ks_statistic(mo, mn),
        'mean_shift_over_pooled_sd': float(abs(mo.mean() - mn.mean()) / pooled_sd),
    }
    out['win_rate'] = {'old': float(old['win'].mean()),
                       'new': float(new['win'].mean()),
                       'shift': float(abs(old['win'].mean() - new['win'].mean()))}
    out['searched_frac'] = {'old': float(old['searched'].mean()),
                            'new': float(new['searched'].mean()),
                            'shift': float(abs(old['searched'].mean()
                                               - new['searched'].mean()))}
    out['aux_delta_ks_D'] = {
        'h%d' % (i + 1): ks_statistic(old['aux_delta'][:, i], new['aux_delta'][:, i])
        for i in range(old['aux_delta'].shape[1])}
    rpg_old = len(mo) / float(games)
    rpg_new = len(mn) / float(games)
    out['rows_per_game'] = {'old': rpg_old, 'new': rpg_new,
                            'rel_shift': float(abs(rpg_old - rpg_new) / rpg_old)}
    verdicts = {
        'ks_margin': out['margin']['ks_D'] <= lines['ks'],
        'mean_margin': out['margin']['mean_shift_over_pooled_sd'] <= lines['mean_shift'],
        'win_rate': out['win_rate']['shift'] <= lines['win_shift'],
        'searched_frac': out['searched_frac']['shift'] <= lines['searched_shift'],
        'rows_per_game': out['rows_per_game']['rel_shift'] <= lines['rpg_rel'],
    }
    out['verdicts'] = verdicts
    out['pass'] = all(verdicts.values())
    measured = {
        'ks_margin': (out['margin']['ks_D'], lines['ks']),
        'mean_margin': (out['margin']['mean_shift_over_pooled_sd'], lines['mean_shift']),
        'win_rate': (out['win_rate']['shift'], lines['win_shift']),
        'searched_frac': (out['searched_frac']['shift'], lines['searched_shift']),
        'rows_per_game': (out['rows_per_game']['rel_shift'], lines['rpg_rel']),
    }
    out['failure_points'] = [
        {'check': 'distributions', 'metric': k,
         'measured': measured[k][0], 'line': measured[k][1]}
        for k, passed in verdicts.items() if not passed]
    return out


# ---------------------------------------------------------------- (a)
def check_forward(shard_path, ckpt, src, device, dtype, batch_rows,
                  tol_hard, tol_report):
    """Fixed-batch forward consistency: batch-1 vs assembled batch."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import vec_selfplay2 as v2
    v2._load_deps()
    layout = v2.layout
    if src is None:
        src = layout.MASTER3S2
    with np.load(shard_path) as z:
        X = z['x'][:batch_rows].astype(np.float32) / 100.0
        SC = z['sc'][:batch_rows].astype(np.float32)
    backend = v2.TorchBatchBackend({'src': src, 'ckpt': ckpt, 'hof': [],
                                    'device': device, 'dtype': dtype,
                                    'threads': 4, 'batch_cap': 1 << 30})
    out = {}
    for tag in ('cur', 'm3s2'):
        rows_lw, rows_lb, rows_v = [], [], []
        for i in range(len(X)):
            lw, lb, v = backend.forward_batch(tag, X[i:i + 1], SC[i:i + 1])
            rows_lw.append(lw[0])
            rows_lb.append(lb[0])
            rows_v.append(v[0])
        blw, blb, bv = backend.forward_batch(tag, X, SC)
        d = {'max_dlw': float(np.max(np.abs(blw - np.stack(rows_lw)))),
             'max_dlb': float(np.max(np.abs(blb - np.stack(rows_lb)))),
             'max_dv': float(np.max(np.abs(bv - np.stack(rows_v))))}
        d['hard_pass'] = max(d.values()) <= tol_hard
        d['report_pass'] = max(d.values()) <= tol_report
        out[tag] = d
    out['pass'] = all(out[t]['hard_pass'] for t in ('cur', 'm3s2'))
    out['failure_points'] = [
        {'check': 'forward', 'metric': '%s max|dlw/dlb/dv|' % t,
         'measured': max(out[t]['max_dlw'], out[t]['max_dlb'], out[t]['max_dv']),
         'line': tol_hard}
        for t in ('cur', 'm3s2') if not out[t]['hard_pass']]
    return out


# ---------------------------------------------------------------- run
def run_generators(args):
    old_dir = os.path.join(args.workdir, 'old')
    new_dir = os.path.join(args.workdir, 'new')
    os.makedirs(old_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)
    base = ['--src', args.src, '--ckpt', args.ckpt, '--games', str(args.run_games),
            '--seed', str(args.seed), '--deep-frac', str(args.deep_frac),
            '--device', args.device, '--dtype', args.dtype]
    cmd1 = [sys.executable, args.vec1] + base + ['--workers', str(args.workers),
                                                 '--out', old_dir]
    cmd2 = [sys.executable, args.vec2] + base + ['--pool-games', str(args.pool_games),
                                                 '--out', new_dir]
    for cmd in (cmd1, cmd2):
        print('gate run:', ' '.join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit('generator failed (rc=%d): %s' % (rc, cmd[1]))
    return old_dir, new_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--shard', default=None, help='shard for the fixed batch (a)')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--src', default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    ap.add_argument('--batch-rows', type=int, default=512)
    ap.add_argument('--tol-hard', type=float, default=0.25,
                    help='hard line for (a): pack_submission FORWARD_TOL magnitude')
    ap.add_argument('--tol-report', type=float, default=0.02,
                    help='stricter report line for (a) (NEW, pending strategy)')
    ap.add_argument('--old-dir', default=None)
    ap.add_argument('--new-dir', default=None)
    ap.add_argument('--run-games', type=int, default=0,
                    help='>0: run both generators for this many games first (b)')
    ap.add_argument('--vec1', default=None, help='path to vec_selfplay.py')
    ap.add_argument('--vec2', default=None, help='path to vec_selfplay2.py')
    ap.add_argument('--workdir', default='/tmp/gate_vec2_run')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--pool-games', type=int, default=32)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--deep-frac', type=float, default=0.25)
    ap.add_argument('--games', type=int, default=0,
                    help='games denominator for rows-per-game (default: --run-games)')
    ap.add_argument('--ks-line', type=float, default=0.05)
    ap.add_argument('--mean-shift-line', type=float, default=0.05)
    ap.add_argument('--win-shift-line', type=float, default=0.02)
    ap.add_argument('--searched-shift-line', type=float, default=0.01)
    ap.add_argument('--rpg-rel-line', type=float, default=0.01)
    ap.add_argument('--report', default='gate_report.json')
    args = ap.parse_args()

    report = {'lines': {'tol_hard_FORWARD_TOL_magnitude': args.tol_hard,
                        'tol_report_NEW_pending_strategy': args.tol_report,
                        'ks_NEW_pending_strategy': args.ks_line,
                        'mean_shift_NEW_pending_strategy': args.mean_shift_line,
                        'win_shift_NEW_pending_strategy': args.win_shift_line,
                        'searched_shift_NEW_pending_strategy': args.searched_shift_line,
                        'rpg_rel_NEW_pending_strategy': args.rpg_rel_line}}
    ok = True

    if args.run_games > 0:
        if not (args.vec1 and args.vec2 and args.ckpt):
            raise SystemExit('--run-games needs --vec1/--vec2/--ckpt')
        args.old_dir, args.new_dir = run_generators(args)
        if not args.games:
            args.games = args.run_games

    for name, d in (('old', args.old_dir), ('new', args.new_dir)):
        if d:
            nf, nr, problems = check_schema_dir(d)
            report['schema_%s' % name] = {'files': nf, 'rows': nr,
                                          'problems': problems}
            ok = ok and not problems
    if args.shard and not (args.old_dir or args.new_dir):
        nf, nr, problems = check_schema_dir(os.path.dirname(args.shard) or '.')
        report['schema_shard_dir'] = {'files': nf, 'rows': nr, 'problems': problems}
        ok = ok and not problems

    if args.old_dir and args.new_dir:
        if not args.games:
            raise SystemExit('--games (or --run-games) is required for (b)')
        lines = {'ks': args.ks_line, 'mean_shift': args.mean_shift_line,
                 'win_shift': args.win_shift_line,
                 'searched_shift': args.searched_shift_line,
                 'rpg_rel': args.rpg_rel_line}
        report['distributions'] = check_distributions(
            args.old_dir, args.new_dir, args.games, lines)
        ok = ok and report['distributions']['pass']

    if args.shard:
        if not args.ckpt:
            raise SystemExit('--ckpt is required for (a)')
        report['forward'] = check_forward(
            args.shard, args.ckpt, args.src, args.device, args.dtype,
            args.batch_rows, args.tol_hard, args.tol_report)
        ok = ok and report['forward']['pass']

    # Failure-point record: one entry per failed hard check, always in
    # the form {check, metric, measured, line}.  There is deliberately
    # NO retry / repair logic anywhere in this gate: vec_selfplay2 is a
    # bypass artifact -- fail means discard (owner ruling 2026-11-20).
    failure_points = []
    for k in report:
        if k.startswith('schema'):
            for p in report[k]['problems']:
                failure_points.append({'check': 'schema', 'metric': p,
                                       'measured': 'mismatch',
                                       'line': 'SHARD_KEYS/schema verbatim'})
    for section in ('distributions', 'forward'):
        if section in report:
            failure_points.extend(report[section].get('failure_points', []))
    report['failure_points'] = failure_points
    report['gate_pass'] = bool(ok)
    report['verdict'] = ('PASS' if ok else
                         'FAIL -> DISCARD vec_selfplay2 (bypass artifact; '
                         'no iterative repair loop)')
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({'gate_pass': bool(ok),
                      'forward': report.get('forward'),
                      'distributions_pass': (report.get('distributions') or {}).get('pass'),
                      'schema_problems': [report[k]['problems'] for k in report
                                          if k.startswith('schema') and report[k]['problems']]},
                     indent=1))
    print('==== 等价性闸结论 ====')
    if ok:
        print('结论：通过（vec_selfplay2 可进入生产数据路径的下一审批步）')
    else:
        print('结论：不过 → 弃置（vec_selfplay2 为旁路件，闸不过即弃置，'
              '不开迭代修复循环）')
        print('失败点记录（指标 / 实测值 / 线值）：')
        for fp in failure_points:
            print('  - [%s] %s：实测 %s，线 %s'
                  % (fp['check'], fp['metric'], fp['measured'], fp['line']))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
