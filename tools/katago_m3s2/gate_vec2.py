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
    Tag coverage: cur, m3s2 AND the hof series (hof0, hof1, ...), all
    under the same per-row-vs-whole-batch comparison and the same two
    lines.  Hof nets are loaded by the backend from the --hof npz list;
    when no --hof list is given, the --ckpt checkpoint itself (the gen0
    checkpoint) is used as the single hof fixture, so hof0 is covered
    BY DEFAULT.  The report's forward section lists the actually
    covered tag set explicitly (tags_covered); a run whose report does
    not list a tag has NOT covered that tag.

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
    aux_delta: the per-horizon (h1/h2/h3) two-sample KS statistics ARE
    judged, with the SAME form and the SAME line value as the margin
    KS above (D <= lines['ks'], i.e. --ks-line, default 0.05) -- the
    (b) section's existing KS convention, inherited verbatim
    （承袭惯例）.  This aux_delta line is PROVISIONAL, pending
    calibration （暂行、待校准）: the measured per-horizon KS values are
    recorded in the report as the calibration basis for 策略组's final
    ruling (confirm / re-line / demote back to report-only).
    Run mode (--run-games): the gate passes --value-scale and --threads
    through to BOTH generators verbatim, and the report records both
    generators' actual command lines (run_commands) plus the effective
    value_scale/threads, so the u-bridge path is exercised at run
    level, not only at code level.

(c) Shard schema mechanical check.  Every shard in both directories
    must have exactly SHARD_KEYS in order, with the per-key dtype and
    trailing shape of vec_selfplay.py's write_shard / record
    construction.  Any deviation fails the gate outright.

Usage:
  python3 gate_vec2.py --shard FIXED.npz --ckpt CKPT.npz [--src SRC]
      [--hof HOF0.npz,...] [--device cuda] [--dtype bf16]
      [--batch-rows 512]      # (a) + (c); no --hof => ckpt is hof0
  python3 gate_vec2.py --old-dir DIR --new-dir DIR           # (b) + (c)
  python3 gate_vec2.py --run-games 200 --seed 1 --ckpt CKPT.npz
      --vec1 tools/katago_m3s2/vec_selfplay.py
      --vec2 vec_selfplay2.py --workdir /tmp/gate_run ...     # runs (b)

Claim scope （声称划线）: any equivalence claim from this gate covers
ONLY (actually-run forward tags) x (actually-run --value-scale mode)
x (shard schema mechanical identity), with --src bounded to the
master3s2 source (md5 57f9a594cce4f2ec9e3573d8d9d26ca1).  Other
chassis (fix3 etc.) are outside this gate's claim.  The report
carries a claim_scope section stating the actually-run values, and a
gate_md5 stamp (this file's own md5, computed at runtime) so every
re-verification report names the exact gate bytes that produced it.

Exit code 0 iff every applicable hard check passes.  A JSON report is
written to --report (default: gate_report.json in the cwd).
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _self_md5():
    """md5 of this gate file itself, computed at runtime (report stamp)."""
    with open(os.path.abspath(__file__), 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

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
    # aux_delta IS judged (strategy ruling d291d4c5): same form and same
    # line value as the margin KS verdict -- the (b) section's existing
    # KS convention (lines['ks'], --ks-line, default 0.05), inherited
    # verbatim.  The line is PROVISIONAL, pending calibration
    # （暂行、待校准）; the measured values above are the calibration
    # basis for the final ruling.
    out['aux_delta_line'] = {
        'value': lines['ks'],
        'form': 'two-sample KS D <= line, per horizon (h1/h2/h3)',
        'source': "inherited: (b) section margin KS convention (lines['ks'])",
        'status': 'PROVISIONAL pending calibration （暂行、待校准）'}
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
    for hname, kval in out['aux_delta_ks_D'].items():
        verdicts['aux_delta_%s' % hname] = kval <= lines['ks']
    out['verdicts'] = verdicts
    out['pass'] = all(verdicts.values())
    measured = {
        'ks_margin': (out['margin']['ks_D'], lines['ks']),
        'mean_margin': (out['margin']['mean_shift_over_pooled_sd'], lines['mean_shift']),
        'win_rate': (out['win_rate']['shift'], lines['win_shift']),
        'searched_frac': (out['searched_frac']['shift'], lines['searched_shift']),
        'rows_per_game': (out['rows_per_game']['rel_shift'], lines['rpg_rel']),
    }
    for hname, kval in out['aux_delta_ks_D'].items():
        measured['aux_delta_%s' % hname] = (kval, lines['ks'])
    out['failure_points'] = [
        {'check': 'distributions', 'metric': k,
         'measured': measured[k][0], 'line': measured[k][1]}
        for k, passed in verdicts.items() if not passed]
    return out


# ---------------------------------------------------------------- (a)
def check_forward(shard_path, ckpt, src, device, dtype, batch_rows,
                  tol_hard, tol_report, hof=None):
    """Fixed-batch forward consistency: batch-1 vs assembled batch.

    Tags: cur, m3s2 and the hof series.  `hof` is a list of hof npz
    paths; when it is None/empty, [ckpt] (the gen0 checkpoint) is used
    as the single hof fixture, so hof0 is covered by default.  The
    backend interface is unchanged: TorchBatchBackend already loads
    cfg['hof'] paths as nets hof0, hof1, ... with the same
    forward_batch(tag, X, SC) call form as cur/m3s2 -- the only data
    difference is the net source (load_main_npz of the hof path vs the
    chassis source payload for m3s2); inputs need no other
    construction.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import vec_selfplay2 as v2
    v2._load_deps()
    layout = v2.layout
    if src is None:
        src = layout.MASTER3S2
    with np.load(shard_path) as z:
        X = z['x'][:batch_rows].astype(np.float32) / 100.0
        SC = z['sc'][:batch_rows].astype(np.float32)
    hof_explicit = bool(hof)
    hof_paths = list(hof) if hof_explicit else [ckpt]
    backend = v2.TorchBatchBackend({'src': src, 'ckpt': ckpt, 'hof': hof_paths,
                                    'device': device, 'dtype': dtype,
                                    'threads': 4, 'batch_cap': 1 << 30})
    tags = ['cur', 'm3s2'] + ['hof%d' % i for i in range(len(hof_paths))]
    out = {'tags_covered': tags,
           'hof_sources': hof_paths,
           'hof_mode': ('explicit --hof list' if hof_explicit else
                        'gen0 ckpt fixture (default coverage: --ckpt '
                        'itself loaded as hof0)')}
    for tag in tags:
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
        # report_pass fix (REVIEW_VEC2 S4): derive BOTH booleans from
        # the max of the three float deviations ONLY.  The old form
        # took max(d.values()) AFTER inserting hard_pass, so the bool
        # (True == 1) poisoned the report comparison and report_pass
        # was False for every input.
        m = max(d['max_dlw'], d['max_dlb'], d['max_dv'])
        d['hard_pass'] = m <= tol_hard
        d['report_pass'] = m <= tol_report
        out[tag] = d
    out['pass'] = all(out[t]['hard_pass'] for t in tags)
    out['failure_points'] = [
        {'check': 'forward', 'metric': '%s max|dlw/dlb/dv|' % t,
         'measured': max(out[t]['max_dlw'], out[t]['max_dlb'], out[t]['max_dv']),
         'line': tol_hard}
        for t in tags if not out[t]['hard_pass']]
    return out


# ---------------------------------------------------------------- run
def run_generators(args):
    old_dir = os.path.join(args.workdir, 'old')
    new_dir = os.path.join(args.workdir, 'new')
    os.makedirs(old_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)
    # --value-scale / --threads passthrough (REVIEW_VEC2 5.5): both
    # generators receive the gate's values verbatim, so the u-bridge
    # path runs at gate-run level.  getattr defaults keep direct
    # programmatic callers on the generators' own defaults.
    base = ['--src', args.src, '--ckpt', args.ckpt, '--games', str(args.run_games),
            '--seed', str(args.seed), '--deep-frac', str(args.deep_frac),
            '--device', args.device, '--dtype', args.dtype,
            '--value-scale', getattr(args, 'value_scale', 'terminal'),
            '--threads', str(getattr(args, 'threads', 4))]
    cmd1 = [sys.executable, args.vec1] + base + ['--workers', str(args.workers),
                                                 '--out', old_dir]
    cmd2 = [sys.executable, args.vec2] + base + ['--pool-games', str(args.pool_games),
                                                 '--out', new_dir]
    args.run_commands = [' '.join(cmd1), ' '.join(cmd2)]
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
    ap.add_argument('--hof', default='',
                    help='comma-separated hof npz paths for (a); empty => '
                         '--ckpt itself is the hof0 fixture (default coverage)')
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
    ap.add_argument('--value-scale', choices=['terminal', 'u'], default='terminal',
                    help='passed through to BOTH generators in --run-games mode')
    ap.add_argument('--threads', type=int, default=4,
                    help='passed through to BOTH generators in --run-games mode')
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

    report = {'gate_md5': _self_md5(),
              'gate_file': os.path.basename(os.path.abspath(__file__)),
              'lines': {'tol_hard_FORWARD_TOL_magnitude': args.tol_hard,
                        'tol_report_NEW_pending_strategy': args.tol_report,
                        'ks_NEW_pending_strategy': args.ks_line,
                        'mean_shift_NEW_pending_strategy': args.mean_shift_line,
                        'win_shift_NEW_pending_strategy': args.win_shift_line,
                        'searched_shift_NEW_pending_strategy': args.searched_shift_line,
                        'rpg_rel_NEW_pending_strategy': args.rpg_rel_line,
                        'aux_delta_ks_PROVISIONAL_pending_calibration': args.ks_line,
                        'aux_delta_ks_source': "inherited (b) margin KS "
                                               "convention, same form/value"}}
    ok = True

    if args.run_games > 0:
        if not (args.vec1 and args.vec2 and args.ckpt):
            raise SystemExit('--run-games needs --vec1/--vec2/--ckpt')
        args.old_dir, args.new_dir = run_generators(args)
        report['run_commands'] = list(args.run_commands)
        report['run_params'] = {'value_scale': args.value_scale,
                                'threads': args.threads,
                                'seed': args.seed, 'run_games': args.run_games,
                                'deep_frac': args.deep_frac}
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
        hof_list = [p for p in args.hof.split(',') if p]
        report['forward'] = check_forward(
            args.shard, args.ckpt, args.src, args.device, args.dtype,
            args.batch_rows, args.tol_hard, args.tol_report, hof=hof_list)
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
    # Claim scope （声称划线）: only what this invocation actually ran.
    if args.run_games > 0:
        vs_mode = '%s (gate-run, passed through to both generators)' % args.value_scale
    elif args.old_dir and args.new_dir:
        vs_mode = ('not gate-run: (b) compared pre-generated shard dirs; '
                   'their value-scale is the producer config of those shards')
    else:
        vs_mode = 'not applicable (no distribution run in this invocation)'
    report['claim_scope'] = {
        'forward_tags_covered': (report.get('forward') or {}).get('tags_covered'),
        'value_scale_mode': vs_mode,
        'schema': 'mechanical identity vs SHARD_KEYS/SCHEMA only',
        'src_boundary': 'master3s2 source md5 '
                        '57f9a594cce4f2ec9e3573d8d9d26ca1 (--src default); '
                        'fix3 and other chassis are OUTSIDE this claim',
        'text': 'equivalence claim is limited to (actually-run tags) x '
                '(actually-run value-scale mode) x (schema mechanical '
                'identity), src bounded to master3s2 as above'}
    report['gate_pass'] = bool(ok)
    report['verdict'] = ('PASS' if ok else
                         'FAIL -> DISCARD vec_selfplay2 (bypass artifact; '
                         'no iterative repair loop)')
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({'gate_pass': bool(ok),
                      'gate_md5': report['gate_md5'],
                      'forward': report.get('forward'),
                      'distributions_pass': (report.get('distributions') or {}).get('pass'),
                      'distributions_verdicts': (report.get('distributions') or {}).get('verdicts'),
                      'aux_delta_ks_D': (report.get('distributions') or {}).get('aux_delta_ks_D'),
                      'aux_delta_line': (report.get('distributions') or {}).get('aux_delta_line'),
                      'run_commands': report.get('run_commands'),
                      'claim_scope': report.get('claim_scope'),
                      'schema_problems': [report[k]['problems'] for k in report
                                          if k.startswith('schema') and report[k]['problems']]},
                     indent=1))
    if 'forward' in report:
        fw = report['forward']
        print('(a) 前向层：tags_covered=%s pass=%s'
              % (fw.get('tags_covered'), fw.get('pass')))
        for t in fw.get('tags_covered') or []:
            dd = fw[t]
            print('    %s: max_dlw=%s max_dlb=%s max_dv=%s hard_pass=%s '
                  'report_pass=%s'
                  % (t, dd['max_dlw'], dd['max_dlb'], dd['max_dv'],
                     dd['hard_pass'], dd['report_pass']))
    if 'distributions' in report:
        dist = report['distributions']
        print('(b) 分布层：pass=%s verdicts=%s' % (dist.get('pass'), dist.get('verdicts')))
        print('    aux_delta KS（逐 horizon，判线 %s，暂行、待校准）: %s'
              % (args.ks_line, dist.get('aux_delta_ks_D')))
    for k in sorted(report):
        if k.startswith('schema'):
            print('(c) schema 层 [%s]: files=%s rows=%s problems=%s'
                  % (k, report[k]['files'], report[k]['rows'], report[k]['problems']))
    print('声称范围：%s' % report['claim_scope']['text'])
    print('gate_md5: %s' % report['gate_md5'])
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
