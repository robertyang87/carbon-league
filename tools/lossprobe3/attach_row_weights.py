"""Attach per-row weights to replay shards (lossprobe ②, conversion side).

Weight rule (pinned by Strategy Group, registration 029c3d2b; design
reports/DESIGN_SEARCH_CORRECTION_20261120.md §6.2):
  * SUPPRESSED games (the 11 losing games the classifier did NOT judge
    mistake-lost): policy weight 0.5, value weight 1.0.
  * Everything else (mistake-lost games, winning games, any game not in
    the classification table): policy weight 1.0, value weight 1.0.
  * Correction rows (searched=1) are NOT re-weighted here: train.py's
    pw = searched + (1-searched)*fast_policy_weight already gives them
    effective policy weight 1, and their row weight is 1.0.

Composition (defined by this file; the trainer applies it):
  effective policy weight per row = pw * wpw
  value loss per row is multiplied by wvw, then mean-normalized by wvw.sum()
  (see train2_w.py). Aux heads (delta/fog/win) are never weighted.

Input forms (exactly one):
  * per-game files: --input <dir|file> ... ; the game id of a file is its
    parent directory name when that name is a 32-hex game id, else the
    filename stem when that is a 32-hex game id, else "unknown" (default
    weights 1.0/1.0, counted in the report).
  * merged file: --merged <rows.npz> --provenance <rows_games.json>
    (row ranges per game, as written by the ③ merge step).

Output: the same files under --out-dir (same relative layout), each with
every original key copied array-exact plus two new float32 keys:
  wpw: per-row policy weight, shape (n,)
  wvw: per-row value weight, shape (n,)
A report (attach_report.json) is written next to the outputs. Every output
is reloaded and verified before the tool exits: original keys array-equal
to the input, wpw/wvw exactly the rule's values. Pure numpy, no torch.

This tool never edits in place and never touches train.py / train2.py.
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HEX32 = re.compile(r'^[0-9a-f]{32}$')

DEFAULT_SUPPRESSED_POLICY = 0.5
DEFAULT_SUPPRESSED_VALUE = 1.0


def load_classification(path):
    """game id -> class ('MISTAKE' / 'SUPPRESSED'), from the filed table."""
    with open(path) as f:
        table = json.load(f)
    mapping = {}
    for row in table['rows']:
        mapping[row['game']] = row['class']
    return mapping


def weights_for(game, mapping, sup_pw, sup_vw):
    """(wpw, wvw) for one game id; unknown games default to (1.0, 1.0)."""
    if mapping.get(game) == 'SUPPRESSED':
        return float(sup_pw), float(sup_vw)
    return 1.0, 1.0


def game_id_of(path):
    """Game id from parent dir name, else filename stem; else None."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if HEX32.match(parent):
        return parent
    stem = os.path.splitext(os.path.basename(path))[0]
    if HEX32.match(stem):
        return stem
    return None


def collect_inputs(paths):
    """Expand dirs (recursive *.npz) and files into a sorted file list."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for name in sorted(names):
                    if name.endswith('.npz'):
                        files.append(os.path.join(root, name))
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise SystemExit('input not found: %s' % p)
    if not files:
        raise SystemExit('no .npz inputs found')
    return sorted(files)


def attach_to_arrays(arrays, wpw_val, wvw_val):
    """Return a new dict: original keys copied, plus wpw/wvw.

    Original arrays are passed through untouched (same objects); the two
    weight arrays are fresh float32 vectors of length n.
    """
    n = int(arrays['x'].shape[0])
    out = dict(arrays)
    out['wpw'] = np.full((n,), wpw_val, dtype=np.float32)
    out['wvw'] = np.full((n,), wvw_val, dtype=np.float32)
    return out


def verify_output(path, src_arrays, wpw_val, wvw_val):
    """Reload and check: original keys array-equal, weights exact."""
    with np.load(path) as z:
        got_keys = set(z.files)
        for k in src_arrays:
            if k not in got_keys:
                raise SystemExit('output %s lost key %s' % (path, k))
            if not np.array_equal(z[k], src_arrays[k]):
                raise SystemExit('output %s key %s differs from input'
                                 % (path, k))
        for key, val in (('wpw', wpw_val), ('wvw', wvw_val)):
            if key not in got_keys:
                raise SystemExit('output %s lost key %s' % (path, key))
            arr = z[key]
            if arr.dtype != np.float32 or arr.shape != (int(src_arrays['x'].shape[0]),):
                raise SystemExit('output %s key %s has wrong dtype/shape'
                                 % (path, key))
            if not bool((arr == val).all()):
                raise SystemExit('output %s key %s has wrong values'
                                 % (path, key))


def write_npz(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        np.savez_compressed(f, **arrays)
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def run_per_game(inputs, mapping, out_dir, sup_pw, sup_vw):
    """Per-game mode; returns the report dict."""
    report = {'mode': 'per-game', 'files': [], 'games_seen': [],
              'unknown_games': [], 'rows_total': 0, 'rows_suppressed': 0}
    # Common root for relative layout: longest shared prefix of inputs.
    common = os.path.commonpath([os.path.abspath(p) for p in inputs])
    if os.path.isfile(common):
        common = os.path.dirname(common)
    for src in inputs:
        rel = os.path.relpath(os.path.abspath(src), common)
        dst = os.path.join(out_dir, rel)
        gid = game_id_of(src)
        wpw_val, wvw_val = weights_for(gid, mapping, sup_pw, sup_vw)
        with np.load(src) as z:
            arrays = {k: z[k] for k in z.files}
        n = int(arrays['x'].shape[0])
        out = attach_to_arrays(arrays, wpw_val, wvw_val)
        md5 = write_npz(dst, out)
        verify_output(dst, arrays, wpw_val, wvw_val)
        report['files'].append({'src': src, 'dst': dst, 'game': gid,
                                'rows': n, 'wpw': wpw_val, 'wvw': wvw_val,
                                'md5': md5})
        report['games_seen'].append(gid)
        if gid is None:
            report['unknown_games'].append(src)
        report['rows_total'] += n
        if wpw_val != 1.0 or wvw_val != 1.0:
            report['rows_suppressed'] += n
    return report


def run_merged(merged_path, provenance_path, mapping, out_dir, sup_pw, sup_vw):
    """Merged mode: one npz + row-range provenance; weights by game."""
    with open(provenance_path) as f:
        prov = json.load(f)
    with np.load(merged_path) as z:
        arrays = {k: z[k] for k in z.files}
    n = int(arrays['x'].shape[0])
    wpw = np.ones((n,), dtype=np.float32)
    wvw = np.ones((n,), dtype=np.float32)
    games = []
    for entry in prov['games']:
        gid = entry['game']
        s, cnt = int(entry['start_row']), int(entry['n_rows'])
        wpw_val, wvw_val = weights_for(gid, mapping, sup_pw, sup_vw)
        wpw[s:s + cnt] = wpw_val
        wvw[s:s + cnt] = wvw_val
        games.append({'game': gid, 'rows': cnt, 'wpw': wpw_val,
                      'wvw': wvw_val})
    out = dict(arrays)
    out['wpw'] = wpw
    out['wvw'] = wvw
    dst = os.path.join(out_dir, os.path.basename(merged_path))
    md5 = write_npz(dst, out)
    # Full verify: original keys array-equal + per-row weights exact.
    with np.load(dst) as z:
        for k in arrays:
            if not np.array_equal(z[k], arrays[k]):
                raise SystemExit('merged output key %s differs' % k)
        if not np.array_equal(z['wpw'], wpw) or not np.array_equal(z['wvw'], wvw):
            raise SystemExit('merged output weights differ')
    report = {'mode': 'merged', 'src': merged_path, 'dst': dst,
              'rows_total': n,
              'rows_suppressed': int((wpw != 1.0).sum()),
              'games': games, 'md5': md5}
    return report


def main():
    ap = argparse.ArgumentParser(
        description='Attach per-row policy/value weights to replay shards.')
    ap.add_argument('--input', nargs='+', default=None,
                    help='per-game .npz files or directories (recursive)')
    ap.add_argument('--merged', default=None,
                    help='merged .npz (with --provenance)')
    ap.add_argument('--provenance', default=None,
                    help='row-range provenance JSON for --merged')
    ap.add_argument('--classification', required=True,
                    help='classification table JSON (game -> class)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--suppressed-policy-weight', type=float,
                    default=DEFAULT_SUPPRESSED_POLICY)
    ap.add_argument('--suppressed-value-weight', type=float,
                    default=DEFAULT_SUPPRESSED_VALUE)
    args = ap.parse_args()
    if bool(args.merged) != bool(args.provenance):
        ap.error('--merged and --provenance must be given together')
    if args.merged and args.input:
        ap.error('use either --input (per-game) or --merged, not both')
    if not args.merged and not args.input:
        ap.error('give --input or --merged')

    mapping = load_classification(args.classification)
    if args.merged:
        report = run_merged(args.merged, args.provenance, mapping,
                            args.out_dir, args.suppressed_policy_weight,
                            args.suppressed_value_weight)
    else:
        files = collect_inputs(args.input)
        report = run_per_game(files, mapping, args.out_dir,
                              args.suppressed_policy_weight,
                              args.suppressed_value_weight)
    report_path = os.path.join(args.out_dir, 'attach_report.json')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=1)
    print('rows_total=%d rows_suppressed=%d files=%d'
          % (report['rows_total'], report['rows_suppressed'],
             len(report.get('files', report.get('games', [])))))
    print('report: %s' % report_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
