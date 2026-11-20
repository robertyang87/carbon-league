#!/usr/bin/env python3
"""LP3 game-level classification + issuance (phase-2 finisher).

Reads a completed run outdir (per-game positions.jsonl +
rows_correction_candidates.npz + run_manifest.json), aggregates each
game's mistake mass at the pinned step threshold, classifies the game,
and materializes the final correction rows:

  * over-threshold step: searched, search disagrees with the played move,
    and adv_u > STEP_TH (u scale) -- one identifiable own mistake.
  * mistake mass: N_over = count of over-threshold steps;
    S_over = sum of adv_u over those steps; max_adv = largest single-step
    adv_u in the game.
  * class: MISTAKE (错输局) iff N_over >= N_LINE and S_over >= S_LINE,
    else SUPPRESSED (被压制局).
  * issuance: correction rows are issued ONLY for MISTAKE games (their
    candidates file becomes rows_correction.npz). SUPPRESSED games'
    candidates are quarantined as rows_correction_suppressed.npz
    (audit only, never for training; those games take the (b)
    downweight route).
  * disposition per game (no blanks): MISTAKE -> route (d), corrected
    step count = N_over, uncorrected steps keep ORIGINAL weight;
    SUPPRESSED -> route (b), corrected steps 0, downweight tier as
    pinned at the Gen-5 approval (proposed: policy weight x0.5, value
    weight unchanged).

The classification rule and thresholds are recipe once pinned in the
design (DESIGN_LOSSPROBE3): any change is a recipe change requiring
re-approval. This tool refuses to run if its threshold arguments do not
match the run manifest.
"""
import argparse
import csv
import json
import os
import shutil

import numpy as np

CKPT_MD5 = '708f8cca57f5674d00d1e5e56758e14c'


def classify_game(recs, step_th, n_line, s_line):
    over = [r for r in recs
            if r.get('searched') and not r.get('agree')
            and r.get('adv_u') is not None and r['adv_u'] > step_th]
    n_over = len(over)
    s_over = float(sum(r['adv_u'] for r in over))
    advs = [r['adv_u'] for r in recs if r.get('adv_u') is not None]
    max_adv = float(max(advs)) if advs else 0.0
    cls = 'MISTAKE' if (n_over >= n_line and s_over >= s_line) else 'SUPPRESSED'
    return {'n_over': n_over, 's_over': s_over, 'max_adv': max_adv,
            'class': cls, 'over_steps': [r['step'] for r in over]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--step-th', type=float, required=True)
    ap.add_argument('--n-line', type=int, required=True)
    ap.add_argument('--s-line', type=float, required=True)
    ap.add_argument('--downweight-tier', default='policy x0.5 (proposed; '
                    'final tier pinned at Gen-5 approval)')
    args = ap.parse_args()

    with open(os.path.join(args.rundir, 'run_manifest.json')) as f:
        manifest = json.load(f)
    assert abs(manifest['threshold_u'] - args.step_th) < 1e-12, \
        'step threshold mismatch vs run manifest -- thresholds are recipe'
    pack_md5 = manifest['pack_md5']

    rows_out = []
    for corpus in sorted(os.listdir(args.rundir)):
        cdir = os.path.join(args.rundir, corpus)
        if not os.path.isdir(cdir):
            continue
        for game_id in sorted(os.listdir(cdir)):
            gdir = os.path.join(cdir, game_id)
            if not os.path.isdir(gdir):
                continue
            with open(os.path.join(gdir, 'positions.jsonl')) as f:
                recs = [json.loads(ln) for ln in f]
            c = classify_game(recs, args.step_th, args.n_line, args.s_line)
            cand = os.path.join(gdir, 'rows_correction_candidates.npz')
            n_cand = 0
            if os.path.exists(cand):
                n_cand = len(np.load(cand)['x'])
            assert n_cand == c['n_over'], \
                f'{game_id}: candidate rows {n_cand} != over-threshold steps {c["n_over"]}'
            if c['class'] == 'MISTAKE':
                if n_cand:
                    shutil.copyfile(
                        cand, os.path.join(gdir, 'rows_correction.npz'))
                disposition = '(d) correction'
                corrected = n_cand
                rest = 'original weight'
            else:
                if n_cand:
                    shutil.copyfile(
                        cand, os.path.join(gdir,
                                           'rows_correction_suppressed.npz'))
                disposition = '(b) downweight'
                corrected = 0
                rest = args.downweight_tier
            rows_out.append({
                'game': game_id, 'corpus': corpus, 'class': c['class'],
                'n_over': c['n_over'], 's_over': round(c['s_over'], 4),
                'max_adv': round(c['max_adv'], 4),
                'disposition': disposition, 'corrected_steps': corrected,
                'uncorrected_disposition': rest,
                'ckpt_md5': CKPT_MD5, 'pack_md5': pack_md5})
            print(f"{corpus} {game_id[:8]} {c['class']:10s} "
                  f"N={c['n_over']} S={c['s_over']:.3f} max={c['max_adv']:.3f} "
                  f"corrected={corrected}")
    rows_out.sort(key=lambda r: (r['corpus'], r['game']))
    with open(os.path.join(args.rundir, 'classification_table.json'), 'w') as f:
        json.dump(rows_out, f, indent=1)
    with open(os.path.join(args.rundir, 'classification_table.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    n_m = sum(1 for r in rows_out if r['class'] == 'MISTAKE')
    print(f'games {len(rows_out)}: MISTAKE {n_m}, SUPPRESSED '
          f'{len(rows_out) - n_m}; total corrected steps '
          f'{sum(r["corrected_steps"] for r in rows_out)}')


if __name__ == '__main__':
    main()
