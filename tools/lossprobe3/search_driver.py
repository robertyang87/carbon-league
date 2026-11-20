#!/usr/bin/env python3
"""LP3 search driver: per-position fixed-budget re-search of losing games.

For each game (Echo9/Echo10 ladder loss with a corpus replay):
  * take the state sequence from the recorded observations (row t =
    chassis step t: pre-state steps[t][0].observation, whose 'step' field
    equals t; played action steps[t+1][seat]['action']; terminal state
    steps[299]) and rebuild the conversion row for every Echo decision
    position with the exact field semantics of the production conversion
    (vec_selfplay schema:
    x = encode(players, t, me, TRUE carbon (full_carbon), known=all-ones,
    prev) quantized
    rint(clip(x,0,2.55)*100) uint8; sc = _scalars; tw/tb one-hot of the
    played joint action over current workers/bases; margin/win from the
    final cash margin; aux_delta = cash-margin change at +15/+40/+80 steps
    clamped at the end; carbon = true map float16; searched = 0);
  * (pilot/run modes) at every position run the instrumented fixed-budget
    pack's agent() on the reconstructed state with the actual played action
    injected as a candidate, and read the same-search values of the search's
    chosen action and of the actual action on both scales (bridged
    _terminal scale `val`, pre-bridge u scale `valu`);
  * log one JSON line per position; emit base rows and (run mode) correction
    rows (base row with tw/tb replaced by the search action's one-hots and
    searched=1) for positions whose u-scale advantage exceeds the pinned
    threshold.

Modes: fidelity (reconstruction only, compare vs reference row streams),
pilot (search, no correction rows), run (search + correction rows).
Determinism: fresh module instance per game, module _SEED pinned (default
999), CPU forced (CUDA_VISIBLE_DEVICES=''), CARBON_DEBUG=1 (search errors
raise instead of silently degrading).
"""
import argparse
import gzip
import json
import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('CARBON_DEBUG', '1')

import numpy as np

AUX_HORIZONS = (15, 40, 80)
STORE_DTYPES = {'x': np.uint8, 'sc': np.float32, 'tw': np.float16,
                'tb': np.float16, 'wmask': np.uint8, 'bmask': np.uint8,
                'carbon': np.float16, 'margin': np.float32,
                'aux_delta': np.float32, 'win': np.float32,
                'searched': np.uint8}
ROW_KEYS = list(STORE_DTYPES.keys())
N_CELLS = 225


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_obs(state, me):
    return {'step': state.step, 'player': me, 'players': state.players,
            'carbon': list(state.carbon), 'trees': state.trees_meta,
            'remainingOverageTime': 0}


def onehot_targets(mod, players, me, action):
    """tw/tb/wmask/bmask for a joint action dict at a position."""
    workers, _trees, bases, _cash = mod.parse(players)
    tw = np.zeros((N_CELLS, 5), dtype=np.float32)
    tb = np.zeros((N_CELLS, 3), dtype=np.float32)
    wmask = np.zeros(N_CELLS, dtype=np.uint8)
    bmask = np.zeros(N_CELLS, dtype=np.uint8)
    action = action or {}
    for wid, w in workers.items():
        if w[3] != me:
            continue
        pos = w[0]
        wmask[pos] = 1
        tw[pos, mod.WACT.index(action.get(wid, 'NONE'))] = 1.0
    for bid, pos in bases[me]:
        bmask[pos] = 1
        tb[pos, mod.BACT.index(action.get(bid, 'NONE'))] = 1.0
    return tw, tb, wmask, bmask


def base_row(mod, obs, me, prev_players, action):
    players = obs['players']
    step = obs['step']
    x = mod.encode(players, step, me, obs['full_carbon'],
                   [1.0] * N_CELLS, prev_players)
    x = np.rint(np.clip(x, 0, 2.55) * 100).astype(np.uint8)
    sc = np.asarray(mod._scalars(players, step, me), dtype=np.float32)
    # Converter caliber (measured, fidelity sample 2026-11-20): the
    # production conversion's sc carries the RAW cargo scalar -- the
    # chassis _scalars endgame cargo adjustment (moves_left <= 18 branch)
    # is NOT applied in conversion rows. Only component 1 is affected;
    # all other components are the chassis' own. (The search internally
    # keeps using the pack's _scalars, adjustment included.)
    workers, _trees, _bases, _cash = mod.parse(players)
    cargo = [0.0, 0.0]
    for w in workers.values():
        cargo[w[3]] += w[1]
    sc[1] = np.float32((cargo[me] - cargo[1 - me]) / 3000.0)
    tw, tb, wmask, bmask = onehot_targets(mod, players, me, action)
    return {'x': x, 'sc': sc, 'tw': tw.astype(np.float16),
            'tb': tb.astype(np.float16), 'wmask': wmask, 'bmask': bmask,
            'carbon': np.array(obs['full_carbon'], dtype=np.float16),
            'searched': np.uint8(0)}


def finish_rows(rows, traj):
    """Fill margin/win/aux_delta from the cash-margin trajectory."""
    final = traj[-1]
    win = 1.0 if final > 0 else (0.5 if final == 0 else 0.0)
    for t, rec in enumerate(rows):
        rec['margin'] = np.float32(final)
        rec['win'] = np.float32(win)
        rec['aux_delta'] = np.array(
            [traj[min(t + h, len(traj) - 1)] - traj[t] for h in AUX_HORIZONS],
            dtype=np.float32)
    return rows


def write_shard(path, rows):
    arrs = {k: np.stack([r[k] for r in rows]).astype(STORE_DTYPES[k])
            for k in ROW_KEYS}
    with open(path, 'wb') as f:
        np.savez_compressed(f, **arrs)


def decide(out, search_action, actual_action, threshold_u):
    """Pure issuance decision from one position's search export."""
    if out is None:
        return {'searched': False, 'agree': search_action == actual_action,
                'adv_u': None, 'adv_term': None, 'issued': False,
                'v_u_search': None, 'v_u_actual': None,
                'v_term_search': None, 'v_term_actual': None}
    best = out['best']
    ai = out['actual_idx']
    valu, val = out['valu'], out['val']
    v_u_s = valu.get((best, 0))
    v_u_a = valu.get((ai, 0)) if ai >= 0 else None
    v_t_s = val.get((best, 0))
    v_t_a = val.get((ai, 0)) if ai >= 0 else None
    adv_u = (v_u_s - v_u_a) if (v_u_s is not None and v_u_a is not None) else None
    adv_t = (v_t_s - v_t_a) if (v_t_s is not None and v_t_a is not None) else None
    agree = (search_action == actual_action)
    issued = bool((not agree) and adv_u is not None and adv_u > threshold_u)
    return {'searched': True, 'agree': agree, 'adv_u': adv_u,
            'adv_term': adv_t, 'issued': issued,
            'v_u_search': v_u_s, 'v_u_actual': v_u_a,
            'v_term_search': v_t_s, 'v_term_actual': v_t_a}


def load_replay(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def process_game(mod, game, do_search, threshold_u, max_positions=None,
                 obs_carbon='fog'):
    """Returns (position_log, base_rows, correction_rows, meta).

    State sequence comes from the recorded observations themselves:
    row t (chassis step t, 0..298) has pre-state steps[t][0].observation
    (the full observation entry; its 'step' field equals t) and played
    action steps[t+1][seat]['action']; the terminal state is
    steps[299][0].observation. Row features use the TRUE carbon map
    (observation['full_carbon'], conversion caliber); the search is fed
    an observation in live shape whose 'carbon' is the Echo seat's own
    recorded view (obs_carbon='fog', the live input) or the true map
    (obs_carbon='true', the offline-harness convention).
    """
    rep = load_replay(game['replay'])
    me = game['echo_seat']
    cfg = dict(mod.DEFAULT_CONFIG)
    cfg.update(rep.get('configuration') or {})
    steps = rep['steps']

    def full_obs(t):
        return steps[t][0]['observation']

    def cash_margin(t):
        pl = full_obs(t)['players']
        return float(pl[me][0]) - float(pl[1 - me][0])

    log = []
    rows = []
    corr = []
    traj = [cash_margin(t) for t in range(299)] + [cash_margin(299)]
    n_steps = 299
    if max_positions:
        n_steps = min(n_steps, max_positions)
    for t in range(n_steps):
        o = full_obs(t)
        players = o['players']
        prev_players = full_obs(t - 1)['players'] if t > 0 else None
        actual = steps[t + 1][me]['action'] or {}
        row = base_row(mod, o, me, prev_players, actual)
        rec = {'game': game['game_id'], 'corpus': game['corpus'], 'step': t,
               'seat': me, 'actual_action': actual}
        if do_search:
            fog = steps[t][me]['observation']['carbon']
            sobs = {'step': t, 'player': me, 'players': players,
                    'carbon': list(o['full_carbon']) if obs_carbon == 'true'
                    else list(fog),
                    'trees': o['trees'], 'remainingOverageTime': 0}
            mod._LP3['actual'] = dict(actual)
            mod._LP3['out'] = None
            mod._LP3['actual_idx'] = -1
            t0 = time.perf_counter()
            search_action = mod.agent(sobs, cfg)
            rec['sec'] = time.perf_counter() - t0
            out = mod._LP3['out']
            d = decide(out, search_action, actual, threshold_u)
            rec.update(d)
            rec['search_action'] = search_action
            if out is not None:
                rec.update({'depth': out['depth'], 'rolls': out['rolls'],
                            'n_cands': out['n_cands'], 'best_idx': out['best'],
                            'actual_idx': out['actual_idx'],
                            'actual_in_alive': out['actual_idx'] in out['alive']})
            if d['issued']:
                tw, tb, _wm, _bm = onehot_targets(mod, players, me,
                                                  search_action)
                crow = dict(row)
                crow['tw'] = tw.astype(np.float16)
                crow['tb'] = tb.astype(np.float16)
                crow['searched'] = np.uint8(1)
                corr.append((t, crow))
            # tracker hygiene: the agent recorded its SEARCH action; overwrite
            # with the action actually played so beliefs track the real game.
            mod._ST['tr'].set_actions(players, actual)
        rows.append(row)
        log.append(rec)
    rows = finish_rows(rows, traj)
    corr_rows = []
    for t, crow in corr:
        crow['margin'] = rows[t]['margin']
        crow['win'] = rows[t]['win']
        crow['aux_delta'] = rows[t]['aux_delta']
        corr_rows.append(crow)
    meta = {'game': game['game_id'], 'corpus': game['corpus'], 'seat': me,
            'final_margin': traj[-1], 'n_positions': len(log),
            'n_issued': len(corr_rows)}
    return log, rows, corr_rows, meta


# ---------------- fidelity reference streams ----------------

def segment_stream(ref_dir):
    """Concatenate a conversion rows dir and split into 299-row games."""
    import glob
    files = sorted(glob.glob(os.path.join(ref_dir, 'shard_*.npz')))
    assert files, f'no shards in {ref_dir}'
    parts = {k: [] for k in ROW_KEYS}
    for f in files:
        z = np.load(f)
        for k in ROW_KEYS:
            parts[k].append(z[k])
    arrs = {k: np.concatenate(v) for k, v in parts.items()}
    margin = arrs['margin']
    bounds = [0]
    for i in range(1, len(margin)):
        if margin[i] != margin[i - 1]:
            bounds.append(i)
    bounds.append(len(margin))
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        segs.append((a, b, float(margin[a])))
    return arrs, segs


def compare_game(rows, arrs, seg):
    a, b, _m = seg
    out = {}
    for k in ROW_KEYS:
        mine = np.stack([r[k] for r in rows]).astype(STORE_DTYPES[k])
        ref = arrs[k][a:b]
        if mine.shape != ref.shape:
            out[k] = f'SHAPE mine {mine.shape} ref {ref.shape}'
            continue
        if k in ('x', 'wmask', 'bmask', 'searched'):
            out[k] = int((mine != ref).sum())
        else:
            out[k] = int((mine.view(np.uint8) != ref.view(np.uint8)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fidelity', 'pilot', 'run'], required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--games', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--only', default=None)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--threshold-u', type=float, default=0.05)
    ap.add_argument('--ref-dirs', default=None,
                    help='echo9=<dir>,echo10=<dir> (fidelity mode)')
    ap.add_argument('--max-positions', type=int, default=None)
    ap.add_argument('--obs-carbon', choices=['fog', 'true'], default='fog')
    args = ap.parse_args()

    with open(args.games) as f:
        games = json.load(f)
    if args.only:
        keep = set(args.only.split(','))
        games = [g for g in games if g['game_id'] in keep]
    os.makedirs(args.outdir, exist_ok=True)
    ref = {}
    if args.mode == 'fidelity':
        assert args.ref_dirs, 'fidelity mode needs --ref-dirs'
        for kv in args.ref_dirs.split(','):
            tag, d = kv.split('=', 1)
            ref[tag] = segment_stream(d)
    do_search = args.mode in ('pilot', 'run')
    done = []
    for gi, game in enumerate(games):
        mod = load_module(args.pack, f'lp3_pack_{gi}')
        mod._SEED = args.seed
        gdir = os.path.join(args.outdir, game['corpus'], game['game_id'])
        os.makedirs(gdir, exist_ok=True)
        log, rows, corr_rows, meta = process_game(
            mod, game, do_search, args.threshold_u, args.max_positions,
            args.obs_carbon)
        with open(os.path.join(gdir, 'positions.jsonl'), 'w') as f:
            for rec in log:
                f.write(json.dumps(rec) + '\n')
        write_shard(os.path.join(gdir, 'rows_base.npz'), rows)
        if args.mode == 'run':
            # Step-level candidates only; the game-level classification
            # (classify_issue.py) decides which games' candidates become
            # correction rows. Suppressed games' candidates are quarantined.
            write_shard(os.path.join(gdir, 'rows_correction_candidates.npz'),
                        corr_rows)
        if args.mode == 'fidelity':
            arrs, segs = ref[game['corpus']]
            cand = [s for s in segs
                    if abs(s[2] - game['margin']) < 0.02 and s[1] - s[0] == len(rows)]
            assert len(cand) == 1, f'{game["game_id"]}: {len(cand)} ref segments'
            meta['fidelity'] = compare_game(rows, arrs, cand[0])
            write_shard(os.path.join(gdir, 'rows_ref.npz'),
                        [{k: arrs[k][cand[0][0]:cand[0][1]][i] for k in ROW_KEYS}
                         for i in range(len(rows))])
        with open(os.path.join(gdir, 'summary.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        done.append(meta)
        with open(os.path.join(args.outdir, 'progress.json'), 'w') as f:
            json.dump({'done': len(done), 'total': len(games),
                       'games': [m['game'] for m in done]}, f)
        print(f"[{len(done)}/{len(games)}] {game['corpus']} {game['game_id']} "
              f"margin {meta['final_margin']:.2f} issued {meta['n_issued']}",
              flush=True)
    if args.mode == 'run':
        import hashlib
        with open(args.pack, 'rb') as f:
            pack_md5 = hashlib.md5(f.read()).hexdigest()
        with open(os.path.join(args.outdir, 'run_manifest.json'), 'w') as f:
            json.dump({'seed': args.seed, 'threshold_u': args.threshold_u,
                       'obs_carbon': args.obs_carbon, 'pack': args.pack,
                       'pack_md5': pack_md5, 'games': len(games)}, f, indent=1)
    print('ALL DONE')


if __name__ == '__main__':
    main()
