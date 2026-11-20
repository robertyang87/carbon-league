"""Teach H1 pruning with the same native search continued for every exposed candidate."""
import argparse
import copy
import gzip
import json
import os
import time
import types
import zipfile

import numpy as np
from register import BASE, HERE, PARENT, PARENT_SHA, sha


def instrument():
    src = PARENT.read_text()
    def change(src, before, after):
        if src.count(before) != 1:
            raise ValueError(before)
        return src.replace(before, after)
    for line in ('z = _torch.cat([z.mean(2), z.amax(2), s], 1)',
                 'z = _torch.cat([z.mean(0), z.amax(0), s], 1)'):
        src = change(src, line, line+'\n                self.study_z = z.numpy().copy()')
    marker = 'v.extend(float(x) for x in vv); passes += 1'
    src = change(src, marker, marker + '''
            if t == 1:
                for board, features in zip(part, net.study_z[::w]):
                    _STUDY['features'][board[0]] = features.tolist()''')
    depth_line = '        depth = t\n        if t == D2 or not n:'
    src = change(src, depth_line, '''        depth = t
        if t in (1, 4, 8):
            _STUDY[str(t)] = [float(val[c, 0]) for c in range(len(cands))]
        if t == D2 or not n:''')
    src = change(src, '    depth = 0; cut = True; rolls = M',
                 "    _STUDY['menu'] = [dict(a) for a in cands]\n    depth = 0; cut = True; rolls = M")
    marker = 'r1 = np.random.RandomState(seed0 + 7919 * t + 31 * m + 1); r2 = np.random.RandomState(seed0 + 7919 * t + 31 * m + 2)'
    salt = marker.replace('seed0 +', 'seed0 + _LATER_SALT +')
    src = change(src, marker, salt)
    model = types.ModuleType('offline_native_root')
    exec(compile(src, str(PARENT), 'exec'), model.__dict__)
    assert model._SEARCH_OK and model._net() is model._RNET and model._S_D2 == 8
    model._S_K = 8
    model._S_STAGES = ()
    model._S_ABORT = 1e6
    return model


def collect(card, protocol, cfg, model, salt):
    archive_path = BASE/card['archive']
    assert sha(archive_path.read_bytes()) == card['sha256']
    with zipfile.ZipFile(archive_path) as z:
        obs_rows = gzip.decompress(z.read(card['role'] + '/observations.jsonl.gz')).splitlines()
        turns = z.read(card['role'] + '/turns.jsonl').splitlines()
        receipt = json.loads(z.read(card['role'] + '/receipt.json'))
    seat = card['seat']
    assert len(obs_rows) == len(turns) == receipt['official_transitions'] == 299
    tracker = model.BeliefTracker(model._Bel, seat)
    prev = None
    rows = []
    for t, (line, cmd) in enumerate(zip(obs_rows, turns)):
        obs = json.loads(line)['observations'][seat]
        issued = json.loads(cmd)['calls'][seat]['response']['action']
        assert t == obs['step'] and obs['player'] == seat and 'full_carbon' not in obs
        if t in protocol['steps']:
            seed = (card['seed'] + t * 1009 + 17 * seat) % (2**31)
            model._ST = dict(tr=copy.deepcopy(tracker), prev=prev, rng=np.random.RandomState(seed))
            model._STUDY = dict(features={})
            model._LATER_SALT = salt
            model._S_COST[:] = [0.0]*4
            model._act(obs, cfg, time.perf_counter())
            study = model._STUDY
            if 'menu' in study:
                n = len(study['menu'])
                assert set(study['features']) == set(range(n))
                assert all(len(study[str(d)]) == n for d in (1, 4, 8))
                study['features'] = [study['features'][i] for i in range(n)]
            rows.append(dict(seed=card['seed'], seat=seat, step=t, later_salt=salt, **study))
        tracker.step(obs['players'], t, obs['carbon'])
        tracker.set_actions(obs['players'], issued)
        prev = obs['players']
    return rows


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('split', choices=['train', 'held'])
    args.add_argument('--salt', type=int, default=0)
    args = args.parse_args()
    if args.split == 'held' and not (HERE/'weights.json').is_file():
        raise SystemExit('Fit and freeze training data before held scoring')
    os.environ['CARBON_DEBUG'] = '1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert sha(PARENT.read_bytes()) == PARENT_SHA
    protocol = json.loads((HERE/'protocol.json').read_bytes())
    cfg_tape = BASE.parent/'sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/direct_original/games/candidate/raw_621758346_1.json.gz'
    cfg = json.loads(gzip.decompress(cfg_tape.read_bytes()))['game_configuration']
    model = instrument()
    out = HERE/f'{args.split}_{args.salt}'
    out.mkdir(exist_ok=True)
    for card in protocol['sources']:
        if card['split'] != args.split:
            continue
        path = out/f"{card['seed']}.json.gz"
        if path.exists():
            continue
        rows = collect(card, protocol, cfg, model, args.salt)
        tmp = path.with_suffix('.partial')
        tmp.write_bytes(gzip.compress(json.dumps(rows, allow_nan=False).encode(), mtime=0))
        tmp.replace(path)
        print(card['seed'], [len(r.get('menu',[])) for r in rows], flush=True)
