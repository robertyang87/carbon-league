"""Reconstruct the first submitted joint order on the signed physical seats."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from pydantic_core import from_json

HERE = Path(__file__).resolve().parent
BLEND = HERE.parent
ACTOR = BLEND.parent
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import cnet2_feat as F
from cnet2_build import FIELDS

CACHE = Path('/workspace/sapling-all-top5-20261119/replays')
EXPECTED = {'train': 218, 'val': 10, 'holdout': 18, 'diagnostic_seen_loss': 18}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), sort_keys=True,
                      allow_nan=False).encode()


def sha_arrays(arrays):
    digest = hashlib.sha256()
    for name, item in sorted(arrays.items()):
        a = np.ascontiguousarray(item)
        digest.update(encode([name, a.dtype.str, list(a.shape)]))
        digest.update(a.tobytes())
    return digest.hexdigest()


def signed_cards():
    parent = json.loads((BLEND / 'protocol.json').read_bytes())
    panel = (BLEND / 'panel.json').read_bytes()
    loss_split = (ACTOR / 'data_loss_split.json').read_bytes()
    loss_receipt = (ACTOR / 'data_loss_receipts.json.gz').read_bytes()
    assert sha(panel) == parent['panel_sha256']
    losses = from_json(gzip.decompress(loss_receipt))
    assert sha(loss_receipt) == parent['diagnostic_previously_exposed_losses_receipt']
    assert sha(loss_split) == losses['data_split_sha256']
    cards = json.loads(panel)['games'] + json.loads(loss_split)['games']
    receipts = {r['id']: r for r in losses['games']}
    for row in cards:
        if row['split'] == 'diagnostic_seen_loss':
            receipt = receipts[row['id']]
            assert receipt['gid'] == row['gid'] and receipt['source_sha256'] == row['sha256']
    assert Counter(c['split'] for c in cards) == EXPECTED
    for field in ('id', 'gid', 'terrain_sha256'):
        assert len({c[field] for c in cards}) == len(cards), 'overlap: ' + field
    return cards, {'panel': sha(panel), 'diagnostic_loss_split': sha(loss_split),
                   'diagnostic_loss_receipt': sha(loss_receipt)}


def info_for_base(slot, frame, x, uid, visible, pref, legal, y, after, seat):
    players = frame['players']
    mine, them = players[seat], players[1 - seat]
    cell = mine[1][uid]
    dist = F.DIST[cell]
    geom = {
        'distance_to_other_friendly_bases': sorted(int(dist[v]) for u, v in mine[1].items() if u != uid),
        'distance_to_enemy_bases': sorted(int(dist[v]) for v in them[1].values()),
        'enemy_workers_by_distance_role_cargo': sorted([int(dist[v[0]]), v[2], v[1]] for v in them[2].values()),
        'own_workers_by_distance_role_cargo': sorted([int(dist[v[0]]), v[2], v[1]] for v in mine[2].values()),
        'visible_carbon_within_three': sorted([int(i), int(dist[i]), frame['carbon'][i]]
                                              for i in visible if dist[i] <= 3),
    }
    born = sorted([v[2], u] for u, v in after[2].items() if v[0] == cell and u not in mine[2])
    label = int(y[slot])
    actual_birth = {'RECCOLLECTOR': ['COLLECTOR'], 'RECPLANTER': ['PLANTER'], 'NONE': []}
    assert [v[0] for v in born] == actual_birth[F.BASE_ACTS[label]], 'source recruit/birth disagreement'
    canonical = int(F.CANON[cell]) if seat else cell
    assert int(x['tok_pos'][slot]) == canonical
    return {
        'slot': slot, 'uid': uid, 'physical_cell': cell, 'canonical_cell': canonical,
        'geometry': geom, 'token_occupied': bool(x['tok_mask'][slot]),
        'workers_on_this_base_pre': sorted(u for p in players for u, v in p[2].items() if v[0] == cell),
        'cash_at_decoder_slot': float(pref.cash), 'slots_free_at_decoder_slot': int(pref.slots),
        'demonstrated_label': label, 'demonstrated_action': F.BASE_ACTS[label],
        'legal_labels': np.flatnonzero(legal).tolist(),
        'observed_births_next_frame_for_audit_only': born,
    }


def read_one(card, index):
    packed = (CACHE / (card['id'] + '.json.gz')).read_bytes()
    assert sha(packed) == card['sha256']
    game = from_json(gzip.decompress(packed))
    frames, cfg, seat = game['steps'], game['configuration'], card['seat']
    first, second = frames[:2]
    shared = first[0]['observation']
    assert len(frames) == 1 + card['turns'] and cfg['randomSeed'] == card['seed']
    assert sha(bytes(shared['full_carbon'])) == card['terrain_sha256']
    raw_margin = game['rewards'][seat] - game['rewards'][1 - seat]
    assert abs(raw_margin - card['margin']) < 1.e-6
    assert all(first[s]['status'] == 'ACTIVE' for s in (0, 1)) and shared['step'] == 0
    assert first[seat]['observation']['player'] == seat

    obs = {k: shared[k] for k in ('step', 'players', 'trees')}
    obs.update(player=seat, carbon=first[seat]['observation']['carbon'])
    visible = set()
    for cell, _, kind in obs['players'][seat][2].values():
        visible.update([cell] if kind == 'COLLECTOR' else F.DEST[cell, 1:].tolist())

    feat = F.Feat(seat)
    x = feat.observe(obs)
    recorded = second[seat]['action'] or {}
    expected = {u: v for u, v in recorded.items() if v not in (None, 'NONE')}
    y = feat.labels(recorded)
    assert feat.to_actions(y) == expected
    base_slots = [j for j in (10, 11) if x['tok_mask'][j]]
    assert base_slots == [10, 11]
    obs_keys = [k for k in FIELDS if k in x]
    semantic = {k: np.asarray(x[k], dtype=FIELDS[k][0]) for k in obs_keys}
    row = semantic.copy()
    row.update(uf=np.zeros((F.OWN, F.UF), np.float16),
               mf=np.zeros((F.OWN, 5, F.MF), np.float16),
               legal=np.zeros((F.OWN, 5), bool), y=y,
               carbon=np.zeros(F.N, np.uint8), value=0, teacher=2, gid=card['gid'], step=0)
    assert set(row) == set(FIELDS), 'disable extra FIELDS for this exact signed contract'
    pref, bases, unit_actions = F.Prefix(x, feat.aux), [], []
    for slot in sorted(np.flatnonzero(x['tok_mask'][:F.OWN]).tolist()):
        uid = feat.own_uid[slot]
        uf, mf, legal = pref.feats(slot)
        k = int(y[slot])
        assert 0 <= k < len(legal) and legal[k]
        row['uf'][slot], row['mf'][slot], row['legal'][slot] = uf, mf, legal
        unit_actions.append({'uid': uid, 'slot': slot, 'command': expected.get(uid, 'NONE'),
                             'label': k, 'legal': np.flatnonzero(legal).tolist()})
        if slot in base_slots:
            after = second[0]['observation']['players'][seat]
            base = info_for_base(slot, obs, x, uid, visible, pref, legal, y, after, seat)
            base['prefix_input_sha256'] = sha_arrays({n: row[n][slot] for n in ('uf', 'mf', 'legal')})
            bases.append(base)
        pref.advance(slot, k)
    details = {k: card[k] for k in ('id', 'gid', 'terrain_sha256', 'seat', 'split', 'opponent',
                                     'wdl', 'cohort', 'sha256', 'margin')}
    details.update(row_index=index, step=0, visible_carbon_cells=len(visible),
                   opening_public_input_sha256=sha_arrays(semantic),
                   own_enemy_cash=[obs['players'][s][0] for s in (seat, 1 - seat)],
                   command_frame=1, action_by_uid={u['uid']: u['command'] for u in unit_actions},
                   units_in_decode_order=unit_actions, bases=bases,
                   role_geometry_exchangeable=(bases[0]['geometry'] == bases[1]['geometry']),
                   source_replay_bytes=len(packed))
    return row, details


def summarize(group, train_mode):
    joint = lambda r: '/'.join(b['demonstrated_action'] for b in r['bases'])
    by_opponent = defaultdict(Counter)
    by_state = defaultdict(Counter)
    for r in group:
        by_opponent[r['opponent']][joint(r)] += 1
        by_state[r['opening_public_input_sha256']][joint(r)] += 1
    return {
        'maps': len(group), 'wdl': Counter(r['wdl'] for r in group),
        'seats': Counter(r['seat'] for r in group), 'ordered_joint_counts': Counter(map(joint, group)),
        'opponent_and_joint_counts': dict(by_opponent), 'public_state_and_joint_counts': dict(by_state),
        'geometry_exchangeable': sum(r['role_geometry_exchangeable'] for r in group),
        'visible_cell_histogram': Counter(r['visible_carbon_cells'] for r in group),
        'different_from_training_modal_order': [r['id'] for r in group if joint(r) != train_mode],
        'visible_worker_on_base_count': sum(bool(b['workers_on_this_base_pre']) for r in group for b in r['bases']),
    }


def main():
    cards, sources = signed_cards()
    records = []
    arrays = {k: np.zeros((len(cards), *shape), dtype) for k, (dtype, shape) in FIELDS.items()}
    for idx, card in enumerate(cards):
        row, details = read_one(card, idx)
        for name, values in arrays.items():
            values[idx] = row[name]
        records.append(details)
        if (idx + 1) % 44 == 0:
            print('validated', idx + 1, 'of', len(cards), flush=True)
    assert not arrays['carbon'].any() and not arrays['value'].any() and not arrays['step'].any()
    assert all(np.isfinite(a).all() for a in arrays.values())
    assert np.all(arrays['tok_mask'][:, 10:12])
    target = HERE / 'roots.npz'
    tmp = target.with_suffix('.partial')
    with tmp.open('wb') as out:
        np.savez_compressed(out, **arrays)
    with np.load(tmp, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays) and all(np.array_equal(a, loaded[k]) for k,a in arrays.items())
    os.replace(tmp, target)
    trained = [r for r in records if r['split'] == 'train']
    train_orders = Counter('/'.join(b['demonstrated_action'] for b in r['bases']) for r in trained)
    train_mode = sorted(train_orders, key=lambda k: (-train_orders[k], k))[0]
    summaries = {n: summarize([r for r in records if r['split'] == n], train_mode) for n in EXPECTED}
    rep = {
        'games': records, 'split_summaries': summaries, 'source_sha256': sources,
        'role': 'Physical-seat Master6 first submitted orders. Each base uses the demonstrated preceding slots. Recruited units at frame 1 are retrospective assertions only.',
        'input_exclusions': 'Opposing submitted commands, full cell carbon, future worker births, rewards and map IDs are absent from the network predictors. gid indexes rows only. carbon and value are zero. Opponent players from kickoff are public. Visibility follows physical seat worker sensor rules. The full board is used only to check the frozen terrain digest.',
        'role_symmetry': 'Relative-role equality compares distance multisets to friend/enemy bases, worker distances/role/cargo and observed carbon near each cell. Absolute/canonical token positions can still distinguish bases. Seat 1 canonicalization reflects the horizontal axis coordinate across columns; it does not exchange physical IDs.',
        'train_modal_order': train_mode,
        'train_opponents': sorted({r['opponent'] for r in trained}),
        'additional_diagnostic': 'The diagnostic_seen_loss panel contains previously inspected losses and is ineligible for blind or training claims.',
        'archive': {'path': target.name, 'rows': len(cards), 'sha256': sha(target.read_bytes()),
                    'fields': {k: {'dtype': a.dtype.str, 'shape': list(a.shape), 'sha256': sha(a.tobytes())}
                               for k, a in arrays.items()}},
        'rebuild': 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 taskset -c 2 python ' + str(Path(__file__).resolve().relative_to(ROOT)),
        'code_sha256': {str(f.relative_to(ROOT)): sha(f.read_bytes()) for f in
                        (Path(__file__).resolve(), ROOT/'code/cnet2_feat.py', ROOT/'code/cnet2_build.py', ACTOR/'data_build.py')},
        'signed_opening_protocol_sha256': sha((HERE/'protocol.json').read_bytes()),
    }
    target = HERE/'panel.json.gz'
    tmp = target.with_suffix('.partial')
    tmp.write_bytes(gzip.compress(encode(rep), compresslevel=9, mtime=0))
    assert len(from_json(gzip.decompress(tmp.read_bytes()))['games']) == len(cards)
    os.replace(tmp, target)
    print('panel', sha(target.read_bytes()), 'roots', rep['archive']['sha256'], flush=True)


if __name__ == '__main__':
    main()
