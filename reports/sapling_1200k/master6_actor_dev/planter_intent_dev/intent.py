"""Learn the terminal value of an order including cases where the order fails to execute."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
TREE = HERE.parent / 'tree_response_training'
sys.path.insert(0, str(TREE / 'response_head'))
import policy

SPEC = json.loads((HERE / 'protocol.json').read_bytes())
NAMES = [*policy.FEATURE_NAMES, 'collector_contact_stay_minus_move',
         'planter_contact_stay_minus_move', 'standing_on_enemy_tree']


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(name, value):
    path = HERE / name
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with tempfile.NamedTemporaryFile(dir=HERE, delete=False) as f:
        tmp = Path(f.name)
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ValueError(f'previously sealed {name} differs')
    finally:
        tmp.unlink()
    return path


def public_features(root, uid, defer):
    obs = root['public_at_root']
    cfg = SPEC['configuration']
    coord = obs['players'][obs['player']][2][uid][0]
    other = obs['players'][1 - obs['player']]
    dst = policy._cell(coord, defer, cfg['size'])

    def reachable(cell, kind):
        return any(k == kind and policy._dist(cell, p, cfg['size']) <= 1
                   for p, _, k in other[2].values())

    contact = [int(reachable(coord, kind)) - int(reachable(dst, kind))
               for kind in ('COLLECTOR', 'PLANTER')]
    existing = any(p == coord and age < cfg['treeLifespan'] for p, age in other[3].values())
    x = [*policy.features(obs, cfg, uid, 'NONE', defer, root['v1']), *contact, int(existing)]
    if not np.isfinite(x).all():
        raise ValueError('nonfinite public features')
    return x


def opportunities(area):
    selection = load(area / 'selection.json')
    plan = load(area / 'protocol.json')
    if sha(area / 'protocol.json') != selection['protocol_sha256'] or plan['parent_pack_sha256'] != SPEC['parent']:
        raise ValueError('incompatible source policy or selection')
    sources = load(area / 'source_receipt.json')
    if sha(area / 'source_receipt.json') != selection['source_receipt_sha256']:
        raise ValueError('wrong source')
    boards = {(g['seed'], g['seat']): g['map_sha256'] for g in sources['games']}
    result = {}
    for reg in selection['cases']:
        prefix = area / reg['case_directory'] / 'root.json.gz'
        root = load(prefix)
        key = sha(prefix)
        pay = reg['old'] == 'NONE'
        defer = reg['new'] if pay else reg['old']
        obs = root['public_at_root']
        if (key != reg['root_sha256'] or key in result or root['old'] != reg['old'] or
                root['new'] != reg['new'] or (reg['paid_arm'] == 'parent') != pay or
                (obs['player'], obs['step']) != (reg['seat'], reg['step'])):
            raise ValueError('registered intervention or identity differs')
        result[key] = dict(root=key, board=boards[reg['seed'], reg['seat']], seed=reg['seed'],
                           seat=reg['seat'], step=reg['step'], uid=reg['uid'], parent_pay=pay,
                           physical_defer=defer, x=public_features(root, reg['uid'], defer))
    return result


def attach_terminal(area, rows):
    labels = load(area / 'training_labels.json')
    if labels['selection_sha256'] != sha(area / 'selection.json'):
        raise ValueError('outcomes from different selection')
    pairs = labels['pairs']
    if len(rows) != len(pairs) or set(rows) != {p['root_sha256'] for p in pairs}:
        raise ValueError('missing or repeated treatment result')
    for p in pairs:
        r = rows[p['root_sha256']]
        y = p['pay']['official_margin'] - p['defer']['official_margin']
        if not np.isfinite(y) or abs(y - p['target_delta_official_margin']) > 1e-6 or r['parent_pay'] != p['pay_is_v1']:
            raise ValueError('official margin orientation')
        r.update(y=y, pay_realized=p['pay_intent_filled'])


def fit(rows):
    map_counts = Counter(r['board'] for r in rows)
    w = np.asarray([1 / (len(map_counts) * map_counts[r['board']]) for r in rows])
    x = np.asarray([r['x'] for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows])
    design = np.column_stack([np.ones(len(x)), x])
    h = SPEC['weights']
    reg = np.diag([h['intercept_penalty']] + [h['slope_penalty']] * len(NAMES))
    beta = np.linalg.solve(design.T @ (design * w[:, None]) + reg, design.T @ (w * y))
    return dict(beta=beta.tolist(), constant=float(w @ y / (1 + h['intercept_penalty'])),
                low=x.min(axis=0).tolist(), high=x.max(axis=0).tolist(), spec=sha(HERE/'protocol.json'),
                features=NAMES, training_boards=sorted(map_counts), roots=len(rows),
                paid=sum(r['pay_realized'] for r in rows))


def choose(row, model, constant=False):
    x = np.asarray(row['x'])
    score = model['constant'] if constant else model['beta'][0] + x @ model['beta'][1:]
    h = SPEC['weights']
    within = np.all(x >= np.asarray(model['low']) - h['support_pad']) and np.all(x <= np.asarray(model['high']) + h['support_pad'])
    parent = row['parent_pay']
    delta = -score if parent else score
    return bool(not parent if within and delta >= h['minimum_predicted_departure'] else parent), float(score)


def decide(rows, model):
    for r in rows:
        if r['board'] in model['training_boards']:
            raise ValueError('decision on seen terrain')
        for name in ('model', 'constant'):
            r[name+'_pay'], r[name+'_forecast'] = choose(r, model, name == 'constant')


def summarize(rows):
    out = []
    for r in rows:
        for name in ('model','constant'):
            r[name+'_gain'] = (int(r[name+'_pay']) - int(r['parent_pay'])) * r['y']
    for b in sorted({r['board'] for r in rows}):
        sub = [r for r in rows if r['board'] == b]
        out.append(dict(board=b, seed=sub[0]['seed'], n=len(sub),
                        changes=sum(r['model_pay'] != r['parent_pay'] for r in sub),
                        mean_model_gain=float(np.mean([r['model_gain'] for r in sub])),
                        mean_constant_gain=float(np.mean([r['constant_gain'] for r in sub]))))
    return dict(terrains=out, equal_map_mean={name:float(np.mean([m['mean_'+name+'_gain'] for m in out]))
                                             for name in ('model','constant')}, roots=rows)


def train():
    rows = []
    for item in SPEC['training_batches']:
        area = TREE / item['dir']
        if sha(area/'training_labels.json') != item['labels_sha256']:
            raise ValueError('unregistered training outcomes')
        group = opportunities(area)
        attach_terminal(area, group)
        rows.extend(group.values())
    model = fit(rows)
    if len(model['training_boards']) != 14 or len(rows) != 56 or model['paid'] != 29:
        raise ValueError('full training cohort differs')
    seal('weights.json', model)
    folds = []
    for board in model['training_boards']:
        group = [r for r in rows if r['board'] == board]
        decide(group, fit([r for r in rows if r['board'] != board]))
        folds.extend(group)
    seal('training_diagnostic.json', summarize(folds))


def forward(action):
    area = TREE / SPEC['heldout_batch']
    model_path = HERE / 'weights.json'
    model = load(model_path)
    if model['spec'] != sha(HERE/'protocol.json') or model['features'] != NAMES:
        raise ValueError('incompatible checkpoint')
    # The decision stage does not read either continuation or its outcome.
    rows = opportunities(area)
    decide(rows.values(), model)
    picks = dict(selection_sha256=sha(area/'selection.json'), weights_sha256=sha(model_path),
                 empty_windows=load(area/'selection.json')['empty_windows'], rows=rows)
    if action == 'predict':
        response_exists = any(area.glob('case_*/live_parent.json')) or any(area.glob('case_*/live_alternative.json'))
        if not (HERE/'prospective_decisions.json').exists() and response_exists:
            raise ValueError('unblinded response exists before decision publication')
        seal('prospective_decisions.json', picks)
        return
    if load(HERE/'prospective_decisions.json') != picks:
        raise ValueError('held-out predictions changed')
    attach_terminal(area, rows)
    seal('prospective_scores.json', summarize(list(rows.values())))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['train', 'predict', 'score', 'wait'])
    a = ap.parse_args().action
    if a == 'train':
        train()
    elif a == 'wait':
        area = TREE / SPEC['heldout_batch']
        while not (area / 'selection.json').is_file():
            time.sleep(5)
        forward('predict')
        while not (area / 'aggregate.json').is_file():
            time.sleep(5)
        forward('score')
    else:
        forward(a)
