"""Train and test a future-rent model on signed original-v2 economic traces."""
import argparse
from collections import Counter, defaultdict
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import zipfile

from features import HORIZON, NAMES, extract

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
sys.path.insert(0, str(FAMILY / 'v2_original_cash_paths'))
from study import EXAMINER, LEDGER, PARENT, select, sha

LATER = {'native_speed_dev', 'switch_margin_dev'}
STEPS = tuple(range(20, 101, 5))


def save(name, obj):
    raw = (json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=None)+'\n').encode()
    if name.endswith('.gz'):
        raw = gzip.compress(raw, mtime=0)
    path = HERE/name
    stage = path.with_suffix('.partial')
    stage.write_bytes(raw)
    stage.replace(path)
    return sha(raw)


def ticket_key(t):
    return t['panel'], t['game']['terrain'], t['game']['seat']


def protocol():
    tickets = select()
    items = []
    for t in tickets:
        panel, terrain, seat = ticket_key(t)
        items.append(dict(panel=panel, terrain=terrain, seat=seat, own=t['game']['seed'],
                          split='held' if panel in LATER else 'fit', tape=t['archive_sha256'],
                          receipt=t['game']['receipt_sha256'], reg=t['registration_sha256']))
    assert len(items) == 44
    return dict(schema='public-live-tree-tail/1', v2=PARENT, original=EXAMINER, horizon=HORIZON,
                steps=STEPS, features=NAMES, rows=items, ridge=.1,
                label='Income from starting live tree under its starting holder, until horizon or first ownership change, '
                      'cash absorption and direct collision only; own or rival both predicted from V2-seat public belief.',
                coefficient_seed=0, feature_sha256=sha((HERE/'features.py').read_bytes()))


def ledger_module():
    spec = importlib.util.spec_from_file_location('original_v2_cash_ledger', LEDGER)
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    return ledger


def received(rows, start, tree_id, holder):
    """Count once per tree and attribute the cash event to the actual holder."""
    totals = Counter()
    for row in rows[start:start+HORIZON]:
        for kind, seat, value, *detail in row['cash_events']:
            if seat == holder and ((kind == 'tree_absorption' and detail[0] == tree_id)
                                   or (kind == 'tree_collision_cash' and detail[2] == tree_id)):
                totals[kind] += value
        if any(change[1] == tree_id for change in row['tree_changes']):
            break
    return dict(totals)


def collect(split, lock):
    ledger = ledger_module()
    cfg = ledger.config()
    raw = (FAMILY / 'ladder/prediction_feedback/package/candidate.py').read_bytes()
    assert sha(raw) == PARENT
    bot = types.ModuleType('signed_v2')
    exec(compile(raw, '<signed-v2>', 'exec'), bot.__dict__)
    frozen = {(b['panel'], b['terrain'], b['seat']): b for b in lock['rows'] if b['split'] == split}
    result = []
    for t in select():
        key = ticket_key(t)
        if key not in frozen:
            continue
        game = t['game']
        ref = frozen[key]
        path = ROOT/t['archive']
        assert sha(path.read_bytes()) == ref['tape'] and t['registration_sha256'] == ref['reg']
        seat = game['seat']
        with zipfile.ZipFile(path) as z:
            book = ledger.replay(game, z, cfg)
            prefix = game['arm']+'/'
            views = gzip.decompress(z.read(prefix+'observations.jsonl.gz')).splitlines()
        rows = book['transitions']
        assert len(rows) == len(views) == 299
        tracker = bot.BeliefTracker(bot._Bel, seat)
        examples = 0
        for step, (source, entry) in enumerate(zip(views, rows, strict=True)):
            if step > STEPS[-1]:
                break
            packet = json.loads(source)
            ob = packet['observations'][seat]
            assert 'full_carbon' not in ob and ob['step'] == packet['step'] == entry['t'] == step
            assert ob['player'] == seat and packet['public_observation_sha256'] == entry['public_observation_sha256']
            digest = sha((json.dumps(ob,separators=(',',':'))+'\n').encode())
            assert digest == packet['public_observation_sha256'][seat]
            est, known = tracker.step(ob['players'], step, ob['carbon'])
            if step in STEPS:
                for tree_id, holder, values in extract(bot, ob['players'], step, est, known):
                    parts = received(rows, step, tree_id, holder)
                    result.append(dict(terrain=game['terrain'], seed=game['seed'], seat=seat, step=step,
                                       holder=holder, tree=tree_id, x=values, y=sum(parts.values()), earned=parts))
                    examples += 1
            tracker.set_actions(ob['players'], entry['commands_by_seat'][seat])
        print(split, game['seed'], seat, examples, flush=True)
    return result


def arrays(rows):
    import numpy as np
    x = np.asarray([r['x'] for r in rows])
    y = np.asarray([r['y'] for r in rows])
    ids = [r['terrain'] for r in rows]
    counts = Counter(ids)
    w = np.asarray([1/counts[i] for i in ids])
    w /= sum(w)
    return x, y, w


def fit(rows, reg):
    import numpy as np
    x, y, w = arrays(rows)
    mean = w@x
    scale = np.maximum(np.sqrt(w@np.square(x-mean)), 1e-6)
    z = (x-mean)/scale
    intercept = float(w@y)
    coef = np.linalg.solve((z.T*w)@z+reg*np.eye(len(NAMES)), (z.T*w)@(y-intercept))
    rate = w@(y/np.maximum(x[:, 1], 1))
    return dict(mean=mean.tolist(), scale=scale.tolist(), intercept=intercept, coef=coef.tolist(),
                simple_per_remaining_turn=float(rate))


def evaluate(rows, weights):
    import numpy as np
    x, y, _ = arrays(rows)
    z = (x-weights['mean']) / weights['scale']
    p = np.maximum(0, weights['intercept']+z@weights['coef'])
    baseline = weights['simple_per_remaining_turn'] * x[:, 1]
    for r, prediction, reference in zip(rows, p, baseline, strict=True):
        r.update(prediction=float(prediction), remaining_turn_baseline=float(reference))
    by_map = defaultdict(list)
    by_root = defaultdict(list)
    for r in rows:
        by_map[r['terrain']].append(r)
        by_root[(r['terrain'],r['seat'],r['step'])].append(r)
    report = {}
    for terrain, group in by_map.items():
        root_group = [v for k,v in by_root.items() if k[0]==terrain]
        report[terrain] = dict(samples=len(group), roots=len(root_group),
            tree_mae={key: float(np.mean([abs(r[key]-r['y']) for r in group]))
                      for key in ('prediction','remaining_turn_baseline')},
            margin_mae={key: float(np.mean([abs(sum((1 if r['seat']==r['holder'] else -1)*(r[key]-r['y']) for r in root))
                                            for root in root_group])) for key in ('prediction','remaining_turn_baseline')})
    return report


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase', choices=('register','fit','held'))
    cli.add_argument('--cpu', type=int, required=True)
    a = cli.parse_args()
    os.sched_setaffinity(0,{a.cpu})
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='1'
    if a.phase == 'register':
        path = HERE/'protocol.json'
        data = protocol()
        assert not path.exists() or json.loads(path.read_bytes()) == json.loads(json.dumps(data))
        print(save('protocol.json',data),flush=True)
        return
    frozen = json.loads((HERE/'protocol.json').read_bytes())
    assert frozen == json.loads(json.dumps(protocol()))
    if a.phase == 'held':
        weights = json.loads((HERE/'weights.json').read_bytes())
        assert weights['train_rows_sha256'] == sha((HERE/'fit_rows.json.gz').read_bytes())
    rows = collect(a.phase,frozen)
    if a.phase == 'fit':
        weights = fit(rows,frozen['ridge'])
    else:
        allowed = {r['terrain'] for r in frozen['rows'] if r['split']=='fit'}
        assert {r['terrain'] for r in rows}.isdisjoint(allowed)
    metrics = evaluate(rows,weights)
    rows_hash = save(a.phase+'_rows.json.gz', rows)
    if a.phase == 'fit':
        weights['train_rows_sha256'] = rows_hash
        save('weights.json',weights)
    save(a.phase+'.json', dict(maps=metrics, rows_sha256=rows_hash, weights_sha256=sha((HERE/'weights.json').read_bytes())))
    print(a.phase,len(rows),len(metrics),flush=True)


if __name__ == '__main__':
    main()
