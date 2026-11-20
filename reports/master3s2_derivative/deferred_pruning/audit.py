"""Test deferred pruning using an equal-or-lower plan against separate simulation draws."""
import argparse
import ast
import gzip
import json
from pathlib import Path
import runpy
from statistics import mean

import numpy as np

from build import FAMILY, HERE, PARENT, PARENT_SHA, sha

PANEL = FAMILY / 'early_pruning/repeat_values'
TRIAL_STAGES = ((4, 0.25),)


def encode(obj):
    return (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode()


def protocol():
    source = (PANEL / 'protocol.json').read_bytes()
    curves = runpy.run_path(str(PANEL / 'repeat.py'))
    cards = json.loads(source)['sources']
    files = [(card, [curves['file_for'](card, salt) for salt in curves['DRAWS']]) for card in cards]
    return dict(
        schema='deferred-pruning-equal-plan/1', parent=PARENT_SHA, data_protocol=sha(source),
        source_sha={str(p.relative_to(FAMILY)): sha(p.read_bytes()) for _, paths in files for p in paths},
        first_elimination=4, final_depth=8,
        width='Largest archived ordered prefix whose planned rows, board advances and passes each fit the full original plan.',
        evaluation='For every draw, choose using that draw. Use the other five H8 outcomes to score the choice. '
                   'Aggregate per root, then per terrain and within existing data splits. H1 and root foe action are shared.',
        caution='All labels are simulated critic values or embedded step-limit values, not official terminal matches.',
    )


def evaluate():
    curves = runpy.run_path(str(PANEL / 'repeat.py'))
    assert sha(PARENT.read_bytes()) == PARENT_SHA
    ns = {'np': np, '_S_ROWS': 32}
    tree = ast.parse(PARENT.read_bytes())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_plan')
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(PARENT), 'exec'), ns)
    plan = ns['_plan']

    def keep(ids, score, n):
        return [0] + sorted((i for i in ids if i), key=lambda i: -score[i])[:n-1]

    outcome = {}
    for split in ('train', 'held', 'extra'):
        rows = []
        for seed, step, values in curves['groups'](split):
            n = values.shape[2]
            ns['_S_STAGES'] = ((1, 0.5), (4, 0.25))
            base_cost = plan(n, 8)
            ns['_S_STAGES'] = TRIAL_STAGES
            width = max(k for k in range(2, n+1) if all(
                u <= v for u, v in zip(plan(k, 8), base_cost, strict=True)
            ))
            trials = []
            for i, (first, fourth, deep) in enumerate(values):
                remaining = keep(range(n), first, max(2, (n+1)//2))
                base = keep(remaining, fourth, max(2, (n+3)//4))
                late = keep(range(width), fourth, max(2, (width+3)//4))
                truth = np.delete(values[:, 2, :], i, axis=0).mean(axis=0)
                b, a = max(base, key=lambda j: deep[j]), max(late, key=lambda j: deep[j])
                ideal = int(truth.argmax())
                trials.append(dict(base=b, trial=a, target_ideal=ideal, base_finalists=base, trial_finalists=late,
                                   scores=[float(truth[j]) for j in (b, a, ideal)], gain=float(truth[a]-truth[b])))
            rows.append(dict(seed=seed, step=step, width=width, native_width=n, base_cost=base_cost,
                             trial_cost=plan(width, 8), trials=trials, gain=mean(t['gain'] for t in trials)))
        pools = {seed: mean(r['gain'] for r in rows if r['seed']==seed) for seed in sorted({r['seed'] for r in rows})}
        outcome[split] = dict(rows=rows, by_map=pools, mean_gain=mean(pools.values()))
    return outcome


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--register', action='store_true')
    args = cli.parse_args()
    p = HERE/'audit_protocol.json'
    raw = encode(protocol())
    if args.register:
        assert not p.exists() or p.read_bytes()==raw
        p.write_bytes(raw)
        print(sha(raw))
    else:
        assert p.read_bytes()==raw
        result = evaluate()
        (HERE/'audit_results.json.gz').write_bytes(gzip.compress(encode(result), mtime=0))
        for split, group in result.items():
            print(split, len(group['rows']), group['mean_gain'], group['by_map'])
