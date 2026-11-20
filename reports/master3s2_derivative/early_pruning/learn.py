"""Fit a shallow-pruning residual on frozen critic activations and score whole held menus."""
import argparse
import gzip
import json
from collections import defaultdict

import numpy as np
from register import HERE, save, sha


def load(split, salt):
    rows = []
    protocol = json.loads((HERE/'protocol.json').read_bytes())
    for card in protocol['sources']:
        if card['split'] != split:
            continue
        archive = HERE/f'{split}_{salt}'/f"{card['seed']}.json.gz"
        observed = json.loads(gzip.decompress(archive.read_bytes()))
        assert [r['step'] for r in observed] == protocol['steps']
        rows.extend(r for r in observed if 'menu' in r)
    return rows


def centered(rows):
    by_map = defaultdict(list)
    for row in rows:
        x = np.asarray(row['features'], dtype=np.float64)
        delta = np.subtract(row['8'], row['1'])
        by_map[row['seed']].append((x-x.mean(axis=0), delta-delta.mean()))
    xs, ys = [], []
    for roots in by_map.values():
        for x, y in roots:
            w = (len(by_map)*len(roots)*len(y))**-0.5
            xs.append(w*x)
            ys.append(w*y)
    return np.vstack(xs), np.concatenate(ys)


def fit():
    rows = load('train', 0)
    protocol = json.loads((HERE/'protocol.json').read_bytes())
    x, y = centered(rows)
    scale = np.maximum(np.sqrt(np.sum(x*x, axis=0)), 1e-6)
    z = x/scale
    w = np.linalg.solve(z.T@z + protocol['ridge_penalty']*np.eye(z.shape[1]), z.T@y) / scale
    assert np.isfinite(w).all()
    save(HERE/'weights.json', dict(coefficients=w.tolist(), count=len(w), roots=len(rows),
         maps=len(set(r['seed'] for r in rows)), pairs_of_root_and_action=len(y),
         protocol_sha256=sha((HERE/'protocol.json').read_bytes()),
         target='H8 minus native H1, centered by menu; correction only changes ordering at H1.'))


def measure(row, weights):
    feats = np.asarray(row['features'])
    one = np.asarray(row['1'])
    four = np.asarray(row['4'])
    deep = np.asarray(row['8'])
    actions = list(range(len(one)))
    def result(first):
        shortlist = [0] + sorted(actions[1:], key=lambda i:-first[i])[:max(2, int(np.ceil(len(actions)*.5)))-1]
        finals = [0] + sorted(shortlist[1:], key=lambda i:-four[i])[:max(2, int(np.ceil(len(actions)*.25)))-1]
        final_score = max(deep[i] for i in finals)
        chosen = max(finals, key=lambda i:deep[i])
        return dict(shortlist=shortlist, finals=finals, chosen=chosen, final_score=float(final_score),
                    regret=float(max(deep)-final_score), best_survives_one=bool(max(deep[i] for i in shortlist) == max(deep)))
    base=result(one)
    learned=result(one+(feats-feats[0])@weights)
    return dict(seed=row['seed'],step=row['step'],seat=row['seat'],later_salt=row['later_salt'],
                menu=len(one),base=base,learned=learned,improvement=base['regret']-learned['regret'])


def settle():
    weights=np.asarray(json.loads((HERE/'weights.json').read_bytes())['coefficients'])
    out = {}
    for salt in (0,79031):
        rows=[measure(r,weights) for r in load('held',salt)]
        seeds=sorted({r['seed'] for r in rows})
        means=[float(np.mean([r['improvement'] for r in rows if r['seed']==s])) for s in seeds]
        terminal=np.mean([r['improvement'] for r in rows if r['step']==291])
        out[str(salt)]=dict(roots=len(rows), map_ids=seeds, map_mean_change=means,
            equal_map_mean=float(np.mean(means)),terminal_change=float(terminal),
            base_regret=float(np.mean([r['base']['regret'] for r in rows])),
            learned_regret=float(np.mean([r['learned']['regret'] for r in rows])),
            pass_criteria=bool(np.mean(means)>0 and sum(d>0 for d in means)>=3 and terminal>=0),
            changed_choices=sum(r['base']['chosen']!=r['learned']['chosen'] for r in rows),details=rows)
    save(HERE/'held_evaluation.json',dict(qualified_for_true_response=all(v['pass_criteria'] for v in out.values()),draws=out))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('fit','settle'))
    if p.parse_args().command=='fit':fit()
    else:settle()
