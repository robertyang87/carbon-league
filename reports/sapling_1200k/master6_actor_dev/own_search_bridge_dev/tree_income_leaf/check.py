"""Audit the incremental value of recent simulated tree income at nonterminal H8 leaves."""
import json
import sys
from pathlib import Path
from statistics import mean
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bridge
import outcome_audit

base_leaf = bridge.leaf

def income_leaf(sim, seat, pre):
    value, terminal = base_leaf(sim, seat, pre)
    if terminal:
        return value, True
    remaining = sim.config['episodeSteps'] - 1 - sim.step_count
    decay = 1.0 - sim.config['cellAbsorptionRate']
    for tree in sim.trees.values():
        life = sim.config['treeLifespan'] - tree.age
        turns = max(0, min(8, remaining, life))
        last_income = sim.trees_dict[tree.id][1]
        expected = last_income * sum(decay ** i for i in range(1, turns + 1))
        value += expected if tree.player == seat else -expected
    return value, False

def summarize(rows):
    maps = sorted({r['terrain'] for r in rows})
    gains = [mean(r['gain'] for r in rows if r['terrain'] == m) for m in maps]
    return dict(roots=len(rows), maps=len(maps), changes=sum(r['changed'] for r in rows),
                wins=sum(r['margin'] > 0 for r in rows), old_wins=sum(r['old_margin']>0 for r in rows),
                gain=mean(gains), positive_maps=sum(v>0 for v in gains), map_gains=dict(zip(maps,gains)))

def run(estimator, destination):
    old = bridge.load(HERE.parent/'outcome_audit.json')['roots']
    with patch.object(bridge, 'leaf', estimator):
        forecast = outcome_audit.public_input([r['case'] for r in old])
    rows = []
    for r in old:
        case = r['case']
        delta = forecast[case]['mean_delta']['h8']
        chosen = 'alternative' if delta > 0 else 'parent'
        original = r['policies']['h8']['arm']
        scores = {}
        for arm, rec in r['arms'].items():
            path = bridge.ROOT/case/f'live_{arm}.json'
            assert bridge.sha(path) == rec['receipt_sha256']
            scores[arm] = bridge.load(path)['margin']
        rows.append(dict(case=case, terrain=r['terrain'], batch=r['batch'], delta=delta,
                         chosen=chosen, old=original, changed=chosen!=original, margin=scores[chosen],
                         old_margin=scores[original], gain=scores[chosen]-scores[original]))
    assert len({r['case'] for r in rows}) == 64
    stats = {k:summarize(g) for k,g in [('development',[r for r in rows if r['batch']<=6]),
                                        ('prior_hold',[r for r in rows if r['batch']>6])]}
    stats['admit'] = stats['prior_hold']['gain']>0 and stats['prior_hold']['positive_maps']>=3
    bridge.save(destination,dict(summary=stats, rows=rows))
    print(json.dumps(stats))
if __name__=='__main__':
    run(income_leaf, HERE/'results.json')
