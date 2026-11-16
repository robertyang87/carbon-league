#!/usr/bin/env python3
"""Merge gate-2 shard outputs and gate-5 game outputs into summary tables.
Usage: merge_gates.py <dir_with_jsons>  (expects gate2_out_*.json, gate5_out_*.json)
"""
import glob, json, sys
import numpy as np

D = sys.argv[1]

def q(a, ps=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray(a, dtype=float)
    return {f'p{p}': round(float(np.percentile(a, p)), 4) for p in ps}

rows = []
for f in sorted(glob.glob(f'{D}/gate2_out_*.json')):
    rows.extend(json.load(open(f))['rows'])
rows.sort(key=lambda r: r['root_index'])
n = len(rows)
searched = [r for r in rows if r['A_searched'] > 0]
shallow = [r for r in searched if r['A_shallow'] > 0]
changed_s = [r for r in searched if r['changed']]
changed_all = [r for r in rows if r['changed']]
depths = [r['A_depth'] for r in searched]
depths_ns = [r['A_depth'] for r in searched if r['A_shallow'] == 0]
hist = {}
for d in depths:
    hist[d] = hist.get(d, 0) + 1
g2 = {'n_roots': n,
      'A_searched': len(searched),
      'A_shallow_of_searched': len(shallow),
      'A_errors_sum': int(sum(r['A_errors'] for r in rows)),
      'A_exc_count': sum(1 for r in rows if r['A_exc']),
      'orig_exc_count': sum(1 for r in rows if r['orig_exc']),
      'changed_over_searched': len(changed_s),
      'rate_over_searched': round(len(changed_s) / len(searched), 6) if searched else None,
      'changed_over_all': len(changed_all),
      'rate_over_all': round(len(changed_all) / n, 6),
      'changed_stat_mismatch': sum(1 for r in rows if bool(r['A_changed_stat']) != bool(r['changed'])),
      'depth_quantiles_searched': q(depths),
      'depth_quantiles_searched_nonshallow': q(depths_ns) if depths_ns else None,
      'depth_hist_searched': dict(sorted(hist.items())),
      'A_wall_quantiles': q([r['A_wall'] for r in rows]),
      'A_wall_quantiles_searched': q([r['A_wall'] for r in searched])}
print('=== GATE2 ===')
print(json.dumps(g2, indent=1, ensure_ascii=False))

for f in sorted(glob.glob(f'{D}/gate5_out_*.json')):
    g = json.load(open(f))
    rs = g['rows']
    w = np.array([r['wall'] for r in rs]); c = np.array([r['cpu'] for r in rs])
    slow = w[w > 1.0]
    out = {'seed': g['seed'], 'calls': len(rs),
           'searched_calls': int(sum(r['searched'] for r in rs)),
           'wall_mean': round(float(w.mean()), 4), 'wall_p95': round(float(np.percentile(w, 95)), 4),
           'wall_max': round(float(w.max()), 4),
           'cpu_mean': round(float(c.mean()), 4), 'cpu_p95': round(float(np.percentile(c, 95)), 4),
           'cpu_max': round(float(c.max()), 4),
           'slow_gt1s_count': int(len(slow)),
           'slow_gt1s_excess_sum': round(float((slow - 1.0).sum()), 4) if len(slow) else 0.0,
           'gt3s_count': int((w > 3.0).sum()),
           'module_load_s': g['module_load_s'], 'first_call_wall': g['first_call_wall'],
           'game_wall_s': g['game_wall_s'], 'game_cpu_s': g['game_cpu_s']}
    print(f'=== GATE5 seed {g["seed"]} ===')
    print(json.dumps(out, indent=1, ensure_ascii=False))
