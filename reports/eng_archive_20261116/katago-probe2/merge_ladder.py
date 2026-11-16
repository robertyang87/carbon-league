#!/usr/bin/env python3
"""Merge ladder shard outputs for one rung -> per-rung gate numbers."""
import glob, json, sys
import numpy as np
from collections import Counter

rung = sys.argv[1]
files = sorted(glob.glob(f'/home/arm/junyang/katago_probe_scratch/probe2b/ladder/out_{rung}_*.json')) \
    if len(sys.argv) < 3 else sys.argv[2:]
rows = []
for f in files:
    rows += json.load(open(f))['rows']
rows.sort(key=lambda r: r['root_index'])
searched = [r for r in rows if r['searched']]
def stats(arr):
    a = np.array(arr, dtype=float)
    return {'n': len(a), 'mean': round(float(a.mean()), 4), 'p50': round(float(np.percentile(a, 50)), 4),
            'p95': round(float(np.percentile(a, 95)), 4), 'max': round(float(a.max()), 4)} if len(a) else None
depths = [r['depth'] for r in searched]
out = {
    'rung_sims': int(rung), 'n_roots': len(rows), 'n_searched': len(searched),
    'wall_all': stats([r['wall'] for r in rows]),
    'wall_searched': stats([r['wall'] for r in searched]),
    'cpu_all': stats([r['cpu'] for r in rows]),
    'cpu_searched': stats([r['cpu'] for r in searched]),
    'depth_counts': dict(sorted(Counter(depths).items())),
    'depth_median_searched': float(np.median(depths)) if depths else None,
    'depth_p25': float(np.percentile(depths, 25)) if depths else None,
    'depth_p75': float(np.percentile(depths, 75)) if depths else None,
    'depth_p95': float(np.percentile(depths, 95)) if depths else None,
    'leaf_max': max((r['leaf'] for r in rows), default=0),
    'leaf_mean_searched': round(float(np.mean([r['leaf'] for r in searched])), 2) if searched else None,
    'sims_eq_leaf_all_searched': all(r['sims_step'] == r['leaf'] for r in searched),
    'change_rate_searched': round(float(np.mean([1 if r['changed'] else 0 for r in searched])), 4) if searched else None,
    'errors': int(sum(r['errors'] for r in rows)),
    'shallow': int(sum(r['shallow'] for r in rows)),
}
print(json.dumps(out, indent=1))
json.dump(out, open(f'/home/arm/junyang/katago_probe_scratch/probe2b/ladder/summary_{rung}.json', 'w'), indent=1)
