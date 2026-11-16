#!/usr/bin/env python3
"""Gate-2 mechanism-distribution driver (PROVISIONAL settings, re-run after
lock): rootset32 roots x pack-B player (SEARCH=True, CLOCK=sims, SIMS=200
provisional). Per root a FRESH agent; records depth / changed / sims /
searched / shallow / dt. Task construction identical to probe-1 gate_driver
(steps [200,299), both seats, per-seat status None/ACTIVE only).
Reading line (war-room caliber this trip): depth median >= 3 over searched roots.
"""
import base64, gzip, importlib.util, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np

PACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'echo3_sims_pack.py')
ROOTS_DIR = '/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/roots'
ROOTSET = '/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/rootset32.json'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_dist')

_M = None
_W = None

def _init():
    global _M, _W
    spec = importlib.util.spec_from_file_location('pack_B_dist', PACK)
    _M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_M)
    w1 = _M._dequantize(zlib.decompress(base64.b64decode(_M._B1)), _M._Q1)
    w1['meta'] = np.array(_M._M1)
    _W = w1

def tasks_from_replays(replays_dir, rootset_path):
    games = json.load(open(rootset_path))
    tasks = []
    idx = 0
    for g in games:
        d = json.load(gzip.open(os.path.join(replays_dir, g['key']), 'rt'))
        steps = d['steps']
        for seat in (0, 1):
            for t in range(200, 299):
                if steps[t][seat].get('status') not in (None, 'ACTIVE'):
                    continue
                o0 = steps[t][0]['observation']
                ep = steps[t][seat]['observation']
                obs = {'step': int(o0['step']), 'players': o0['players'],
                       'carbon': ep['carbon'], 'player': seat,
                       'trees': o0.get('trees') or {},
                       'remainingOverageTime': ep.get('remainingOverageTime',
                                                      o0.get('remainingOverageTime'))}
                tasks.append({'game_key': g['key'], 'seat': seat, 'step': t,
                              'root_index': idx, 'obs': obs})
                idx += 1
    return tasks

def one(task):
    np.random.seed(20261114 + task['root_index'])
    ag = _M._S.make_search_agent(dict(_M.CONF), weights=_W, rweights=None, oweights=None)
    t0 = time.time()
    ag(task['obs'])
    st = ag.stats
    return {'game_key': task['game_key'], 'seat': task['seat'], 'step': task['step'],
            'root_index': task['root_index'], 'searched': int(st.get('searched', 0)),
            'depth': int(st.get('depth_sum', 0)), 'changed': int(st.get('changed', 0)),
            'sims': (st.get('sims_steps') or [[0, 0]])[-1][1] if st.get('sims_steps') else 0,
            'shallow': int(st.get('shallow', 0)), 'errors': int(st.get('errors', 0)),
            'dt': round(time.time() - t0, 4)}

def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    tasks = tasks_from_replays(ROOTS_DIR, ROOTSET)
    print(f'tasks: {len(tasks)} roots', flush=True)
    os.makedirs(OUT, exist_ok=True)
    with Pool(workers, initializer=_init) as p:
        rows = p.map(one, tasks, chunksize=16)
    with open(os.path.join(OUT, 'results.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    searched = [r for r in rows if r['searched']]
    depths = np.array([r['depth'] for r in searched], dtype=float)
    from collections import Counter
    summary = {
        'n_roots': len(rows), 'n_searched': len(searched),
        'depth_counts': dict(Counter(r['depth'] for r in searched)),
        'depth_median': float(np.median(depths)) if len(depths) else None,
        'depth_p25': float(np.percentile(depths, 25)) if len(depths) else None,
        'depth_p75': float(np.percentile(depths, 75)) if len(depths) else None,
        'depth_p95': float(np.percentile(depths, 95)) if len(depths) else None,
        'change_rate': float(np.mean([1 if r['changed'] else 0 for r in searched])) if searched else None,
        'changed_roots': int(sum(1 for r in searched if r['changed'])),
        'sims_median': float(np.median([r['sims'] for r in searched])) if searched else None,
        'errors': int(sum(r['errors'] for r in rows)),
        'settings': 'PROVISIONAL: pack B stock CONF (SEARCH=True, CLOCK=sims, SIMS=200); re-run after lock',
        'reading_line': 'depth median >= 3 (war-room caliber this trip)',
    }
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == '__main__':
    main()
