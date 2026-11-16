#!/usr/bin/env python3
"""Calibration ladder shard runner (PREREG_SEARCH.md S1): rootset32 roots,
pack B with SIMS=<rung>; each shard process runs under its own cgroup
CPUQuota=150% (1.5 cores, via systemd-run --user --scope, see run_ladder.sh).
Fresh agent per root; per-root seed 20261114+root_index (gate convention).
Records per root: wall/cpu seconds, searched, depth, leaf_evals (counting
point: leaf()), sims_step, changed, shallow, errors.
Usage: ladder.py <SIMS> <shard_idx> <n_shards> <out_json>
"""
import base64, importlib.util, json, os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, 'echo3_sims_pack.py')
ROOTS_DIR = '/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/roots'
ROOTSET = '/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/rootset32.json'

def tasks_from_replays(replays_dir, rootset_path):
    import gzip
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

def main():
    sims_rung, shard, nsh, out_json = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    spec = importlib.util.spec_from_file_location('pack_B_ladder', PACK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    w = m._dequantize(zlib.decompress(base64.b64decode(m._B1)), m._Q1)
    w['meta'] = np.array(m._M1)
    conf = dict(m.CONF); conf['SIMS'] = sims_rung
    tasks = tasks_from_replays(ROOTS_DIR, ROOTSET)
    mine = [t for i, t in enumerate(tasks) if i % nsh == shard]
    print(f'rung={sims_rung} shard {shard}/{nsh}: {len(mine)} roots (total {len(tasks)})', flush=True)
    rows = []
    for k, t in enumerate(mine):
        np.random.seed(20261114 + t['root_index'])
        ag = m._S.make_search_agent(conf, weights=w, rweights=None, oweights=None)
        w0 = time.perf_counter(); c0 = time.process_time()
        ag(t['obs'])
        st = ag.stats
        rows.append({'root_index': t['root_index'], 'game_key': t['game_key'],
                     'seat': t['seat'], 'step': t['step'],
                     'wall': round(time.perf_counter() - w0, 4),
                     'cpu': round(time.process_time() - c0, 4),
                     'searched': int(st.get('searched', 0)),
                     'depth': int(st.get('depth_sum', 0)),
                     'leaf': int(st.get('leaf_evals', 0)),
                     'sims_step': (st.get('sims_steps') or [[0, 0]])[-1][1] if st.get('sims_steps') else 0,
                     'changed': int(st.get('changed', 0)),
                     'shallow': int(st.get('shallow', 0)),
                     'errors': int(st.get('errors', 0))})
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(mine)}', flush=True)
    json.dump({'rung': sims_rung, 'shard': shard, 'rows': rows}, open(out_json, 'w'))
    print(f'rung={sims_rung} shard {shard} DONE', flush=True)

if __name__ == '__main__':
    main()
