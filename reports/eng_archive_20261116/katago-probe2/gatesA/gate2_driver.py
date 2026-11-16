#!/usr/bin/env python3
"""Part 1 re-registration, pre-field GATE 2 (A form): rootset32 ablation.

Per registered root (rootset32, steps [200,299), both seats, ACTIVE-status
filter -- task assembly identical to the probe-2 construction gate /
calibration ladder), drive a FRESH agent of the ORIGINAL pack (SEARCH=False,
its action = the default move) and of pack A (SEARCH=True, native wall-clock
BUDGET=0.7) on the same recorded observation with the same per-root seed
(20261114 + root_index, reset before each arm). changed := A action != orig
action (byte-compared canonical JSON).

Each shard process runs under its own systemd-run CPUQuota=150% scope
(1.5 cores) so A's wall-clock budget experience matches a quiet 1.5-core
seat (see run_gate2.sh).

Usage: gate2_driver.py <shard_idx> <n_shards> <out_json>
Env:   LIMIT=N  -> only the first N tasks of this shard (smoke).
"""
import base64, importlib.util, json, os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK_A = os.path.join(HERE, 'echo3_search_pack.py')
PACK_O = os.path.join(HERE, 'echo3_ref.py')
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


def load_pack(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    w = m._dequantize(zlib.decompress(base64.b64decode(m._B1)), m._Q1)
    w['meta'] = np.array(m._M1)
    return m, w


def call(m, w, conf, obs, seed):
    np.random.seed(seed)
    ag = m._S.make_search_agent(conf, weights=w, rweights=None, oweights=None)
    w0 = time.perf_counter(); c0 = time.process_time()
    try:
        a = ag(obs)
        err = None
    except Exception as e:
        a, err = None, repr(e)[:200]
    return a, ag.stats, err, time.perf_counter() - w0, time.process_time() - c0


def main():
    shard, nsh, out_json = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    mo, wo = load_pack('pack_orig_g2', PACK_O)
    mA, wA = load_pack('pack_A_g2', PACK_A)
    conf_o, conf_A = dict(mo.CONF), dict(mA.CONF)
    assert conf_o['SEARCH'] is False and conf_A['SEARCH'] is True
    tasks = tasks_from_replays(ROOTS_DIR, ROOTSET)
    assert len(tasks) == 6336, f'task count {len(tasks)} != 6336'
    mine = [t for i, t in enumerate(tasks) if i % nsh == shard]
    limit = int(os.environ.get('LIMIT', '0'))
    if limit:
        mine = mine[:limit]
    print(f'gate2 shard {shard}/{nsh}: {len(mine)} roots (total {len(tasks)})', flush=True)
    rows = []
    for k, t in enumerate(mine):
        seed = 20261114 + t['root_index']
        ao, _so, eo, _wo, _co = call(mo, wo, conf_o, t['obs'], seed)
        aA, sA, eA, wA_, cA = call(mA, wA, conf_A, t['obs'], seed)
        jo = json.dumps(ao, sort_keys=True) if ao is not None else None
        jA = json.dumps(aA, sort_keys=True) if aA is not None else None
        rows.append({'root_index': t['root_index'], 'game_key': t['game_key'],
                     'seat': t['seat'], 'step': t['step'],
                     'orig_action': jo, 'A_action': jA,
                     'changed': (jo is not None and jA is not None and jo != jA),
                     'A_searched': int(sA.get('searched', 0)),
                     'A_shallow': int(sA.get('shallow', 0)),
                     'A_depth': int(sA.get('depth_sum', 0)),
                     'A_changed_stat': int(sA.get('changed', 0)),
                     'A_errors': int(sA.get('errors', 0)),
                     'A_exc': eA, 'orig_exc': eo,
                     'A_wall': round(wA_, 4), 'A_cpu': round(cA, 4)})
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(mine)}', flush=True)
    json.dump({'shard': shard, 'rows': rows}, open(out_json, 'w'))
    print(f'gate2 shard {shard} DONE', flush=True)


if __name__ == '__main__':
    main()
