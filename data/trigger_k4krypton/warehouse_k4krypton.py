#!/usr/bin/env python3
"""Warehouse k4v3 + krypton gap fills via GitHub API.
Pattern: trigger_litev8/warehouse_litev8.py (two labels).
List refresh is MONOTONIC: only write the pull-time list if it is longer
than the current warehoused list (never shrink a fresher patrol list)."""
import sys, os, json, hashlib, subprocess, base64, random
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

BASE = '/home/hatch/workspace/carbon-incoming/trigger_k4krypton'
DGX = 'arm@100.84.148.73'
SSH = '/home/hatch/workspace/carbon-incoming/ssh_dgx.sh'

LABELS = [
    {'label': 'k4v3', 'dir': 'data/k4v3_corpus', 'list': 'data/k4v3_matches.json',
     'expect': 181, 'manifest': 'pull0030_k4v3_gap.manifest', 'listfile': 'k4v3_matches.json'},
    {'label': 'krypton', 'dir': 'data/krypton_corpus', 'list': 'data/krypton_matches.json',
     'expect': 186, 'manifest': 'pull0030_krypton_gap.manifest', 'listfile': 'krypton_matches.json'},
]

def md5_file(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def count_games(obj):
    if isinstance(obj, dict):
        return len(obj.get('matches', []))
    if isinstance(obj, list):
        return len(obj)
    return 0

def warehouse(cfg):
    label = cfg['label']
    man = {}
    with open(os.path.join(BASE, cfg['manifest'])) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                man[os.path.basename(parts[1])] = parts[0]
    print(f'[{label}] manifest entries: {len(man)}', flush=True)
    assert len(man) == cfg['expect'], f"manifest {len(man)} != expect {cfg['expect']}"
    entries = gh_api.get_file('robertyang87', 'carbon-league', cfg['dir'], ref='main')
    existing = {e['name'] for e in entries}
    print(f'[{label}] main corpus now: {len(existing)}', flush=True)
    missing = sorted(set(man) - existing)
    extra = sorted(existing - set(man))
    print(f'[{label}] missing: {len(missing)}, extra on main: {len(extra)}', flush=True)
    assert not extra, f'unexpected extra files on main: {extra[:5]}'
    staging = os.path.join(BASE, 'staging', label)
    os.makedirs(staging, exist_ok=True)
    ok = fail = 0
    for i, name in enumerate(missing):
        src = f'{DGX}:/home/arm/junyang/carbon-repo/{cfg["dir"]}/{name}'
        r = subprocess.run(['rsync', '-a', '-e', SSH, src, staging + '/'],
                           capture_output=True, text=True)
        p = os.path.join(staging, name)
        if r.returncode in (0, 23) and os.path.exists(p) and md5_file(p) == man[name]:
            ok += 1
        else:
            fail += 1
            print(f'  FAIL {name}: rc={r.returncode} md5={md5_file(p) if os.path.exists(p) else "missing"}', flush=True)
        if (i + 1) % 20 == 0:
            print(f'  rsync {i+1}/{len(missing)}', flush=True)
    print(f'[{label}] rsync+md5: ok {ok} / fail {fail}', flush=True)
    assert fail == 0, f'{fail} files failed rsync/md5'
    put_ok = put_fail = 0
    for i, name in enumerate(missing):
        with open(os.path.join(staging, name), 'rb') as f:
            data = f.read()
        try:
            gh_api.put_file('robertyang87', 'carbon-league', f'{cfg["dir"]}/{name}', data,
                            f'{label} corpus: add {name} (gap fill, {cfg["expect"]}/{cfg["expect"]})', branch='main')
            put_ok += 1
        except Exception as e:
            put_fail += 1
            print(f'  PUT FAIL {name}: {e}', flush=True)
        if (i + 1) % 20 == 0:
            print(f'  put {i+1}/{len(missing)}', flush=True)
    print(f'[{label}] put: ok {put_ok} / fail {put_fail}', flush=True)
    assert put_fail == 0, f'{put_fail} puts failed'
    cur = gh_api.get_file('robertyang87', 'carbon-league', cfg['list'], ref='main')
    cur_list = json.loads(base64.b64decode(cur['content']))
    cur_n = count_games(cur_list)
    print(f'[{label}] warehoused list now: {cur_n}', flush=True)
    if cfg['expect'] > cur_n:
        new_list = json.load(open(os.path.join(BASE, cfg['listfile'])))
        new_games = new_list.get('matches', new_list) if isinstance(new_list, dict) else new_list
        assert len(new_games) == cfg['expect']
        gh_api.put_file('robertyang87', 'carbon-league', cfg['list'], json.dumps(new_list).encode(),
                        f'{label} matches: refresh to {cfg["expect"]} games (gap fill)',
                        sha=cur['sha'], branch='main')
        print(f'[{label}] list refreshed {cur_n} -> {cfg["expect"]}', flush=True)
    else:
        print(f'[{label}] list already >= pull-time ({cur_n} >= {cfg["expect"]}), no refresh', flush=True)
    entries2 = gh_api.get_file('robertyang87', 'carbon-league', cfg['dir'], ref='main')
    names2 = {e['name'] for e in entries2}
    assert names2 == set(man), f'read-back id set mismatch (missing {len(set(man)-names2)}, extra {len(names2-set(man))})'
    cur2 = gh_api.get_file('robertyang87', 'carbon-league', cfg['list'], ref='main')
    n2 = count_games(json.loads(base64.b64decode(cur2['content'])))
    assert n2 >= cfg['expect'], f'list read-back {n2} < {cfg["expect"]}'
    random.seed(7)
    sample = random.sample(sorted(man), 3)
    for name in sample:
        blob = gh_api.get_file('robertyang87', 'carbon-league', f'{cfg["dir"]}/{name}', ref='main')
        got = hashlib.md5(base64.b64decode(blob['content'])).hexdigest()
        assert got == man[name], f'md5 mismatch {name}'
    print(f'[{label}] READ-BACK OK: corpus {len(names2)}/{cfg["expect"]}, list {n2}, md5 spot 3/3', flush=True)
    return {'missing': len(missing), 'corpus': len(names2), 'list': n2}

def main():
    results = {}
    for cfg in LABELS:
        results[cfg['label']] = warehouse(cfg)
    print('ALL_DONE', json.dumps(results), flush=True)

if __name__ == '__main__':
    main()
