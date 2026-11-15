#!/usr/bin/env python3
"""Warehouse Lite-V8 gap fill (185/185) via GitHub API.
Pattern: trigger_gap3/warehouse_gap3.py (single label).
List refresh is MONOTONIC: only write the pull-time list if it is longer
than the current warehoused list (never shrink a fresher patrol list)."""
import sys, os, json, hashlib, subprocess, base64, random
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

BASE = '/home/hatch/workspace/carbon-incoming/trigger_litev8'
LABEL = 'lite-v8'
DIR = 'data/lite-v8_corpus'
LIST = 'data/lite-v8_matches.json'
EXPECT = 185
MANIFEST = 'pull0030_lite-v8_gap.manifest'
DGX = 'arm@100.84.148.73'
SSH = '/home/hatch/workspace/carbon-incoming/ssh_dgx.sh'

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

def main():
    man = {}
    with open(os.path.join(BASE, MANIFEST)) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                man[os.path.basename(parts[1])] = parts[0]
    print(f'[{LABEL}] manifest entries: {len(man)}', flush=True)
    assert len(man) == EXPECT, f"manifest {len(man)} != expect {EXPECT}"
    entries = gh_api.get_file('robertyang87', 'carbon-league', DIR, ref='main')
    existing = {e['name'] for e in entries}
    print(f'[{LABEL}] main corpus now: {len(existing)}', flush=True)
    missing = sorted(set(man) - existing)
    extra = sorted(existing - set(man))
    print(f'[{LABEL}] missing: {len(missing)}, extra on main: {len(extra)}', flush=True)
    assert not extra, f'unexpected extra files on main: {extra[:5]}'
    staging = os.path.join(BASE, 'staging', LABEL)
    os.makedirs(staging, exist_ok=True)
    ok = fail = 0
    for i, name in enumerate(missing):
        src = f'{DGX}:/home/arm/junyang/carbon-repo/{DIR}/{name}'
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
    print(f'[{LABEL}] rsync+md5: ok {ok} / fail {fail}', flush=True)
    assert fail == 0, f'{fail} files failed rsync/md5'
    put_ok = put_fail = 0
    for i, name in enumerate(missing):
        with open(os.path.join(staging, name), 'rb') as f:
            data = f.read()
        try:
            gh_api.put_file('robertyang87', 'carbon-league', f'{DIR}/{name}', data,
                            f'{LABEL} corpus: add {name} (gap fill, {EXPECT}/{EXPECT})', branch='main')
            put_ok += 1
        except Exception as e:
            put_fail += 1
            print(f'  PUT FAIL {name}: {e}', flush=True)
        if (i + 1) % 20 == 0:
            print(f'  put {i+1}/{len(missing)}', flush=True)
    print(f'[{LABEL}] put: ok {put_ok} / fail {put_fail}', flush=True)
    assert put_fail == 0, f'{put_fail} puts failed'
    # 5. refresh list MONOTONICALLY (only forward)
    cur = gh_api.get_file('robertyang87', 'carbon-league', LIST, ref='main')
    cur_list = json.loads(base64.b64decode(cur['content']))
    cur_n = count_games(cur_list)
    print(f'[{LABEL}] warehoused list now: {cur_n}', flush=True)
    if EXPECT > cur_n:
        new_list = json.load(open(os.path.join(BASE, 'lite-v8_matches.json')))
        new_games = new_list.get('matches', new_list) if isinstance(new_list, dict) else new_list
        assert len(new_games) == EXPECT
        gh_api.put_file('robertyang87', 'carbon-league', LIST, json.dumps(new_list).encode(),
                        f'{LABEL} matches: refresh to {EXPECT} games (gap fill)',
                        sha=cur['sha'], branch='main')
        print(f'[{LABEL}] list refreshed {cur_n} -> {EXPECT}', flush=True)
    else:
        print(f'[{LABEL}] list already >= pull-time ({cur_n} >= {EXPECT}), no refresh', flush=True)
    # 6. read-back verify
    entries2 = gh_api.get_file('robertyang87', 'carbon-league', DIR, ref='main')
    names2 = {e['name'] for e in entries2}
    assert names2 == set(man), f'read-back id set mismatch (missing {len(set(man)-names2)}, extra {len(names2-set(man))})'
    cur2 = gh_api.get_file('robertyang87', 'carbon-league', LIST, ref='main')
    n2 = count_games(json.loads(base64.b64decode(cur2['content'])))
    assert n2 >= EXPECT, f'list read-back {n2} < {EXPECT}'
    random.seed(7)
    sample = random.sample(sorted(man), 3)
    for name in sample:
        blob = gh_api.get_file('robertyang87', 'carbon-league', f'{DIR}/{name}', ref='main')
        got = hashlib.md5(base64.b64decode(blob['content'])).hexdigest()
        assert got == man[name], f'md5 mismatch {name}'
    print(f'[{LABEL}] READ-BACK OK: corpus {len(names2)}/{EXPECT}, list {n2}, md5 spot 3/3', flush=True)
    print('ALL_DONE', json.dumps({LABEL: {'missing': len(missing), 'corpus': len(names2), 'list': n2}}), flush=True)

if __name__ == '__main__':
    main()
