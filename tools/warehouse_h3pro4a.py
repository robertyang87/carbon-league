#!/usr/bin/env python3
"""Warehouse Hakimi v3pro trigger 4a: corpus gap (12 files) + 100-game matches
list from DGX direct pull to main. Adapted from warehouse_trigger_corpus3.py.
Phase A (--phase list): manifest vs main dir listing -> needed.txt.
Phase B (--phase put): md5-check pulled files, put them, read back counts.
The matches list is put separately (existing file, needs full blob sha).
"""
import sys, os, hashlib, argparse, json
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

BASE = '/home/hatch/workspace/carbon-incoming/trigger_h3pro4a'
OWNER, REPO = 'robertyang87', 'carbon-league'
LABEL = 'hakimi3pro'
MANIFEST = 'pull0030_hakimi3pro_4a.manifest'
REMOTE_DIR = 'data/hakimi3pro_corpus'
MATCHES_LOCAL = os.path.join(BASE, 'hakimi3pro_matches_100.json')
MATCHES_REMOTE = 'data/hakimi3pro_matches.json'

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_manifest(name):
    entries = []
    with open(os.path.join(BASE, name)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            md5, rel = line.split(None, 1)
            entries.append((md5, rel.strip()))
    return entries

def main_contents(remote_dir):
    try:
        items = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/{remote_dir}?ref=main')
        if isinstance(items, list):
            return {it['name'] for it in items}
        return set()
    except Exception as e:
        print(f'{remote_dir}: contents list failed ({e}); treating as empty', flush=True)
        return set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['list', 'put'], required=True)
    args = ap.parse_args()
    entries = load_manifest(MANIFEST)
    existing = main_contents(REMOTE_DIR)
    missing = [(md5, rel) for md5, rel in entries
               if os.path.basename(rel) not in existing]
    print(f'{LABEL}: manifest {len(entries)} on-main {len(existing)} missing {len(missing)}', flush=True)
    if args.phase == 'list':
        with open(os.path.join(BASE, 'needed.txt'), 'w') as f:
            for md5, rel in missing:
                f.write(rel + '\n')
        with open(os.path.join(BASE, 'needed_meta.txt'), 'w') as f:
            for md5, rel in missing:
                f.write(f'{LABEL}\t{md5}\t{rel}\t{REMOTE_DIR}\n')
        print('wrote needed.txt + needed_meta.txt', flush=True)
        return
    # phase put: corpus files
    bad = []
    for md5, rel in missing:
        local = os.path.join(BASE, rel)
        if not os.path.exists(local):
            bad.append((rel, 'MISSING'))
        elif md5_of(local) != md5:
            bad.append((rel, 'MD5-MISMATCH'))
    print(f'local md5 check bad={len(bad)}', bad[:5], flush=True)
    if bad:
        print('ABORT (md5 failures)', flush=True)
        return
    ok, fail = 0, []
    for md5, rel in missing:
        local = os.path.join(BASE, rel)
        with open(local, 'rb') as f:
            data = f.read()
        name = os.path.basename(rel)
        try:
            gh_api.put_file(OWNER, REPO, f'{REMOTE_DIR}/{name}', data,
                            f'corpus {LABEL}: add {name} (trigger insurance, DGX direct pull)',
                            branch='main')
            ok += 1
        except Exception as e:
            fail.append((name, str(e)[:120]))
    print(f'{LABEL}: put ok={ok} fail={len(fail)}', fail[:5], flush=True)
    items = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/{REMOTE_DIR}?ref=main')
    print(f'{LABEL}: READBACK main count = {len(items)} (target {len(entries)})', flush=True)
    # matches list (existing file: full blob sha required)
    with open(MATCHES_LOCAL, 'rb') as f:
        mdata = f.read()
    cur = gh_api.get_file(OWNER, REPO, MATCHES_REMOTE, ref='main')
    cur_games = len(json.loads(mdata))
    print(f'matches: main currently {cur.get("sha")[:12]}... putting {cur_games}-game list', flush=True)
    gh_api.put_file(OWNER, REPO, MATCHES_REMOTE, mdata,
                    f'matches {LABEL}: refresh to 100 games (trigger, DGX direct pull)',
                    sha=cur['sha'], branch='main')
    back = gh_api.get_file(OWNER, REPO, MATCHES_REMOTE, ref='main')
    import base64
    back_games = len(json.loads(base64.b64decode(back['content'])))
    print(f'matches: READBACK main games = {back_games} (target 100)', flush=True)

if __name__ == '__main__':
    main()
