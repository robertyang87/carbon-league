#!/usr/bin/env python3
"""Warehouse hakimi3pro pull0038 gap (20 new files; corpus 161 -> 181) +
refreshed matches list (181 entries) to main in ONE Git Data API commit.
Re-reads HEAD fresh inside the push. Aborts on any local md5 mismatch.
"""
import sys, os, json, hashlib, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
B = '/home/hatch/workspace/carbon-incoming/hakimi3pro_pull0038'

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    missing = []
    for line in open(os.path.join(B, 'missing.txt')):
        line = line.strip()
        if not line:
            continue
        md5, rel = line.split(None, 1)
        missing.append((md5, rel.strip()))
    assert len(missing) == 20, f'missing {len(missing)} != 20'
    payload = []
    for md5, rel in missing:
        payload.append((rel, os.path.join(B, 'corpus_new', os.path.basename(rel)), md5))
    data = open(os.path.join(B, 'hakimi3pro_matches.json'), 'rb').read()
    n = len(json.loads(data))
    print(f'matches data/hakimi3pro_matches.json: {n} entries, md5 {hashlib.md5(data).hexdigest()}', flush=True)
    assert n == 181, f'matches entries {n}'
    payload.append(('data/hakimi3pro_matches.json', os.path.join(B, 'hakimi3pro_matches.json'), None))
    print(f'payload: {len(payload)} files (20 corpus + 1 matches)', flush=True)

    bad = []
    for remote, local, md5 in payload:
        if not os.path.exists(local):
            bad.append((remote, 'MISSING'))
        elif md5 and md5_of(local) != md5:
            bad.append((remote, 'MD5-MISMATCH'))
    print(f'local md5 check bad={len(bad)}', bad[:5], flush=True)
    if bad:
        print('ABORT (md5 failures)', flush=True)
        return 1

    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    base_tree = commit['tree']['sha']
    print(f'HEAD {head[:12]} base_tree {base_tree[:12]}', flush=True)

    tree_entries = []
    for remote, local, md5 in payload:
        with open(local, 'rb') as f:
            data = f.read()
        blob = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                          {'content': base64.b64encode(data).decode(), 'encoding': 'base64'})
        tree_entries.append({'path': remote, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
    print(f'blobs done: {len(tree_entries)}', flush=True)

    tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                      {'base_tree': base_tree, 'tree': tree_entries})
    new_commit = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                            {'message': 'corpus hakimi3pro: gap pull0038 +20 games (corpus 161->181); matches list refreshed to 181 (DGX direct pull pull0038)',
                             'tree': tree['sha'], 'parents': [head]})
    print('new commit:', new_commit['sha'], flush=True)
    patched = gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main',
                         {'sha': new_commit['sha']})
    print('ref updated to:', patched['object']['sha'], flush=True)
    assert patched['object']['sha'] == new_commit['sha']
    print('WAREHOUSE OK commit', new_commit['sha'], flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
