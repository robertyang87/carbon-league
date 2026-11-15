#!/usr/bin/env python3
"""Warehouse pull0043 (krypton-prosearch +26) + pull0044 (k4v3 +22) new corpus
files + 2 refreshed matches lists to main in ONE Git Data API commit.
Aborts on any local md5 mismatch. On PATCH 422 (main moved during upload),
re-parents the same tree onto the fresh head once.
"""
import sys, os, json, hashlib, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
B = '/home/hatch/workspace/carbon-incoming/pull0043_0044'
MSG = ('corpus pulls 0043-0044 (DGX direct pull, 117 endpoint): '
       'krypton-prosearch +26 (132->158), k4v3 +22 (229->251); 2 matches lists refreshed')
MATCHES = [
    ('krypton-prosearch_matches.json', 'data/krypton-prosearch_matches.json', 158),
    ('k4v3_matches.json', 'data/k4v3_matches.json', 251),
]
EXPECT_NEW = {'krypton-prosearch_corpus': 26, 'k4v3_corpus': 22}
EXPECT_TOTAL = {'krypton-prosearch_corpus': 158, 'k4v3_corpus': 251}

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    payload = []
    for corpus, n_expect in EXPECT_NEW.items():
        entries = []
        for line in open(os.path.join(B, f'new_{corpus}.txt')):
            line = line.strip()
            if not line:
                continue
            md5, rel = line.split(None, 1)
            entries.append((md5, rel.strip()))
        assert len(entries) == n_expect, f'{corpus} new {len(entries)} != {n_expect}'
        for md5, rel in entries:
            payload.append((rel, os.path.join(B, 'files', rel), md5))
    for local_name, remote, n_expect in MATCHES:
        data = open(os.path.join(B, local_name), 'rb').read()
        n = len(json.loads(data))
        print(f'matches {remote}: {n} entries, md5 {hashlib.md5(data).hexdigest()}', flush=True)
        assert n == n_expect, f'{remote} entries {n} != {n_expect}'
        payload.append((remote, os.path.join(B, local_name), None))
    print(f'payload: {len(payload)} files (48 corpus + 2 matches)', flush=True)
    assert len(payload) == 50

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

    blob_shas = []
    for remote, local, md5 in payload:
        with open(local, 'rb') as f:
            data = f.read()
        blob = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                          {'content': base64.b64encode(data).decode(), 'encoding': 'base64'})
        blob_shas.append({'path': remote, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
    print(f'blobs done: {len(blob_shas)}', flush=True)

    for attempt in (1, 2):
        ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
        head = ref['object']['sha']
        commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
        base_tree = commit['tree']['sha']
        print(f'attempt {attempt}: HEAD {head[:12]} base_tree {base_tree[:12]}', flush=True)
        tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                          {'base_tree': base_tree, 'tree': blob_shas})
        new_commit = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                                {'message': MSG, 'tree': tree['sha'], 'parents': [head]})
        print('new commit:', new_commit['sha'], flush=True)
        try:
            patched = gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main',
                                 {'sha': new_commit['sha']})
        except Exception as ex:
            print(f'PATCH failed attempt {attempt}: {ex}', flush=True)
            if attempt == 2:
                raise
            continue
        assert patched['object']['sha'] == new_commit['sha']
        break

    ok = True
    for corpus, total in EXPECT_TOTAL.items():
        listing = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/data/{corpus}?ref=main')
        n = len([e for e in listing if e['type'] == 'file'])
        print(f'readback {corpus}: {n} (expect {total})', flush=True)
        if n != total:
            ok = False
    print('WAREHOUSE OK commit', new_commit['sha'], 'readback_ok=', ok, flush=True)
    return 0 if ok else 2

if __name__ == '__main__':
    sys.exit(main())
