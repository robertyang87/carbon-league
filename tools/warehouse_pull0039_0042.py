#!/usr/bin/env python3
"""Warehouse pull0039-0042 new corpus files (288) + 4 refreshed matches lists
to main in ONE Git Data API commit. Re-reads HEAD fresh inside the push.
Aborts on any local md5 mismatch. Then read-back verifies dir counts.
"""
import sys, os, json, hashlib, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
B = '/home/hatch/workspace/carbon-incoming/pull0039_0042'

MATCHES = [
    ('fable6_matches.json', 'data/fable6_matches.json', 256),
    ('lite-v8_matches.json', 'data/lite-v8_matches.json', 268),
    ('krypton-search256-v2_matches.json', 'data/krypton-search256-v2_matches.json', 105),
    ('krypton-cloudsearch_matches.json', 'data/krypton-cloudsearch_matches.json', 125),
]
EXPECT_NEW = {'fable6_corpus': 31, 'lite-v8_corpus': 27,
              'krypton-search256-v2_corpus': 105, 'krypton-cloudsearch_corpus': 125}
EXPECT_TOTAL = {'fable6_corpus': 256, 'lite-v8_corpus': 268,
                'krypton-search256-v2_corpus': 105, 'krypton-cloudsearch_corpus': 125}

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    payload = []  # (remote, local, md5)
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
    print(f'payload: {len(payload)} files (288 corpus + 4 matches)', flush=True)
    assert len(payload) == 292

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
    for i, (remote, local, md5) in enumerate(payload):
        with open(local, 'rb') as f:
            data = f.read()
        blob = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                          {'content': base64.b64encode(data).decode(), 'encoding': 'base64'})
        tree_entries.append({'path': remote, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
        if (i + 1) % 50 == 0:
            print(f'blobs {i+1}/{len(payload)}', flush=True)
    print(f'blobs done: {len(tree_entries)}', flush=True)

    tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                      {'base_tree': base_tree, 'tree': tree_entries})
    new_commit = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                            {'message': 'corpus pulls 0039-0042 (DGX direct pull, 117 endpoint): fable6 +31 (225->256), lite-v8 +27 (241->268), krypton-search256-v2 first pull 105, krypton-cloudsearch-v1 first pull 125; 4 matches lists refreshed',
                             'tree': tree['sha'], 'parents': [head]})
    print('new commit:', new_commit['sha'], flush=True)
    patched = gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main',
                         {'sha': new_commit['sha']})
    print('ref updated to:', patched['object']['sha'], flush=True)
    assert patched['object']['sha'] == new_commit['sha']

    # read-back: per-dir counts on the new commit's tree state
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
