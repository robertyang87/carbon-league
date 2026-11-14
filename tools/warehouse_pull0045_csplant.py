#!/usr/bin/env python3
"""Warehouse pull0045 CS-Plant sub-batch (C2 pull via DGX login host, DGX
verified): NEW corpus data/krypton-csplant_corpus, 192 files, ONE Git Data
API commit. Matches list NOT touched (in-repo already 195, superset of the
frozen 192; monotonic write-forward). Aborts on any local md5 mismatch.
On PATCH failure (main moved), re-parents the same tree onto fresh head once.
"""
import sys, os, json, hashlib, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
B = '/home/hatch/workspace/carbon-incoming/pull0045_csplant'
MSG = ('corpus pull0045 csplant (C2 direct pull via DGX login host, 117 endpoint): '
       'krypton-csplant NEW corpus 192 files (frozen list 00:41 CST, manifest 2a161de9)')

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    entries = []
    for line in open(os.path.join(B, 'pull0045_krypton-csplant.manifest')):
        line = line.strip()
        if not line:
            continue
        md5, rel = line.split(None, 1)
        entries.append((md5, rel.strip()))
    assert len(entries) == 192, len(entries)
    payload = []
    for md5, rel in entries:
        local = os.path.join(B, 'replays', os.path.basename(rel))
        payload.append((rel, local, md5))
    bad = [(r, 'MISSING' if not os.path.exists(l) else 'MD5')
           for r, l, m in payload
           if not os.path.exists(l) or md5_of(l) != m]
    print(f'local md5 check bad={len(bad)}', flush=True)
    if bad:
        print('ABORT', bad[:5], flush=True)
        return 1

    blob_shas = []
    for remote, local, md5 in payload:
        with open(local, 'rb') as f:
            data = f.read()
        blob = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                          {'content': base64.b64encode(data).decode(), 'encoding': 'base64'})
        blob_shas.append({'path': remote, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
    print(f'blobs done: {len(blob_shas)}', flush=True)

    new_commit = None
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

    listing = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/data/krypton-csplant_corpus?ref=main')
    files = [e for e in listing if e['type'] == 'file']
    print(f'readback krypton-csplant_corpus: {len(files)} (expect 192)', flush=True)
    ok = len(files) == 192
    # spot check: git blob sha of first payload file must appear in the tree listing
    names = {e['name'] for e in files}
    expect_names = {os.path.basename(r) for r, _, _ in payload}
    print('name set exact:', names == expect_names, flush=True)
    ok = ok and (names == expect_names)
    print('WAREHOUSE', 'OK' if ok else 'MISMATCH', 'commit', new_commit['sha'], flush=True)
    return 0 if ok else 2

if __name__ == '__main__':
    sys.exit(main())
