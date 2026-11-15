#!/usr/bin/env python3
"""Warehouse krypton pull0036 (prosearch, first pull, 132 files) +
pull0037 (krypton-search256 gap, 25 new files) + 2 refreshed matches lists
to main in ONE Git Data API commit (blob -> tree -> commit -> PATCH ref).
Re-reads HEAD fresh inside the push. Aborts on any local md5 mismatch.
"""
import sys, os, json, hashlib, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
B = '/home/hatch/workspace/carbon-incoming/krypton_pull0036_0037'

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_manifest(name):
    entries = []
    for line in open(os.path.join(B, name)):
        line = line.strip()
        if not line:
            continue
        md5, rel = line.split(None, 1)
        entries.append((md5, rel.strip()))
    return entries

def main():
    # --- gather payload ---
    payload = []  # (remote_path, local_path, expected_md5 or None)
    man36 = load_manifest('pull0036_krypton-prosearch.manifest')
    assert len(man36) == 132, f'prosearch manifest {len(man36)} != 132'
    for md5, rel in man36:
        payload.append((rel, os.path.join(B, 'prosearch_corpus', os.path.basename(rel)), md5))
    man37 = load_manifest('pull0037_krypton-search256.manifest')
    assert len(man37) == 136, f's256 manifest {len(man37)} != 136'
    missing = [l.strip() for l in open(os.path.join(B, 's256_missing.txt')) if l.strip()]
    assert len(missing) == 25, f's256 missing {len(missing)} != 25'
    man37d = {rel: md5 for md5, rel in man37}
    for rel in missing:
        payload.append((rel, os.path.join(B, 's256_corpus', os.path.basename(rel)), man37d[rel]))
    # matches lists (refresh; main currently 129 / 134 entries)
    for local, remote in [('krypton-prosearch_matches.json', 'data/krypton-prosearch_matches.json'),
                          ('krypton-search256_matches.json', 'data/krypton-search256_matches.json')]:
        data = open(os.path.join(B, local), 'rb').read()
        n = len(json.loads(data))
        print(f'matches {remote}: {n} entries, md5 {hashlib.md5(data).hexdigest()[:12]}', flush=True)
        assert n in (132, 136), f'{remote} entries {n}'
        payload.append((remote, os.path.join(B, local), None))
    print(f'payload: {len(payload)} files (132 prosearch + 25 s256 + 2 matches)', flush=True)

    # --- local md5 gate ---
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

    # --- fresh HEAD, then blobs ---
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
        if (i + 1) % 40 == 0:
            print(f'blobs {i+1}/{len(payload)}', flush=True)
    print(f'blobs done: {len(tree_entries)}', flush=True)

    tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                      {'base_tree': base_tree, 'tree': tree_entries})
    new_commit = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                            {'message': 'corpus krypton: prosearch first pull 132 games + search256 gap 25 (total 136); matches lists refreshed to 132/136 (DGX direct pull pull0036/pull0037)',
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
