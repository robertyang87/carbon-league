#!/usr/bin/env python3
"""Warehouse K0 three-arm packs (code/exp/k0_{b0,a0,a1}.py) via Git Data API:
blobs -> tree(base_tree=HEAD tree) -> commit -> PATCH ref. One commit.
Then read back each blob via API and md5-verify. Checkpointed blob shas."""
import sys, os, json, base64, hashlib
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
STAGE = '/home/hatch/workspace/carbon-live/k0_packs_staging'
FILES = {
    'k0_b0.py': '59a8b4cd751d724a2bb44632384aaee4',
    'k0_a0.py': 'e20c48ba0450477a04f9817e9b33ff8f',
    'k0_a1.py': 'fd3e9fca6343b0105f96a3cb359a4b17',
}
ckpt_path = os.path.join(STAGE, 'blob_shas.json')
blobs = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

# 0. local sanity
for name, want in FILES.items():
    data = open(os.path.join(STAGE, name), 'rb').read()
    got = hashlib.md5(data).hexdigest()
    assert got == want, f'{name} local md5 {got} != {want}'
print('local md5 all OK', flush=True)

# 1. blobs (checkpointed)
for name in FILES:
    if name in blobs:
        continue
    data = open(os.path.join(STAGE, name), 'rb').read()
    r = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                   {'content': base64.b64encode(data).decode(),
                    'encoding': 'base64'})
    blobs[name] = r['sha']
    json.dump(blobs, open(ckpt_path, 'w'))
    print(f'blob {name} -> {r["sha"]}', flush=True)

# 2. one commit, retry from fresh head on PATCH conflict
commit_sha = None
for attempt in range(5):
    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    base_tree = commit['tree']['sha']
    entries = [{'path': f'code/exp/{name}', 'mode': '100644',
                'type': 'blob', 'sha': blobs[name]} for name in FILES]
    tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                      {'base_tree': base_tree, 'tree': entries})
    msg = ('K0 three-arm packs warehouse: code/exp/k0_b0.py '
           '(md5 59a8b4cd751d724a2bb44632384aaee4), k0_a0.py '
           '(md5 e20c48ba0450477a04f9817e9b33ff8f), k0_a1.py '
           '(md5 fd3e9fca6343b0105f96a3cb359a4b17) - DGX-built, '
           'war-room Git Data API warehouse')
    newc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                      {'message': msg, 'tree': tree['sha'],
                       'parents': [head]})
    try:
        gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main',
                   {'sha': newc['sha'], 'force': False})
        commit_sha = newc['sha']
        print(f'COMMIT {commit_sha} (parent {head})', flush=True)
        break
    except Exception as e:
        print(f'PATCH conflict attempt {attempt}: {e}', flush=True)
else:
    raise SystemExit('PATCH failed after retries')
json.dump({'commit': commit_sha, 'blobs': blobs},
          open(os.path.join(STAGE, 'commits.json'), 'w'))

# 3. read back: directory listing on main + per-blob md5
listing = gh_api.req('GET',
                     f'/repos/{OWNER}/{REPO}/contents/code/exp?ref=main')
by_name = {e['name']: e for e in listing if e['type'] == 'file'}
ok = True
for name, want in FILES.items():
    e = by_name.get(name)
    if e is None:
        print(f'READBACK {name}: MISSING from code/exp listing'); ok = False
        continue
    sha_match = (e['sha'] == blobs[name])
    blob = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/blobs/{e["sha"]}')
    raw = base64.b64decode(blob['content'])
    got = hashlib.md5(raw).hexdigest()
    md5_match = (got == want)
    print(f'READBACK {name}: size {len(raw)} blob_sha_match={sha_match} '
          f'md5={got} match={md5_match}', flush=True)
    ok = ok and sha_match and md5_match
print('WAREHOUSE_DONE commit=' + commit_sha + ' readback_ok=' + str(ok),
      flush=True)
