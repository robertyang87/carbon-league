#!/usr/bin/env python3
"""Warehouse m9xj600 corpus (DGX direct pull, 52 replays) -> carbon-league main.
Pattern from sync_20261115/warehouse_batch1.py: blob-sha dedupe, 422 re-parent,
read-back verify. Verifies each replay md5 against DGX manifest first."""
import sys, os, json, base64, hashlib
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
S = '/home/hatch/workspace/carbon-incoming/pull_m9xj600'
MANIFEST = f'{S}/pull_m9xj600.manifest'

def blob_sha(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()

def md5(data):
    return hashlib.md5(data).hexdigest()

# 1. verify against DGX manifest
want = {}
for line in open(MANIFEST):
    parts = line.split()
    if len(parts) >= 2:
        want[os.path.basename(parts[1])] = parts[0]
files = sorted(os.listdir(f'{S}/replays'))
assert len(files) == 52, f'replay count {len(files)} != 52'
bad = []
items = []
for fn in files:
    data = open(f'{S}/replays/{fn}', 'rb').read()
    if want.get(fn) != md5(data):
        bad.append(fn)
    items.append({'src': f'{S}/replays/{fn}', 'dst': f'data/m9xj600_corpus/{fn}',
                  'data': data, 'sha': blob_sha(data)})
print(f'MANIFEST_CHECK bad={len(bad)} {bad}', flush=True)
assert not bad, bad
mdata = open(MANIFEST, 'rb').read()
items.append({'src': MANIFEST, 'dst': 'data/m9xj600_corpus/MANIFEST.md5',
              'data': mdata, 'sha': blob_sha(mdata)})
print(f'ITEMS {len(items)} TOTAL_BYTES {sum(len(i["data"]) for i in items)}', flush=True)

def head_and_tree():
    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    tree_sha = commit['tree']['sha']
    tree = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/trees/{tree_sha}?recursive=1')
    m = {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'}
    return head, tree_sha, m

head, tree_sha, cur = head_and_tree()
print('HEAD', head, flush=True)
todo = [i for i in items if cur.get(i['dst']) != i['sha']]
print(f'TODO {len(todo)} SKIP_IDENTICAL {len(items) - len(todo)}', flush=True)

for i in todo:
    for attempt in range(2):
        try:
            r = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                           {'content': base64.b64encode(i['data']).decode(), 'encoding': 'base64'})
            break
        except Exception as e:
            if attempt == 1:
                raise
            print('blob retry', i['dst'], e, flush=True)
    assert r['sha'] == i['sha'], (i['dst'], r['sha'], i['sha'])
    print('blob ok', i['dst'], flush=True)

newhead = None
for attempt in range(4):
    entries = [{'path': i['dst'], 'mode': '100644', 'type': 'blob', 'sha': i['sha']} for i in todo]
    nt = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees', {'base_tree': tree_sha, 'tree': entries})
    nc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                    {'message': 'm9xj600 corpus 52 games (DGX direct pull, frozen list 821bfd91; ladder-replay-pull blocked): data/m9xj600_corpus/ + MANIFEST.md5 (war room)',
                     'tree': nt['sha'], 'parents': [head]})
    try:
        gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main', {'sha': nc['sha']})
        newhead = nc['sha']
        break
    except Exception as e:
        print('patch failed, re-parenting:', e, flush=True)
        head, tree_sha, cur = head_and_tree()
        todo = [i for i in todo if cur.get(i['dst']) != i['sha']]
        if not todo:
            newhead = head
            break
if newhead is None:
    raise SystemExit('FAILED: ref patch')
print('NEW_HEAD', newhead, flush=True)

commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{newhead}')
tree = gh_api.req('GET', f"/repos/{OWNER}/{REPO}/git/trees/{commit['tree']['sha']}?recursive=1")
m = {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'}
bad = [i['dst'] for i in items if m.get(i['dst']) != i['sha']]
print('READBACK', 'ALL_OK' if not bad else 'BAD:' + ','.join(bad), flush=True)
