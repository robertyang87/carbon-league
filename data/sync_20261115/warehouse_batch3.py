#!/usr/bin/env python3
"""Batch-3 warehouse: k0 merge/join driver logs -> carbon-league main."""
import sys, hashlib, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
G = '/home/hatch/workspace/carbon-incoming/sync_20261115/DGX'
pairs = [
    (f'{G}/k0_join_prod.log', 'reports/K0_MERGE_JOIN_20261115/k0_join_prod.log'),
    (f'{G}/k0_merge_driver.log', 'reports/K0_MERGE_JOIN_20261115/k0_merge_driver.log'),
]

def blob_sha(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()

items = []
for src, dst in pairs:
    data = open(src, 'rb').read()
    items.append({'dst': dst, 'data': data, 'sha': blob_sha(data)})
print(f'ITEMS {len(items)}', flush=True)

def head_and_tree():
    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    tree_sha = commit['tree']['sha']
    tree = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/trees/{tree_sha}?recursive=1')
    return head, tree_sha, {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'}

head, tree_sha, cur = head_and_tree()
print('HEAD', head, flush=True)
todo = [i for i in items if cur.get(i['dst']) != i['sha']]
print(f'TODO {len(todo)} SKIP_IDENTICAL {len(items) - len(todo)}', flush=True)
for i in todo:
    r = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                   {'content': base64.b64encode(i['data']).decode(), 'encoding': 'base64'})
    assert r['sha'] == i['sha']
    print('blob ok', i['dst'], flush=True)

newhead = None
for attempt in range(4):
    entries = [{'path': i['dst'], 'mode': '100644', 'type': 'blob', 'sha': i['sha']} for i in todo]
    nt = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees', {'base_tree': tree_sha, 'tree': entries})
    nc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                    {'message': 'sync 2026-11-15 batch3: k0 merge/join driver logs (war room)',
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
