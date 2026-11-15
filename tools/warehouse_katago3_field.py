#!/usr/bin/env python3
"""Warehouse the six KataGo three-leg field per-game lines to carbon-league main.

Sources are the war-room exam_snapshots mirrors (h29/h31/h32/h33), each already
mirror-verified against its machine. kgattr_b is one leg split across two
machines: h29 part (62 rows, spot-29) then h31 part (138 rows, spot-31),
concatenated in run order; seed union verified complete/unique before upload.
Read-back verifies blob shas on main.
"""
import sys, json, base64, hashlib
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
BASE = '/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots'

def read(p):
    with open(p, 'rb') as f:
        return f.read()

parts = {
    'kgattr_a': [f'{BASE}/h29/kgattr_a.jsonl'],
    'kgattr_b': [f'{BASE}/h29/kgattr_b.jsonl', f'{BASE}/h31/kgattr_b.jsonl'],
    'kgduel_a': [f'{BASE}/h31/kgduel_a.jsonl'],
    'kgduel_b': [f'{BASE}/h32/kgduel_b.jsonl'],
    'kgref_a':  [f'{BASE}/h32/kgref_a.jsonl'],
    'kgref_b':  [f'{BASE}/h33/kgref_b.jsonl'],
}
provenance = {
    'kgattr_a': 'spot-29 (C1), mirror h29, 200 rows',
    'kgattr_b': 'spot-29 62 rows + spot-31 138 rows (C1->C2), mirrors h29+h31, concatenated in run order',
    'kgduel_a': 'spot-31 (C2), mirror h31, 200 rows',
    'kgduel_b': 'spot-32 (C2), mirror h32, 200 rows',
    'kgref_a':  'spot-32 (C2), mirror h32, 200 rows',
    'kgref_b':  'spot-33 (C2), mirror h33, 200 rows',
}

items = []
manifest_lines = [
    '# KataGo three-leg field (katago3) per-game lines - MANIFEST',
    '# field: seeds 985000-985199 per leg, 200 rows per leg, 1200 total; err 0 everywhere;',
    '# harness hh=354bcd1bc2dc8bc188c813f53d28c893 on every row.',
    '# md5  bytes  file  provenance',
]
for leg, ps in parts.items():
    data = b''.join(read(p) for p in ps)
    rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
    seeds = [r['seed'] for r in rows]
    assert len(rows) == 200 and set(seeds) == set(range(985000, 985200)) and len(set(seeds)) == 200, leg
    assert all(not r.get('err') for r in rows), leg
    assert all(r.get('hh') == '354bcd1bc2dc8bc188c813f53d28c893' for r in rows), leg
    md5 = hashlib.md5(data).hexdigest()
    print(f'{leg}: rows=200 md5={md5} bytes={len(data)}', flush=True)
    manifest_lines.append(f'{md5}  {len(data)}  {leg}.jsonl  {provenance[leg]}')
    items.append({'dst': f'data/katago3_field/{leg}.jsonl', 'data': data})

manifest = ('\n'.join(manifest_lines) + '\n').encode()
items.append({'dst': 'data/katago3_field/MANIFEST.md5', 'data': manifest})

def blob_sha(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()

for i in items:
    i['sha'] = blob_sha(i['data'])
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
                    {'message': 'katago3 field: six legs per-game lines (1200 rows, seeds 985000-985199) + MANIFEST (war room)',
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
