#!/usr/bin/env python3
"""Batch-2 warehouse: DGX staging -> carbon-league main. Same mechanics as batch 1."""
import sys, os, hashlib, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
G = '/home/hatch/workspace/carbon-incoming/sync_20261115/DGX'
pairs = [
    (f'{G}/k0_calib_driver.py', 'tools/k0_calib_driver.py'),
    (f'{G}/roots_calib.json', 'data/k0_calib/roots_calib.json'),
    (f'{G}/gate_summary.json', 'reports/KATAGO_GATE_20261114/gate_summary.json'),
    (f'{G}/gate_examples.json', 'reports/KATAGO_GATE_20261114/gate_examples.json'),
    (f'{G}/gate_results.jsonl', 'reports/KATAGO_GATE_20261114/gate_results.jsonl'),
    (f'{G}/merge_current.json', 'reports/K0_MERGE_JOIN_20261115/merge_current.json'),
    (f'{G}/merge_manifest.json', 'reports/K0_MERGE_JOIN_20261115/merge_manifest.json'),
    (f'{G}/merge_meta.json', 'reports/K0_MERGE_JOIN_20261115/merge_meta.json'),
    (f'{G}/join_current.json', 'reports/K0_MERGE_JOIN_20261115/join_current.json'),
    (f'{G}/join_manifest.json', 'reports/K0_MERGE_JOIN_20261115/join_manifest.json'),
    (f'{G}/join_meta.json', 'reports/K0_MERGE_JOIN_20261115/join_meta.json'),
    (f'{G}/it74_master2.jsonl', 'data/it74_test/it74_master2.jsonl'),
    (f'{G}/it74_echo2.jsonl', 'data/it74_test/it74_echo2.jsonl'),
    (f'{G}/cnet2_search_vleaf.py', 'code/exp/cnet2_search_vleaf.py'),
    (f'{G}/katago_vleaf_value.py', 'code/exp/vleaf_value_pack.py'),
    (f'{G}/katago_vleaf_const.py', 'code/exp/vleaf_const_pack.py'),
]

def blob_sha(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()

items = []
for src, dst in pairs:
    data = open(src, 'rb').read()
    items.append({'src': src, 'dst': dst, 'data': data, 'sha': blob_sha(data)})
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
                    {'message': 'sync 2026-11-15 batch2: k0 calib driver+rootset, katago gate results, k0 merge/join identity, it74 per-game lines (war room)',
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
