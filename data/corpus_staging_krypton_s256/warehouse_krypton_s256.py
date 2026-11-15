#!/usr/bin/env python3
"""Warehouse Krypton-Search256-v1 corpus insurance (new corpus +111 -> 111)
via Git Data API: blobs -> tree(base_tree=HEAD tree) -> commit -> PATCH ref.
One commit, corpus files only (data/krypton-search256_matches.json untouched).
Blob shas checkpointed to blob_shas.json for resume."""
import sys, os, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
STAGE = '/home/hatch/workspace/carbon-live/corpus_staging_krypton_s256'
LABEL = 'krypton-search256'
TARGET = 111
ckpt_path = os.path.join(STAGE, 'blob_shas.json')
blobs = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

def save_ckpt():
    json.dump(blobs, open(ckpt_path, 'w'))

# manifest-derived file list (deterministic order)
names = []
for line in open(os.path.join(STAGE, 'pull0033_krypton-search256.manifest')):
    parts = line.split()
    if len(parts) == 2:
        names.append(os.path.basename(parts[1]))
names = sorted(set(names))
assert len(names) == 111, f'manifest names {len(names)} != 111'

# 1. create blobs (checkpointed)
for i, name in enumerate(names):
    key = f'{LABEL}/{name}'
    if key in blobs:
        continue
    with open(os.path.join(STAGE, 'corpus', name), 'rb') as f:
        data = f.read()
    r = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                   {'content': base64.b64encode(data).decode(),
                    'encoding': 'base64'})
    blobs[key] = r['sha']
    save_ckpt()
    if (i + 1) % 10 == 0:
        print(f'blobs {i+1}/{len(names)}', flush=True)
print(f'blobs complete: {len(names)}', flush=True)

# 2. one commit, retry from fresh head on PATCH conflict
for attempt in range(5):
    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    base_tree = commit['tree']['sha']
    entries = [{'path': f'data/{LABEL}_corpus/{name}', 'mode': '100644',
                'type': 'blob', 'sha': blobs[f'{LABEL}/{name}']}
               for name in names]
    tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                      {'base_tree': base_tree, 'tree': entries})
    newc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                      {'message': f'{LABEL} corpus: new corpus +111 -> 111 (DGX 117 pull, war-room API warehouse; matches list untouched)',
                       'tree': tree['sha'], 'parents': [head]})
    try:
        gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main',
                   {'sha': newc['sha'], 'force': False})
        print(f'COMMIT {newc["sha"]} (parent {head})', flush=True)
        json.dump({LABEL: newc['sha']}, open(os.path.join(STAGE, 'commits.json'), 'w'))
        print('WAREHOUSE_DONE', newc['sha'], flush=True)
        break
    except Exception as e:
        print(f'PATCH conflict attempt {attempt}: {e}', flush=True)
else:
    raise SystemExit('PATCH failed after retries')
