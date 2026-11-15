#!/usr/bin/env python3
"""Warehouse gap4 fable6 fill (+22 -> 225) via Git Data API:
blobs -> tree(base_tree=HEAD tree) -> commit -> PATCH ref.
One commit. Blob shas checkpointed to blob_shas.json for resume."""
import sys, os, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
STAGE = '/home/hatch/workspace/carbon-live/corpus_staging_gap4'
LABEL = 'fable6'
TARGET = 225
diff = json.load(open(os.path.join(STAGE, 'gap4_diff.json')))
ckpt_path = os.path.join(STAGE, 'blob_shas.json')
blobs = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

def save_ckpt():
    json.dump(blobs, open(ckpt_path, 'w'))

# 1. create blobs (checkpointed)
missing = diff['missing']
for i, name in enumerate(missing):
    key = f'{LABEL}/{name}'
    if key in blobs:
        continue
    with open(os.path.join(STAGE, LABEL, name), 'rb') as f:
        data = f.read()
    r = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                   {'content': base64.b64encode(data).decode(),
                    'encoding': 'base64'})
    blobs[key] = r['sha']
    save_ckpt()
    if (i + 1) % 10 == 0:
        print(f'blobs {i+1}/{len(missing)}', flush=True)
print(f'blobs complete: {len(missing)}', flush=True)

# 2. one commit, retry from fresh head on PATCH conflict
for attempt in range(5):
    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    base_tree = commit['tree']['sha']
    entries = [{'path': f'data/{LABEL}_corpus/{name}', 'mode': '100644',
                'type': 'blob', 'sha': blobs[f'{LABEL}/{name}']}
               for name in missing]
    tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                      {'base_tree': base_tree, 'tree': entries})
    newc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                      {'message': f'{LABEL} corpus: gap4 fill +{len(missing)} -> {TARGET} (DGX 117 pull, war-room API warehouse)',
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
