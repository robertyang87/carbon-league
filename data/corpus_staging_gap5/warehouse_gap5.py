#!/usr/bin/env python3
"""Warehouse gap5 fable fill (+21 -> 313) + fresh list 312->313 via Git Data API:
blobs -> tree(base_tree=HEAD tree) -> commit -> PATCH ref.
One commit. Blob shas checkpointed to blob_shas.json for resume.
Adapted from gap4 warehouse_gap4.py (proven pattern)."""
import sys, os, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
STAGE = '/home/hatch/workspace/carbon-live/corpus_staging_gap5'
LABEL = 'fable'
TARGET = 313
new = json.load(open(os.path.join(STAGE, 'gap5_diff.json')))
assert len(new) == 21, len(new)
ckpt_path = os.path.join(STAGE, 'blob_shas.json')
blobs = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

def save_ckpt():
    json.dump(blobs, open(ckpt_path, 'w'))

# 1. create blobs (checkpointed): 21 corpus files + fresh list file
jobs = [(f'{LABEL}/{n}', os.path.join(STAGE, LABEL, n)) for n in new]
jobs.append(('__list__', os.path.join(STAGE, 'fable_matches.json')))
for i, (key, path) in enumerate(jobs):
    if key in blobs:
        continue
    with open(path, 'rb') as f:
        data = f.read()
    r = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                   {'content': base64.b64encode(data).decode(),
                    'encoding': 'base64'})
    blobs[key] = r['sha']
    save_ckpt()
    print(f'blob {i+1}/{len(jobs)} {key} -> {r["sha"]}', flush=True)
print(f'blobs complete: {len(jobs)}', flush=True)

# 2. one commit, retry from fresh head on PATCH conflict
for attempt in range(5):
    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    base_tree = commit['tree']['sha']
    entries = [{'path': f'data/{LABEL}_corpus/{n}', 'mode': '100644',
                'type': 'blob', 'sha': blobs[f'{LABEL}/{n}']}
               for n in new]
    entries.append({'path': 'data/fable_matches.json', 'mode': '100644',
                    'type': 'blob', 'sha': blobs['__list__']})
    tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                      {'base_tree': base_tree, 'tree': entries})
    newc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                      {'message': f'{LABEL} corpus: gap5 fill +21 -> {TARGET}, list 312->313 (DGX 117 pull, war-room API warehouse)',
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
