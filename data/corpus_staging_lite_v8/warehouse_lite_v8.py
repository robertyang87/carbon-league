#!/usr/bin/env python3
"""Warehouse Lite-V8 corpus insurance fill (+20 -> 241) via Git Data API:
blobs -> tree(base_tree=HEAD tree) -> commit -> PATCH ref. One commit.
Corpus files ONLY: the fresh 117 list is byte-identical to main's
data/lite-v8_matches.json (md5 769bd38e77e620ffcbdabdabb37a28da), so the
list file is NOT touched. Blob shas checkpointed for resume.
Adapted from gap5 warehouse_gap5.py (proven pattern)."""
import sys, os, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
STAGE = '/home/hatch/workspace/carbon-live/corpus_staging_lite_v8'
LABEL = 'lite-v8'
TARGET = 241
new = json.load(open(os.path.join(STAGE, 'lite_v8_diff.json')))
assert len(new) == 20, len(new)
ckpt_path = os.path.join(STAGE, 'blob_shas.json')
blobs = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

def save_ckpt():
    json.dump(blobs, open(ckpt_path, 'w'))

jobs = [(f'{LABEL}/{n}', os.path.join(STAGE, LABEL, n)) for n in new]
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

for attempt in range(5):
    ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
    head = ref['object']['sha']
    commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
    base_tree = commit['tree']['sha']
    entries = [{'path': f'data/{LABEL}_corpus/{n}', 'mode': '100644',
                'type': 'blob', 'sha': blobs[f'{LABEL}/{n}']}
               for n in new]
    tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                      {'base_tree': base_tree, 'tree': entries})
    newc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                      {'message': f'{LABEL} corpus: insurance fill +20 -> {TARGET} (DGX 117 pull, war-room API warehouse; matches list untouched, byte-identical)',
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
