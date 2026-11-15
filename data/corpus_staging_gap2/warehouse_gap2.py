#!/usr/bin/env python3
"""Warehouse gap2 corpus fills (hakimi3pro 17 / fable6 19 / k4v2 202) via
Git Data API: blobs -> tree(base_tree=HEAD tree) -> commit -> PATCH ref.
One commit per label. Blob shas checkpointed to blob_shas.json for resume."""
import sys, os, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
STAGE = '/home/hatch/workspace/carbon-live/corpus_staging_gap2'
TARGET = {'hakimi3pro': 161, 'fable6': 203, 'k4v2': 306}
diff = json.load(open('/tmp/gap2_diff.json'))
ckpt_path = os.path.join(STAGE, 'blob_shas.json')
blobs = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

def save_ckpt():
    json.dump(blobs, open(ckpt_path, 'w'))

# 1. create blobs (checkpointed)
for label in ['hakimi3pro', 'fable6', 'k4v2']:
    missing = diff[label]['missing']
    for i, name in enumerate(missing):
        key = f'{label}/{name}'
        if key in blobs:
            continue
        with open(os.path.join(STAGE, label, name), 'rb') as f:
            data = f.read()
        r = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/blobs',
                       {'content': base64.b64encode(data).decode(),
                        'encoding': 'base64'})
        blobs[key] = r['sha']
        save_ckpt()
        if (i + 1) % 50 == 0:
            print(f'[{label}] blobs {i+1}/{len(missing)}', flush=True)
    print(f'[{label}] blobs complete: {len(missing)}', flush=True)

# 2. one commit per label, retry from fresh head on PATCH conflict
commits = {}
for label in ['hakimi3pro', 'fable6', 'k4v2']:
    missing = diff[label]['missing']
    for attempt in range(5):
        ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
        head = ref['object']['sha']
        commit = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/{head}')
        base_tree = commit['tree']['sha']
        entries = [{'path': f'data/{label}_corpus/{name}', 'mode': '100644',
                    'type': 'blob', 'sha': blobs[f'{label}/{name}']}
                   for name in missing]
        tree = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/trees',
                          {'base_tree': base_tree, 'tree': entries})
        newc = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
                          {'message': f'{label} corpus: gap2 fill +{len(missing)} -> {TARGET[label]} (DGX 117 pull, war-room API warehouse)',
                           'tree': tree['sha'], 'parents': [head]})
        try:
            gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main',
                       {'sha': newc['sha'], 'force': False})
            commits[label] = newc['sha']
            print(f'[{label}] COMMIT {newc["sha"]} (parent {head})', flush=True)
            break
        except Exception as e:
            print(f'[{label}] PATCH conflict attempt {attempt}: {e}', flush=True)
    else:
        raise SystemExit(f'{label}: PATCH failed after retries')
json.dump(commits, open(os.path.join(STAGE, 'commits.json'), 'w'))
print('WAREHOUSE_DONE', json.dumps(commits), flush=True)
