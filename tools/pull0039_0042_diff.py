#!/usr/bin/env python3
"""Diff DGX manifests for pull0039-0042 against repo main corpus dirs.
Outputs per-corpus NEW file lists (repo-relative paths) to staging dir.
Read-only vs GitHub (4-5 GET calls).
"""
import sys, os, json
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

OWNER, REPO = 'robertyang87', 'carbon-league'
OUT = '/home/hatch/workspace/carbon-incoming/pull0039_0042'
os.makedirs(OUT, exist_ok=True)

CORPORA = {
    'fable6_corpus': 'pull0039_fable6.manifest',
    'lite-v8_corpus': 'pull0040_lite-v8.manifest',
    'krypton-search256-v2_corpus': 'pull0041_krypton-search256-v2.manifest',
    'krypton-cloudsearch_corpus': 'pull0042_krypton-cloudsearch-v1.manifest',
}

result = {}
for corpus, man in CORPORA.items():
    man_path = os.path.join(OUT, man)
    entries = []  # (md5, rel)
    for line in open(man_path):
        line = line.strip()
        if not line:
            continue
        md5, rel = line.split(None, 1)
        entries.append((md5, rel.strip()))
    try:
        listing = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/data/{corpus}?ref=main')
        repo_names = {e['name'] for e in listing if e['type'] == 'file'}
    except Exception as ex:
        print(f'{corpus}: repo listing failed ({ex}); treating as empty')
        repo_names = set()
    new = [(md5, rel) for md5, rel in entries if os.path.basename(rel) not in repo_names]
    with open(os.path.join(OUT, f'new_{corpus}.txt'), 'w') as f:
        for md5, rel in new:
            f.write(f'{md5}  {rel}\n')
    result[corpus] = {'manifest': len(entries), 'repo': len(repo_names), 'new': len(new)}
    print(f'{corpus}: manifest={len(entries)} repo={len(repo_names)} new={len(new)}', flush=True)

json.dump(result, open(os.path.join(OUT, 'diff_summary.json'), 'w'), indent=1)
print('DIFF DONE', flush=True)
