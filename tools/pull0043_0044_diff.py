import sys, os, json
sys.path.insert(0, '/home/hatch/workspace')
import gh_api
OWNER, REPO = 'robertyang87', 'carbon-league'
B = '/home/hatch/workspace/carbon-incoming/pull0043_0044'
for corpus, man in [('krypton-prosearch_corpus', 'pull0043_krypton-prosearch.manifest'),
                    ('k4v3_corpus', 'pull0044_k4v3.manifest')]:
    entries = []
    for line in open(os.path.join(B, man)):
        line = line.strip()
        if not line: continue
        md5, rel = line.split(None, 1)
        entries.append((md5, rel.strip()))
    try:
        listing = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/data/{corpus}?ref=main')
        repo_names = {e['name'] for e in listing if e['type'] == 'file'}
    except Exception as ex:
        print(f'{corpus}: listing failed {ex}')
        repo_names = set()
    new = [(m, r) for m, r in entries if os.path.basename(r) not in repo_names]
    with open(os.path.join(B, f'new_{corpus}.txt'), 'w') as f:
        for m, r in new:
            f.write(f'{m}  {r}\n')
    print(f'{corpus}: manifest={len(entries)} repo={len(repo_names)} new={len(new)}', flush=True)
print('DIFF DONE', flush=True)
