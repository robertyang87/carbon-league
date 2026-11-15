import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

owner, repo, branch = 'robertyang87', 'carbon-league', 'sync/h24'

# 1. recent commits on sync/h24
commits = gh_api.req('GET', f'/repos/{owner}/{repo}/commits?sha={branch}&per_page=12')
print('== COMMITS sync/h24 ==')
for c in commits:
    print(c['commit']['committer']['date'], c['sha'][:8], c['commit']['message'].splitlines()[0][:100])

# 2. list synced/h24 and bigeval_live
for d in ['synced/h24', 'synced/h24/bigeval_live']:
    try:
        items = gh_api.req('GET', f'/repos/{owner}/{repo}/contents/{d}?ref={branch}')
        print(f'== DIR {d} ==')
        for it in items:
            print(it['type'], it['name'], it.get('size'))
    except Exception as e:
        print(f'== DIR {d} ERROR ==', e)
