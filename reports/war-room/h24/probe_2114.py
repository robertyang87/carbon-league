import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

owner, repo, branch = 'robertyang87', 'carbon-league', 'sync/h24'
MARK = 'h24-2114'

def fetch(path):
    r = gh_api.get_file(owner, repo, path, ref=branch)
    return base64.b64decode(r['content']).decode('utf-8', errors='replace')

print('MARK', MARK)
# 1. recent commits on sync/h24
commits = gh_api.req('GET', f'/repos/{owner}/{repo}/commits?sha={branch}&per_page=5')
print('== COMMITS sync/h24 ==')
for c in commits:
    print(c['commit']['committer']['date'], c['sha'][:8], c['commit']['message'].splitlines()[0][:120])

# 2. list dirs
for d in ['synced/h24', 'synced/h24/bigeval_live']:
    try:
        items = gh_api.req('GET', f'/repos/{owner}/{repo}/contents/{d}?ref={branch}')
        print(f'== DIR {d} == n={len(items)}')
        for it in items:
            if it['name'].endswith('_cal.out') or it['name'].endswith('.out'):
                print(it['type'], it['name'], it.get('size'), it.get('sha', '')[:8])
    except Exception as e:
        print(f'== DIR {d} ERROR ==', e)

# 3. cal.out files full text
for name in ['m3s2r1_cal.out']:
    try:
        txt = fetch(f'synced/h24/{name}')
        print(f'== {name} == len={len(txt)}')
        print(txt)
    except Exception as e:
        print(f'== {name} ERROR ==', e)

# 4. live file line counts (known files only)
for name in ['m3s2r1_e0.jsonl', 'm3s2r1_master2.jsonl']:
    try:
        txt = fetch(f'synced/h24/bigeval_live/{name}')
        lines = [l for l in txt.splitlines() if l.strip()]
        print(f'== bigeval_live/{name} == lines={len(lines)}')
        if lines:
            d = json.loads(lines[-1])
            print('last:', {k: d.get(k) for k in ('t', 'margin', 'seed', 'err') if k in d})
    except Exception as e:
        print(f'== bigeval_live/{name} ERROR ==', e)

# 5. rate limit
try:
    rl = gh_api.req('GET', '/rate_limit')
    print('== RATE == core remaining', rl['resources']['core']['remaining'], 'reset', rl['resources']['core']['reset'])
except Exception as e:
    print('== RATE ERROR ==', e)
