import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

owner, repo, branch = 'robertyang87', 'carbon-league', 'sync/h24'

def fetch(path):
    r = gh_api.get_file(owner, repo, path, ref=branch)
    return base64.b64decode(r['content']).decode('utf-8', errors='replace')

def listdir(path):
    r = gh_api.get_file(owner, repo, path, ref=branch)
    if isinstance(r, list):
        return [(d['name'], d['size']) for d in r]
    return [('NOT_A_DIR', str(r)[:200])]

commits = gh_api.req('GET', f'/repos/{owner}/{repo}/commits?sha={branch}&per_page=3')
print('== LATEST COMMITS ==')
for c in commits:
    print(c['commit']['committer']['date'], c['sha'], c['commit']['message'].splitlines()[0][:120])

print('== synced/h24 listing (count) ==')
top = listdir('synced/h24')
print('n_items=', len(top))
for name, size in top:
    if name.endswith('_cal.out') or name.endswith('.out'):
        print('OUT:', name, size)

print('== m3s2r1_cal.out full ==')
try:
    print(fetch('synced/h24/m3s2r1_cal.out'))
except Exception as e:
    print('ERROR', e)

print('== m3s2r1_echo2.jsonl stats ==')
try:
    txt = fetch('synced/h24/bigeval_live/m3s2r1_echo2.jsonl')
    lines = [json.loads(l) for l in txt.splitlines() if l.strip()]
    margins = [d['margin'] for d in lines]
    wins = sum(1 for m in margins if m > 0)
    print('n=', len(lines), 'wins=', wins, 'mean_margin=', round(sum(margins)/len(margins), 1))
    print('first t=', lines[0]['t'], 'last t=', lines[-1]['t'], 'last seed=', lines[-1].get('seed'))
    print('errs=', sum(1 for d in lines if d.get('err')))
except Exception as e:
    print('ERROR', e)

print('== rate limit ==')
try:
    print(gh_api.req('GET', '/rate_limit')['resources']['core'])
except Exception as e:
    print('ERROR', e)
