import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

owner, repo, branch = 'robertyang87', 'carbon-league', 'sync/h24'
TAG = 'h24-1314'

def fetch(path):
    r = gh_api.get_file(owner, repo, path, ref=branch)
    return base64.b64decode(r['content']).decode('utf-8', errors='replace')

def listdir(path):
    r = gh_api.get_file(owner, repo, path, ref=branch)
    if isinstance(r, list):
        return [(d['name'], d['size'], d.get('sha','')[:8]) for d in r]
    return [('NOT_A_DIR', str(r)[:200], '')]

print('== TAG ==', TAG)
commits = gh_api.req('GET', f'/repos/{owner}/{repo}/commits?sha={branch}&per_page=3')
for c in commits:
    print('COMMIT', c['commit']['committer']['date'], c['sha'][:8], c['commit']['message'].splitlines()[0][:120])

top = listdir('synced/h24')
print('== synced/h24 count ==', len(top))
for name, size, sha in top:
    if name.endswith('_cal.out'):
        print('CAL', name, size, sha)
live = listdir('synced/h24/bigeval_live')
print('== synced/h24/bigeval_live count ==', len(live))

print('== m3s2r1_cal.out full ==')
print(fetch('synced/h24/m3s2r1_cal.out'))

txt = fetch('synced/h24/bigeval_live/m3s2r1_echo2.jsonl')
lines = [json.loads(l) for l in txt.splitlines() if l.strip()]
margins = [d['margin'] for d in lines]
wins = sum(1 for m in margins if m > 0)
print('== m3s2r1_echo2.jsonl ==')
print('n=', len(lines), 'wins=', wins, 'mean_margin=', round(sum(margins)/len(margins), 1))
print('first t=', lines[0]['t'], 'last t=', lines[-1]['t'], 'last seed=', lines[-1].get('seed'))
print('errs=', sum(1 for d in lines if d.get('err')))

print('== rate limit ==')
print(gh_api.req('GET', '/rate_limit')['resources']['core'])
