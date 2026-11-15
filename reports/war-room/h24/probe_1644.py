import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

owner, repo, branch = 'robertyang87', 'carbon-league', 'sync/h24'

def fetch(path):
    r = gh_api.get_file(owner, repo, path, ref=branch)
    return base64.b64decode(r['content']).decode('utf-8', errors='replace')

# 1. recent commits on sync/h24
commits = gh_api.req('GET', f'/repos/{owner}/{repo}/commits?sha={branch}&per_page=12')
print('== COMMITS sync/h24 ==')
for c in commits:
    print(c['commit']['committer']['date'], c['sha'][:8], c['commit']['message'].splitlines()[0][:120])

# 2. list dirs
for d in ['synced/h24', 'synced/h24/bigeval_live']:
    try:
        items = gh_api.req('GET', f'/repos/{owner}/{repo}/contents/{d}?ref={branch}')
        print(f'== DIR {d} ==')
        for it in items:
            print(it['type'], it['name'], it.get('size'))
    except Exception as e:
        print(f'== DIR {d} ERROR ==', e)

# 3. cal.out files full text
for name in ['m3s2r1_cal.out']:
    try:
        print(f'== {name} ==')
        print(fetch(f'synced/h24/{name}'))
    except Exception as e:
        print(f'== {name} ERROR ==', e)

# 4. E0 live file stats
try:
    txt = fetch('synced/h24/bigeval_live/m3s2r1_e0.jsonl')
    lines = [json.loads(l) for l in txt.splitlines() if l.strip()]
    margins = [d['margin'] for d in lines]
    wins = sum(1 for m in margins if m > 0)
    print('== m3s2r1_e0.jsonl ==')
    print('n=', len(lines), 'wins=', wins, 'mean_margin=', round(sum(margins)/len(margins), 1))
    print('first t=', lines[0]['t'], 'last t=', lines[-1]['t'])
    print('errs=', sum(1 for d in lines if d.get('err')))
    from datetime import datetime
    t0 = datetime.strptime(lines[0]['t'], '%H:%M:%S'); t1 = datetime.strptime(lines[-1]['t'], '%H:%M:%S')
    span_min = (t1-t0).total_seconds()/60
    print('span_min=', round(span_min,1), 'games/min=', round(len(lines)/span_min,2) if span_min>0 else None)
except Exception as e:
    print('== E0 live ERROR ==', e)
