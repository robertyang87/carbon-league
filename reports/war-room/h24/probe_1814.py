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

commits = gh_api.req('GET', f'/repos/{owner}/{repo}/commits?sha={branch}&per_page=5')
print('== LATEST COMMITS ==')
for c in commits:
    print(c['commit']['committer']['date'], c['sha'][:8], c['commit']['message'].splitlines()[0][:120])

print('== synced/h24 listing ==')
for name, size in listdir('synced/h24'):
    print(name, size)
print('== synced/h24/bigeval_live listing ==')
for name, size in listdir('synced/h24/bigeval_live'):
    print(name, size)

# all *_cal.out at root
for name, size in listdir('synced/h24'):
    if name.endswith('_cal.out') or name.endswith('.out'):
        try:
            print(f'== synced/h24/{name} ==')
            print(fetch(f'synced/h24/{name}'))
        except Exception as e:
            print(f'== {name} ERROR ==', e)

# E0 live file stats
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
    # last 15 games stats
    tail = margins[-15:]
    print('last15 n=', len(tail), 'wins=', sum(1 for m in tail if m > 0), 'mean=', round(sum(tail)/len(tail),1))
except Exception as e:
    print('== E0 live ERROR ==', e)

# echo2 smoke live
try:
    txt = fetch('synced/h24/bigeval_live/echo2_smoke.jsonl')
    lines = [json.loads(l) for l in txt.splitlines() if l.strip()]
    print('== echo2_smoke.jsonl == n=', len(lines))
except Exception as e:
    print('== echo2 smoke live ERROR ==', e)
