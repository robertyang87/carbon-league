import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api
r = gh_api.get_file('robertyang87', 'carbon-league', 'synced/h24/bigeval_live/m3s2r1_e0.jsonl', ref='sync/h24')
txt = base64.b64decode(r['content']).decode('utf-8', errors='replace')
lines = [json.loads(l) for l in txt.splitlines() if l.strip()]
margins = [d['margin'] for d in lines]
wins = sum(1 for m in margins if m > 0)
print('n=', len(lines), 'wins=', wins, 'mean_margin=', round(sum(margins)/len(margins), 1))
print('first t=', lines[0]['t'], 'last t=', lines[-1]['t'])
print('errs=', sum(1 for d in lines if d.get('err')))
# pace: games per minute from first to last
from datetime import datetime
t0 = datetime.strptime(lines[0]['t'], '%H:%M:%S'); t1 = datetime.strptime(lines[-1]['t'], '%H:%M:%S')
span_min = (t1-t0).total_seconds()/60
print('span_min=', round(span_min,1), 'games/min=', round(len(lines)/span_min,2) if span_min>0 else None)
