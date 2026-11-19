"""Check new index entries against the archived cohort and score frozen predictions on new Master6 games."""
from concurrent.futures import ThreadPoolExecutor
import sys,json
import numpy as np
from fit import HERE,ACTOR,arrays,stats,take,fetch
sys.path.insert(0,str(ACTOR/'planter_terminal_choice_dev'))
import ladder_delta as old
A='595633d75'
B='a5867c07b'
paths=[p for p in fetch.git('diff','--name-only',A,B,'--','data').decode().splitlines() if fetch.sourcefile(p)]
known={};now={}
for p in paths:
    known.update(old.cards(A,p));now.update(old.cards(B,p))
roster=set()
for p in ((ACTOR/'planter_terminal_choice_dev/ladder_delta.json'),(ACTOR/'planter_terminal_choice_dev/third_stream/ladder_delta.json')):
    roster.update(g['id'] for g in json.loads(p.read_bytes())['games'])
top=ACTOR.parent/'all_top5'
for p in [top/'manifest.json',*sorted(top.glob('appendix_*/manifest.json'))]:
    roster.update(g['id'] for g in json.loads(p.read_bytes())['games'])
keys=sorted(now.keys()-known.keys()-roster)
with ThreadPoolExecutor(2) as pool:games=list(pool.map(old.recover,(now[k] for k in keys)))
models=np.load(HERE/'weights.npz')
for g in games:
    if 'claude-AlphaCarbon-Master6' not in g['names']:continue
    seat=g['names'].index('claude-AlphaCarbon-Master6');data=(fetch.CACHE/'replays'/(g['id']+'.json.gz')).read_bytes()
    env,_,_=fetch.extract(data);rows=[]
    for f,n in zip(env['steps'],env['steps'][1:]):
        take(f[0]['observation']['players'],seat,n[seat].get('action') or {},rows,g['verified']['random_seed'])
    if rows:
        x,y,_=arrays(rows)
        g['forecast']={k:float(stats(models[k],x,y)[0].mean()) for k in ('weights','baseline')}
    g['carrier_rows']=len(rows)
(HERE/'intake.json').write_bytes(fetch.canonical(dict(start=A,end=B,source_indices=paths,games=games)))
print(len(games),sum(g.get('carrier_rows',0) for g in games))
