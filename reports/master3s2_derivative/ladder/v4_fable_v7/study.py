"""Verify all frozen ds-pro-v4/v2 vs the exact same Fable registration and summarize public actions."""
from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent;LADDER=HERE.parent;sys.path.insert(0,str(LADDER/'placement_refresh'))
import refresh
FABLE='2d4e4c772f057208b7121ed76dba4f33'
V4='16a3cd313829ffe78bd24ac9714b2400'
V2='ec94599e5182be7d35eb14bc288625cd'
def sha(data):return hashlib.sha256(data).hexdigest()
def pathid(p):return dict(path=str(p.relative_to(LADDER)),sha256=sha(p.read_bytes()))
def load(c,g):
 key=c['id']+'-'+str(g['game'])
 # Reuse artifacts in the existing main archive; store previously unseen matches in this folder.
 prior=sorted(p for p in LADDER.rglob(key+'.json.gz') if p.parent.name=='replays' and HERE not in p.parents)
 replay=prior[0] if prior else HERE/'replays'/(key+'.json.gz')
 if not replay.exists():
  replay.parent.mkdir(exist_ok=True)
  raw=refresh.get(refresh.BASE+'/replay/'+c['id']+'/'+str(g['game']))
  env,_,_=refresh.extract(raw);refresh.validate(env,c|g)
  replay.write_bytes(gzip.compress(refresh.canonical(env),compresslevel=6,mtime=0))
 env=refresh.read(replay);refresh.validate(env,c|g)
 return replay,env
def calc(m):
 me=V4 if V4 in (m['a'],m['b']) else V2;refresh.ID=me
 assert FABLE in (m['a'],m['b']) and me in (m['a'],m['b']) and m['rated'] and m['state']=='done' and len(m['result'])==1
 g=m['result'][0]; c=refresh.score(m,g);seat=c['seat']
 replay,o=load(m,g);key=c['match']+'-1.ledger.json.gz'
 candidates=[p for p in LADDER.rglob('*'+key) if HERE not in p.parents]
 if candidates:
  ledpath=candidates[0]; l=refresh.read(ledpath)
 else:
  ledpath=HERE/key
  if not ledpath.exists():
   refresh.compare.HERE=replay.parent.parent/'analysis'
   l=refresh.review.losing_game(dict(opponent=c['opponent'],match_id=c['match'],seat=seat,steps=c['steps']),replay)
   refresh.store(ledpath,l)
  else:l=refresh.read(ledpath)
 assert l['official']=={'matched':299,'first_divergence':None} and l['fast_verified']==299 and l['official_scores']==c['score'] and l['seat']==seat
 frames=o['steps']; order=[seat,1-seat]; recruits=[Counter(),Counter()];deaths=[Counter(),Counter()];initial=[{},{}]
 for t in range(299):
  for s in (0,1):
   b=frames[t][0]['observation']['players'][s];a=frames[t+1][0]['observation']['players'][s];cmd=frames[t+1][s]['action'] or {}
   for uid in b[2].keys()-a[2].keys():deaths[s][b[2][uid][2]]+=1
   for uid in a[2].keys()-b[2].keys(): recruits[s][a[2][uid][2]]+=1
   if t==0:initial[s]=cmd
 def window(t):
  result={}
  for w in (3,5,10):
   before=max(0,t-w); events=[e|{'arrival':j+1} for j in range(before,t) for e in l['receipts_by_transition'][j]]
   for e in events:
    uid=e['worker'];s=e['seat'];a=e['arrival']-1
    e['approach']=[dict(t=i,worker=frames[i][0]['observation']['players'][s][2].get(uid),action=(frames[i+1][s]['action'] or {}).get(uid,'NONE')) for i in sorted(set(max(0,a-j) for j in [10,5,3,1,0]))]
   result[w]=dict(start=before,end=t,from_margin=l['steps'][before]['margin'],to_margin=l['steps'][t]['margin'],events=events)
  return result
 rel=[r for r in l['all_streaks'] if r['sign']==-1];last=rel[-1]
 windows={t:window(t) for t in {r['end'] for r in l['worst_windows'].values()}|{last['start']}}
 return dict(card=c,terrain=sha(bytes(frames[0][0]['observation']['full_carbon'])),replay=pathid(replay),ledger=pathid(ledpath),sign_counts=l['sign_counts'],
           first_sustained=l['first_sustained_deficits'],last_lead=l['last_strict_lead'],last_deficit=last,worst=l['worst'],whole=l['whole'],phases=l['phases'],
           samples={t:l['steps'][t] for t in (0,10,25,50,75,100,125,150,175,200,225,250,275,299)},windows=windows,worker_births=[recruits[s] for s in order],
           worker_disappearances=[deaths[s] for s in order],first_orders=[initial[s] for s in order],planting=[{k:v for k,v in team.items() if k!='individual'} for team in l['planting']])
def main():
 matches=refresh.read(HERE/'scorecards.json');out=[calc(m) for m in matches]
 assert len({r['terrain'] for r in out})==len(out)==4
 refresh.store(HERE/'all_four.json',out)
 for r in out:
  c=r['card'];print(c['match'],c['margin'],r['sign_counts'],r['last_lead'],r['last_deficit'],r['whole']['difference'],r['worker_births'],r['worker_disappearances'],flush=True)
if __name__=='__main__':main()
