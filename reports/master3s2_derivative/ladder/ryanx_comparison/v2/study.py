"""Recompute map-specific provenance, pressure and pre-deposit approach distances from both RyanX games."""
import gzip, hashlib, json
from pathlib import Path
from probe import dest,dist,GAME,HERE,LADDER,MATCH,SHA
OPP='d1ab80fcc01010c949592984c617db30'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def summary(game,ledger,seat):
 frames=game['steps'];assert len(frames)==300 and game['statuses']==['DONE','DONE']
 pressure={}
 for side in (seat,1-seat):
  margin=ledger['whole']['end_margin'];bases=list(frames[0][0]['observation']['players'][side][1].values());clock=[]
  for t,row in enumerate(frames[:-1]):
   enemy=row[0]['observation']['players'][1-side][2];clock.append(min((dist(v[0],b) for v in enemy.values() for b in bases),default=999))
  pressure[str(side)]={str(k):sum(d<=k for d in clock) for k in range(3)}
 return dict(seat=seat,official_scores_subject_first=ledger['official_scores'],terrain=hashlib.sha256(bytes(frames[0][0]['observation']['full_carbon'])).hexdigest(),whole=ledger['whole'],phases=ledger['phases'],cash_counts=ledger['sign_counts'],last_lead=ledger['last_strict_lead'],base_pressure_frames_at_most=pressure,worst=ledger['worst'])
def main():
 assert h(GAME)==SHA
 v2=read(GAME);v2card=next(r for r in read(LADDER/'nn_patch_cutoff/matches.json')['versions']['v2']['added'] if r['match']==MATCH)
 p=LADDER/'nn_patch_cutoff'/f'v2_{MATCH}-1.ledger.json.gz';j=read(p)
 v4id='1c80b916b203238a43267baeabc88fbe'
 v4p=LADDER/f'v4_observations/replays/{v4id}-1.json.gz'; lp=LADDER/f'v4_observations/{v4id}-1.ledger.json.gz';v4=read(v4p);l4=read(lp)
 ranked=read(LADDER/'v4_observations/pages.json.gz')
 matches=[r for page in ranked for r in page['data']['matches'] if r['id']==v4id]
 assert v2card['opponent_id']==OPP and len(matches)==1 and OPP in {matches[0]['a'],matches[0]['b']}
 assert l4['opponent']==j['opponent']==v2card['opponent']
 sources=[GAME,p,v4p,lp];decoded={z:{'sha256':h(f),'path':str(f.relative_to(LADDER))} for z,f in zip(('v2_replay','v2_ledger','v4_replay','v4_ledger'),sources)}
 refs= [(55,'player-1-worker-11'),(206,'player-1-worker-20'),(213,'player-1-worker-24')]
 witnesses=[]
 for root,uid in refs:
  rows=[]
  for t in (root-10,root-5,root-3,root-2,root-1,root):
   players=v2['steps'][t][0]['observation']['players'];target=players[1][2][uid]
   enemy_action=(v2['steps'][t+1][1]['action'] or {}).get(uid,'NONE')
   nxt=dest(target[0],enemy_action)
   own=sorted((dist(w[0],nxt),i,w[0],w[1],(v2['steps'][t+1][0]['action'] or {}).get(i,'NONE')) for i,w in players[0][2].items() if w[2]=='COLLECTOR' and w[1]<target[1])
   rows.append(dict(t=t,enemy=target,enemy_action=enemy_action,observed_next_cell=nxt,nearest_lighter_own=own[:1]))
  witnesses.append(dict(t=root,enemy=uid,bank=j['receipts_by_transition'][root],rows=rows))
 out=dict(match_v2=MATCH,match_v4=v4id,opponent_id=OPP,sources=decoded,v2=summary(v2,j,0),v4=summary(v4,l4,1),witnesses=witnesses)
 assert out['v2']['terrain']!=out['v4']['terrain']; (HERE/'cash_routes.json').write_text(json.dumps(out,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
if __name__=='__main__':main()
