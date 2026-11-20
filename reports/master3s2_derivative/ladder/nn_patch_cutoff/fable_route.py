"""Trace the largest enemy delivery in v2's worst ten decisions against Fable v8."""
from pathlib import Path
import gzip
import hashlib
import json

HERE=Path(__file__).resolve().parent

def dist(a,b):
    return sum(min(abs(i-j),15-abs(i-j)) for i,j in zip(divmod(a,15),divmod(b,15)))

def dest(pos,d):
    y,x=divmod(pos,15)
    return 15*((y+{'UP':-1,'DOWN':1}.get(d,0))%15)+(x+{'RIGHT':1,'LEFT':-1}.get(d,0))%15

def main():
    audit=json.loads((HERE/'audit.json').read_bytes())['v2']
    loss=next(row for row in audit['losses'] if row['opponent']=='ClaudeFable5.2-v8')
    raw=HERE.parent/loss['replay'];assert hashlib.sha256(raw.read_bytes()).hexdigest()==loss['replay_sha256']
    steps=json.loads(gzip.decompress(raw.read_bytes()))['steps']
    book=json.loads(gzip.decompress((HERE/loss['ledger']).read_bytes()))
    own=loss['seat'];opp=1-own
    event=max((x for x in book['worst_windows']['10']['returns'] if x['seat']==opp),key=lambda x:x['amount'])
    uid=event['worker'];paid=event['t'];rows=[]
    for t in range(paid-11,paid):
        players=steps[t][0]['observation']['players']; orders=[x['action'] for x in steps[t+1]]
        pos,cargo,role=players[opp][2][uid];assert role=='COLLECTOR'
        cmd=orders[opp].get(uid,'NONE');goal=dest(pos,cmd)
        owned=players[own][2];nearest=min((dist(a[0],goal),wid) for wid,a in owned.items() if a[2]=='COLLECTOR')
        d,wid=nearest
        rows.append(dict(step=t,carrier=uid,cargo=cargo,pos=pos,order=cmd,next_cell=goal,
                         nearest_own_collector_to_actual_destination=dict(uid=wid,d=d,observed=owned[wid],order=orders[own].get(wid,'NONE')),
                         own_bases=list(players[own][1].values()),recruit_into_goal=goal in players[own][1].values()))
        assert steps[t+1][0]['observation']['players'][opp][2][uid][0]==goal
    assert event['amount']==475.26 and all(x['nearest_own_collector_to_actual_destination']['d']>1 and not x['recruit_into_goal'] for x in rows)
    (HERE/'fable_v8_route.json').write_text(json.dumps(dict(match=loss['match'],timestamp=paid,delivery=event['amount'],
                 participant=uid,rows=rows),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
