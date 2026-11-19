"""Census endgame return choices, with one first opportunity per carrier on each top-five appearance."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,sys,gzip
from rescue import HERE,_TO,_D,_DIRECTION
sys.path.insert(0,str(HERE.parent/'planter_terminal_choice_dev'))
import ladder_delta as ladder
F=ladder.fetch
REV='afc3951b2'

def catalog():
    paths=[p for p in F.git('ls-tree','-r','--name-only',REV,'data').decode().splitlines() if F.sourcefile(p)]
    records={}
    with ThreadPoolExecutor(3) as pool:
        for cards in pool.map(lambda p:ladder.cards(REV,p),paths):records.update(cards)
    return records

def examine(item):
    gid,card=item
    path=F.CACHE/'replays'/(gid+'.json.gz')
    if not path.exists():ladder.recover(card)
    env,_,_=F.extract(path.read_bytes());proof=F.validate(env,card)
    steps=env['steps'];rows=[]
    for seat,player in enumerate(proof['seat_model_names']):
        if player not in F.NAMES:continue
        seen=set()
        for t in range(280,min(len(steps)-1,299)):
            obs=steps[t][0]['observation'];own=obs['players'][seat];enemy=obs['players'][1-seat]
            bases=list(own[1].values());left=299-t
            if not bases:continue
            for uid,(pos,load,kind) in own[2].items():
                if kind!='COLLECTOR' or load<40 or uid in seen:continue
                dist=min(int(_D[pos,b]) for b in bases)
                if not 0<dist<=left<=dist+3:continue
                seen.add(uid)
                cmd=(steps[t+1][seat].get('action') or {}).get(uid)
                cell=int(_TO[pos,_DIRECTION.index(cmd)])
                risk=min((int(_D[cell,w[0]]) for w in enemy[2].values() if w[2]=='COLLECTOR' and w[1]<=load),default=15)
                arrival=None;death=None
                for j in range(t+1,len(steps)):
                    after=steps[j][0]['observation']['players'][seat][2].get(uid)
                    if after is None:death=j;break
                    if after[0] in bases and after[1]==0:arrival=j;break
                rows.append(dict(seat=seat,uid=uid,t=t,load=load,distance=dist,slack=left-dist,cmd=cmd,
                    downhill=min(int(_D[cell,b]) for b in bases)<dist,enemy_distance=risk,arrival=arrival,death=death))
    return dict(id=gid,names=proof['seat_model_names'],seed=proof['random_seed'],scores=proof['rewards_by_seat'],steps=len(steps),
                statuses=proof['statuses'],terrain=F.sha(bytes(steps[0][0]['observation']['full_carbon'])),rows=rows)

def safe(item):
    try:return examine(item)
    except Exception as e:return dict(id=item[0],error=str(e))
if __name__=='__main__':
    cards=catalog()
    rows=[safe(v) for v in sorted(cards.items())]
    data=dict(revision=REV,scorecards=len(cards),games=rows)
    path=HERE/'elite_deadlines.json.gz'
    path.write_bytes(gzip.compress(F.canonical(data),mtime=0))
    print(len(cards),sum('error' in g for g in rows),sum(len(g.get('rows',[])) for g in rows))
