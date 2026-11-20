"""Show what was observable at the tree decision and which subsequent commands occurred."""
import gzip
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())

def metric(a,b):
    dx=abs(a%15-b%15);dy=abs(a//15-b//15)
    return min(dx,15-dx)+min(dy,15-dy)

def cases():
    replay=read(HERE/read(HERE/'match.json')['replay_file'])
    f=replay['steps'];p=read(HERE/'plant_holds.json.gz')
    cards=[]
    for t,seat,cell in ((82,1,79),(85,1,80),(158,1,167)):
        e=next(x for x in p if x['seat']==seat and x['t']==t and x['cell']==cell)
        ids=[e['planter']]
        if t<100:ids+=['player-0-worker-22','player-0-worker-24']
        root=f[t][0]['observation']['players'];other=root[1-seat]
        ranked=sorted((metric(w[0],cell),u,w[0],w[2],w[1]) for u,w in other[2].items())
        near=next(v for v in ranked if v[3]=='PLANTER')
        ids.append(near[1])
        tracked={}
        for u in sorted(set(ids)):
            s=int(u.split('-')[1]);tape=[]
            for a in sorted(set([max(0,t-x) for x in (10,5,3)]+list(range(t,min(t+13,299))))):
                w=f[a][0]['observation']['players'][s][2].get(u)
                tape.append({'step':a,'state':w,'issued':(f[a+1][s]['action'] or {}).get(u,'NONE') if w else None})
            tracked[u]=tape
        cards.append({'tree':{k:v for k,v in e.items() if k!='cash'},'nearest_at_root':ranked[0],
                      'nearest_opponent_planter_at_root':near,'afterward_worker_traces':tracked})
    h=read(HERE/'ledger.json.gz')
    jumps=[{'arrival':a+1,**e} for a in range(174,177) for e in h['receipts_by_transition'][a]]
    for e in jumps:
        e['before']=[]
        for b in (10,5,3,1,0):
            t=e['arrival']-b;uid=e['worker'];s=e['seat'];w=f[t][0]['observation']['players'][s][2].get(uid)
            e['before'].append({'step':t,'state':w,'issued':(f[t+1][s]['action'] or {}).get(uid,'NONE')})
    (HERE/'cases.json').write_text(json.dumps(dict(trees=cards,consecutive_deposits=jumps),ensure_ascii=False,indent=2)+'\n')
    return cards,jumps

if __name__=='__main__':
    for c in cases()[0]:
        print(c['tree']['t'],c['nearest_at_root'],c['nearest_opponent_planter_at_root'])
