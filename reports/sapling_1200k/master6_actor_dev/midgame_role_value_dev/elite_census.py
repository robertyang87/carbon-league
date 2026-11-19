"""Count active worker roles and output per physical game for every verified elite appearance."""
from pathlib import Path
from collections import Counter
import sys
from data import HERE,sr
TOP=HERE.parents[1]/'all_top5';sys.path.insert(0,str(TOP))
import fetch
def examine(card):
    raw=fetch.CACHE/card['path'];assert fetch.sha(raw.read_bytes())==card['sha256']
    steps=sr.load(raw)['steps'];stats=[[],[]];counters=[Counter(),Counter()]
    for t,(a,b) in enumerate(zip(steps,steps[1:])):
        ps=a[0]['observation']['players']
        for seat,p in enumerate(ps):
            c=counters[seat];c['turns']+=1
            orders=b[seat].get('action') or {}
            for uid,(pos,load,role) in p[2].items():
                c[role]+=1
                if role=='COLLECTOR' and not orders.get(uid):c['collector_stays']+=1
            for uid in p[1]:
                if orders.get(uid):c[orders[uid]]+=1
        if t+1 in (50,100,150,200,250,len(steps)-1):
            for side in (0,1):stats[side].append(dict(end=t+1,**counters[side]))
            counters=[Counter(),Counter()]
    return dict(id=card['id'],seed=card['seed'],names=card['seat_names'],score=card['score_by_seat'],phases=stats)
if __name__=='__main__':
    manifests=[TOP/'manifest.json',*sorted(TOP.glob('appendix_*/manifest.json'))];games=[]
    for path in manifests:games.extend(sr.load(path)['games'])
    assert len({g['id'] for g in games})==len(games)
    rows=[examine(c) for c in games];sr.retain(HERE/'elite_role_turns.json.gz',rows)
    print(len(rows))
