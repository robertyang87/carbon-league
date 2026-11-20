"""Postprocess frozen K-menus using only the public units and base positions at each root."""
import collections
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path
from probe import HERE, GAME, scanpath, s, put, sha

@lru_cache(None)
def paths(a,b):
    y,x=divmod(a,15);z,w=divmod(b,15);dx=min(abs(w-x),15-abs(w-x))
    return math.comb(s.dist(a,b),dx)


def lanes(players):
    """All cells 1–3 moves toward a nearest enemy base. Mark unavoidable across every shortest route."""
    bases=list(players[1][1].values());result=collections.defaultdict(list)
    for uid,(p,load,role) in players[1][2].items():
        if role!='COLLECTOR' or load<=0:continue
        nearest=min(s.dist(p,b) for b in bases)
        targets=[b for b in bases if s.dist(p,b)==nearest]
        denom=sum(paths(p,b) for b in targets)
        for q in range(225):
            steps=s.dist(p,q)
            if q in bases or not 1<=steps<=3:continue
            n=sum(paths(p,q)*paths(q,b) for b in targets if steps+s.dist(q,b)==nearest)
            if n:result[q].append(dict(uid=uid,distance=steps,load=load,required=n==denom,shortest_path_counts=[n,denom]))
    return result


def public_options(ps,menu):
    corridors=lanes(ps)
    out=[]
    for order in menu:
        workers=ps[0][2];after={u:s.move(p,order.get(u,'NONE')) for u,(p,_,_) in workers.items()}
        occ=collections.Counter(after.values());occ.update(p for u,p in ps[0][1].items() if order.get(u,'NONE')!='NONE')
        options=[]
        for uid,q in after.items():
            p,load,role=workers[uid]
            if role!='COLLECTOR' or occ[q]>1:continue
            for v,targets in corridors.items():
                left=s.dist(q,v)
                if left>=s.dist(p,v):continue
                eligible=[e for e in targets if e['load']>load and 1+left<=e['distance']]
                if eligible:options.append(dict(worker=uid,start=p,cargo=load,root_destination=q,block=v,earliest=1+left,targets=eligible))
        out.append(options)
    return out


def run():
    evidence=HERE/'evidence.json.gz';r=json.load(gzip.open(evidence));g=s.read();rows=[]
    for t,raw in sorted(r['rows'].items(),key=lambda i:int(i[0])):
        t=int(t);m=raw['measurement'];ps=g['steps'][t][0]['observation']['players']
        options=public_options(ps,m['menu'])
        nums=lambda opts:{'any':sum(bool(q) for q in opts),'multi':sum(any(len(o['targets'])>=2 for o in q) for q in opts),
                         'multi_unavoidable':sum(any(sum(e['required'] for e in o['targets'])>=2 for o in q) for q in opts)}
        actual=raw['actual_own']
        focus=[(i,q) for i,lst in enumerate(options) for q in lst if t==267 and q['worker'].endswith('-53') and q['block']==169]
        summaries={'step':t,'K':m['candidates'],'H':m['depth'],'R':m['rollouts'],'actual_index':raw['team_equals_recorded'],'picked':m['best'],
                   'complete_distribution':collections.Counter(m['depth_by_arm']),'selected_has_multi':bool(nums([options[m['best']]])['multi']),
                   **nums(options),'focus_267_worker53_cell169':[i for i,_ in focus],'details':options}
        rows.append(summaries)
    put(HERE/'routes.json.gz',dict(evidence_sha256=sha(evidence.read_bytes()),rows=rows,rule='For a worker to approach a block cell, its chosen move must reduce its Manhattan distance to the cell, can arrive within the target shortest arrival time (1–3 steps) to a nearest enemy base, remains lighter, and the current own joint order is uncontested. Counts assume no intervening change of mind/cargo. Unavoidable means the cell is present on every shortest path including ties among nearest bases.'))
    for c in rows:print({k:v for k,v in c.items() if k!='details'})

if __name__=='__main__':run()
