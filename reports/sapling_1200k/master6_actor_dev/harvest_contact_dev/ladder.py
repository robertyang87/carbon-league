"""Score the frozen escape model on the newly archived top-five games, with the harvester's actual orders."""
import numpy as np
from evade import situations,_DIRECTION
from scan import sr,HERE
import sys
TOP=HERE.parents[1]/'all_top5'
sys.path.insert(0,str(TOP))
import fetch
def main():
    w=sr.load(HERE/'weights.json')['w'];rows=[]
    for card in sr.load(TOP/'appendix_afc3951b/manifest.json')['games']:
        replay=sr.load(fetch.CACHE/card['path']);steps=replay['steps']
        for targetseat,name in enumerate(card['seat_names']):
            if name not in fetch.NAMES:continue
            seat=1-targetseat
            for t in range(50,min(201,len(steps)-1)):
                obs=dict(players=steps[t][0]['observation']['players'],player=seat)
                prev=dict(players=steps[t-1][0]['observation']['players'])
                cmd=steps[t+1][targetseat].get('action') or {}
                for uid,load,cells,x in situations(obs,prev):
                    z=x@w;actual=_DIRECTION.index(cmd.get(uid))
                    rows.append(dict(id=card['id'],name=name,seat=targetseat,t=t,uid=uid,cargo=load,
                                     actual=actual,predicted=int(z.argmax()),win=card['score_by_seat'][targetseat]>card['score_by_seat'][seat]))
    sr.retain(HERE/'elite_flight.json.gz',rows)
    for name in sorted({r['name'] for r in rows}):
        v=[r for r in rows if r['name']==name]; print(name,len(v),sum(not r['actual'] for r in v),sum(r['predicted']==r['actual'] for r in v))
if __name__=='__main__':main()
