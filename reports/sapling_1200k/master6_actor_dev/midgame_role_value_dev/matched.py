"""Compare common opponents on the same physical seat against Master5s and Master6."""
from collections import defaultdict
import numpy as np
from data import sr,HERE
def run():
    games=sr.load(HERE/'elite_role_turns.json.gz')
    masters=('claude-AlphaCarbon-Master5s','claude-AlphaCarbon-Master6')
    result=[]
    for phase in (0,1,2,3,4,5):
        bins=defaultdict(lambda:[[],[]])
        for g in games:
            for seat,name in enumerate(g['names']):
                enemy=g['names'][1-seat]
                if name not in masters or enemy in masters:continue
                if phase>=len(g['phases'][seat]):continue
                a,b=g['phases'][seat][phase],g['phases'][1-seat][phase]
                if a['turns']<(49 if phase==5 else 50):continue
                bins[(enemy,1-seat)][masters.index(name)].append([a.get('COLLECTOR',0)/a['turns'],b.get('COLLECTOR',0)/b['turns']])
        names=defaultdict(list);detail=[]
        for (name,s),pair in sorted(bins.items()):
            if min(map(len,pair))<2:continue
            a,b=(np.mean(v,axis=0) for v in pair);delta=b-a
            names[name].append(delta);detail.append(dict(opponent=name,opponent_seat=s,n=[len(x) for x in pair],difference=delta.tolist()))
        arr=np.array([np.mean(v,axis=0) for v in names.values()])
        result.append(dict(phase=phase,opponents=len(arr),mean_difference=arr.mean(0).tolist(),strata=detail))
    sr.retain(HERE/'matched.json',result)
    for r in result:print(r['phase'],r['opponents'],r['mean_difference'])
if __name__=='__main__':run()
