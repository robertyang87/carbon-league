"""Describe overlap and direct productivity; never compare unobserved hires as if they were randomized."""
from collections import defaultdict
import numpy as np
from data import sr,HERE
def main():
    rows=sr.load(HERE/'hires.json.gz');out=[]
    for lo,hi in ((50,100),(100,150),(150,201)):
        for count in (0,1,2,3):
            b=[r for r in rows if lo<=r['base']['t']<hi and min(3,r['base']['workersP'])==count]
            groups={}
            for role in ('C','P'):
                v=[r for r in b if r['role']==role]
                if v:
                    groups[role]=dict(n=len(v),maps=len({x['seed'] for x in v}),direct50_mean=float(np.mean([x['net50'] for x in v])),
                                      direct50_median=float(np.median([x['net50'] for x in v])),lifetime_mean=float(np.mean([x['lifetime'] for x in v])))
            out.append(dict(start=lo,end=hi,planters=min(3,count),hired=groups))
    sr.retain(HERE/'role_overlap.json',out)
    for r in out:
        if r['planters']==2:print(r)
if __name__=='__main__':main()
