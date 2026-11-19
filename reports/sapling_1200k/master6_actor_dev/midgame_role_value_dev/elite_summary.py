"""Pool collector and planter turns within each physical game before averaging across games."""
from data import sr,HERE
from collections import defaultdict
import numpy as np
def main():
    g=sr.load(HERE/'elite_role_turns.json.gz');groups=defaultdict(list)
    names={r['name'] for r in sr.load(HERE.parents[1]/'all_top5/manifest.json')['top5']}
    for v in g:
        for s,name in enumerate(v['names']):
            if name not in names:continue
            margin=v['score'][s]-v['score'][1-s];result='win' if margin>0 else 'loss' if margin<0 else 'draw'
            for i,p in enumerate(v['phases'][s]):
                enemy=v['phases'][1-s][i];n=p['turns']
                row=[p.get('COLLECTOR',0)/n,p.get('PLANTER',0)/n,enemy.get('COLLECTOR',0)/n,enemy.get('PLANTER',0)/n]
                for label in (result,'all'):groups[(name,label,i)].append(row)
    out=[]
    for k,v in sorted(groups.items()):out.append(dict(name=k[0],result=k[1],phase=k[2],n=len(v),mean=np.mean(v,axis=0).tolist()))
    sr.retain(HERE/'elite_summary.json',out)
    for r in out:
        if r['phase']==3 and r['result']=='all':print(r['name'],r['n'],r['mean'])
if __name__=='__main__':main()
