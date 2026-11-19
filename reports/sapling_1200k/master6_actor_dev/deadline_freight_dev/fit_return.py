"""Fit a descriptive endgame arrival model; evaluate on unseen terrain and all Master6 matches."""
import hashlib,gzip,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
MASTER='claude-AlphaCarbon-Master6'
FIELDS=['intercept','distance','slack','cargo','downhill','downhill_with_zero_slack','reachable_by_light_enemy','enemy_gap','moving']
def features(row):
    return [1,row['distance']/8,row['slack']/3,min(row['load'],500)/250,int(row['downhill']),
            int(row['downhill'] and row['slack']==0),int(row['enemy_distance']<=1),min(row['enemy_distance'],6)/6,int(bool(row['cmd']))]
def prob(x,w):return 1/(1+np.exp(-np.clip(x@w,-30,30)))
def fit(x,y,importance):
    w=np.zeros(x.shape[1]);pen=np.eye(len(w))*2;pen[0,0]=0
    for _ in range(40):
        p=prob(x,w)
        h=x.T@((importance*p*(1-p))[:,None]*x)+pen
        delta=np.linalg.solve(h,x.T@(importance*(p-y))+pen@w)
        w-=delta
        if max(abs(delta))<1e-8:break
    return w
def main():
    games=json.loads(gzip.decompress((HERE/'elite_deadlines.json.gz').read_bytes()))['games']
    clean=[g for g in games if g.get('statuses')==['DONE','DONE']]
    reserved={g['terrain'] for g in clean if MASTER in g['names']}
    parts={k:[] for k in ('train','held_maps','master6')}
    for g in clean:
        key='master6' if g['terrain'] in reserved else ('held_maps' if int(g['terrain'][:8],16)%5==0 else 'train')
        for r in g['rows']:
            if key=='master6' and g['names'][r['seat']]!=MASTER:continue
            parts[key].append((g['terrain'],features(r),float(r['arrival'] is not None)))
    assert all(parts.values())
    def arrays(rows):
        maps,x,y=zip(*rows);maps=np.array(maps);x=np.array(x);y=np.array(y)
        _,ind,cnt=np.unique(maps,return_inverse=True,return_counts=True)
        weights=1/cnt[ind]
        return x,y,weights
    x,y,wt=arrays(parts['train']);w=fit(x,y,wt);base=fit(x[:,:4],y,wt)
    out={}
    for name,row in parts.items():
        x,y,wt=arrays(row);pred=prob(x,w);ctrl=prob(x[:,:4],base)
        out[name]=dict(maps=len({r[0] for r in row}),workers=len(row),arrivals=int(y.sum()),
                       brier=float(np.average((pred-y)**2,weights=wt)),base_brier=float(np.average((ctrl-y)**2,weights=wt)),
                       calibration_error=float(np.average(pred-y,weights=wt)))
    obj=dict(fields=FIELDS,weights=w.tolist(),baseline=base.tolist(),stats=out,label='same worker observed alive at own base with zero cargo by terminal; no causal claim')
    (HERE/'return_weights.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
