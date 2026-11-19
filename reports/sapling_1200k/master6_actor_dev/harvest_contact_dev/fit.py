from evade import situations,intercept,_DIRECTION
from scan import HERE,sr
import numpy as np
from scipy.optimize import minimize
def fit():
    data=[];cases=sr.load(HERE/'coverage.json.gz')
    for c in cases:
        raw=sr.load(sr.ROOT/c['source']);s=c['seat'];sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed']);prev=None
        for t,pair in enumerate(raw['actions'][:201]):
            obs=sim.obs(s)
            if t>=50:
                for uid,load,cells,x in situations(obs,prev):
                    data.append((c['seed'],x,_DIRECTION.index(pair[1-s].get(uid))))
            sim.step(*pair);prev=obs
    def arrays(part):
        v=[(x,y) for seed,x,y in data if (seed<=621758324)==part];return np.stack([a for a,b in v]),np.array([b for a,b in v])
    x,y=arrays(True)
    def fn(w):
        z=x@w;z-=z.max(1)[:,None];p=np.exp(z);p/=p.sum(1)[:,None]
        loss=-np.log(p[np.arange(len(y)),y]).sum()+.5*w@w
        p[np.arange(len(y)),y]-=1
        return loss,np.einsum('na,naf->f',p,x)+w
    res=minimize(fn,np.zeros(x.shape[-1]),jac=True,method='BFGS');w=res.x
    assert np.linalg.norm(fn(w)[1])<1e-3
    metrics=[]
    for split in (True,False):
        xx,yy=arrays(split);metrics.append(dict(train=split,n=len(yy),top1_hits=int(((xx@w).argmax(1)==yy).sum())))
    sr.retain(HERE/'weights.json',dict(w=w.tolist(),metrics=metrics,train_maps=sorted({a for a,b,c in data if a<=621758324})))
    print(metrics)
if __name__=='__main__':fit()
