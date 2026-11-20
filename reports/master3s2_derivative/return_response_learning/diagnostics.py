"""Describe already held-out worker returns without any further parameter fitting."""
import numpy as np
from fit import HERE,arrays,samples,read,save
from model import rr_predict


def main():
    plan=read(HERE/'protocol.json');weights=read(HERE/'weights.json');data=samples([r for r in plan['ordinary'] if r['split']=='val'])
    X,y,w,ids=arrays(data);load=np.array([r[3] for r in data]);subgroups={'all':load>0,'cargo_at_least_200':load>=200,'distance_at_most_2_steps':X[:,1]*8<=2,'next_move_toward_home':X[:,10]>0,'stationary':X[:,8]==1,'last_12_steps':X[:,6]*299>=287};out={}
    for name,pick in subgroups.items():
        y2=y[pick];ids2=ids[pick];groups=np.unique(ids2);pred={key:rr_predict(X[pick],weights[key]) for key in ('enhanced','state_only')};res={}
        for k,v in pred.items():
            res[k]={'map_equal_mean_predicted':float(np.mean([np.mean(v[ids2==i]) for i in groups])),
                    'map_equal_mse':float(np.mean([np.mean((v[ids2==i]-y2[ids2==i])**2) for i in groups]))}
        out[name]=dict(n=int(np.sum(pick)),maps=len(groups),full_repaid=int(sum(y2==1)),zero=int(sum(y2==0)),partial=int(sum((y2>0)&(y2<1))),observed=float(np.mean([np.mean(y2[ids2==i]) for i in groups])),models=res)
    save('worker_diagnostics.json',out)

if __name__=='__main__':main()
