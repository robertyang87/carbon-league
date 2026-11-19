"""Fit precisely one affine economic residual adjustment with all other inference fixed."""
from features import ROOT,GROUP,HERE,J,bridge,read,sha
import itertools
from collections import Counter
import numpy as np
import torch
def stats(table):
    out={}
    for k,v in table.items():
        err=[abs(r[k+'_margin_diff']-r['actual']) for r in v];good=[(r[k+'_choice']==int(r['actual']>0)) for r in v]
        delta=[r['actual']*r[k+'_choice'] for r in v]
        out[k]=dict(roots=len(v),correct=int(sum(good)),accuracy=float(np.mean(good)),paired_margin_mae=float(np.mean(err)),selected_gain=float(np.mean(delta)))
    return out
def run():
    torch.set_num_threads(1);torch.manual_seed(982541)
    collection=read(HERE/'features.json.gz');rows=collection['rows'];train=[j for j,r in enumerate(rows) if r['use']=='fit']
    assert len(train)==80 and not set(rows[i]['terrain'] for i in train)&{r['terrain'] for r in rows if r['use']=='held'}
    w=dict(np.load(J/'shared.npz',allow_pickle=False));parent=torch.from_numpy(w['value.weight'].copy()).double();bias=torch.from_numpy(w['value.bias'].copy()).double()
    h=torch.tensor([r['context'] for r in rows],dtype=torch.float64);cash=torch.tensor([r['cash'] for r in rows],dtype=torch.float64)
    fixed=torch.tensor([[np.nan if v is None else v for v in r['fixed']] for r in rows],dtype=torch.float64)
    actual=torch.tensor([r['margins'][1]-r['margins'][0] for r in rows],dtype=torch.float64)
    delta=torch.zeros(17,dtype=torch.float64,requires_grad=True)
    def predict(adj):
        z=h@parent.T+bias
        residual=z[...,1]+h@adj[:16]+adj[16]
        mar=cash[...,None]*torch.tensor([1.,-1.])+4000*residual
        uv=z[...,0].sigmoid()
        sc=.5*(2*uv-1)+.5*(mar/2000).tanh()
        scalar=(sc[...,0]-sc[...,1])*.5;marg=(mar[...,0]-mar[...,1])*.5
        return torch.where(fixed.isnan(),scalar,fixed),torch.where(fixed.isnan(),marg,cash)
    s0,m0=predict(delta.detach())
    maxerr=max(abs(s0[i,j].item()-r['baseline']['arms'][str(j)]) for i,r in enumerate(rows) for j in (0,1))
    assert maxerr<1e-5,maxerr
    ti=torch.tensor(train);a=actual[ti];count=Counter(rows[i]['terrain'] for i in train)
    importance=torch.tensor([1/count[rows[i]['terrain']] for i in train],dtype=torch.float64);importance/=importance.sum()
    opt=torch.optim.Adam([delta],lr=.015)
    for _ in range(80):
        _,m=predict(delta);d=m[ti,1]-m[ti,0]
        hub=torch.nn.functional.smooth_l1_loss(d/1000,a/1000,reduction='none')
        rank=torch.nn.functional.softplus(-a.sign()*(d/300).clamp(-15,15))
        loss=((hub+.2*rank)*importance).sum()+.15*delta.square().mean()
        opt.zero_grad();loss.backward();opt.step()
    # Reprice after returning the learned tensor to actual shipped float32 precision.
    inc=delta.detach().numpy();w['value.weight'][1]+=inc[:16].astype('f');w['value.bias'][1]+=np.float32(inc[-1])
    orig=dict(np.load(J/'shared.npz',allow_pickle=False));assert all(np.array_equal(w[k],v) for k,v in orig.items() if k not in ('value.weight','value.bias'))
    assert np.array_equal(w['value.weight'][0],orig['value.weight'][0]) and w['value.bias'][0]==orig['value.bias'][0]
    np.savez(HERE/'shared.npz',**w)
    eff=np.r_[w['value.weight'][1]-orig['value.weight'][1],w['value.bias'][1]-orig['value.bias'][1]]
    s1,m1=predict(torch.from_numpy(eff).double());obs=[]
    for i,r in enumerate(rows):
        v={k:r[k] for k in ('case','terrain','family','role','seed','use','number_changed')};v['actual']=float(actual[i])
        for k,s,m in [('parent',s0,m0),('pair',s1,m1)]:
            v[k+'_choice']=int(s[i,1]>s[i,0]);v[k+'_margin_diff']=float(m[i,1]-m[i,0]);v[k+'_score_diff']=float(s[i,1]-s[i,0])
        assert v['parent_choice']==r['baseline']['best'];obs.append(v)
    def info(g):
        per={t:stats({k:[r for r in g if r['terrain']==t] for k in ('parent','pair')}) for t in sorted({r['terrain'] for r in g})}
        av={k:{x:float(np.mean([v[k][x] for v in per.values()])) for x in ('accuracy','paired_margin_mae','selected_gain')} for k in ('parent','pair')}
        return dict(per_terrain=per,maps=len(per),map_equal=av,pooled=stats({k:g for k in ('parent','pair')}))
    groups={k:info([r for r in obs if r['use']==k]) for k in sorted({r['use'] for r in obs})}
    groups['held_original16']=info([r for r in obs if r['use']=='held' and r['family']=='tree_terminal_training'])
    groups['held_collectors']=info([r for r in obs if r['use']=='held' and r['role']=='collector'])
    g=groups['held'];q=g['map_equal'];gate=(q['pair']['accuracy']>max(.5,q['parent']['accuracy']) and q['pair']['paired_margin_mae']<q['parent']['paired_margin_mae'] and g['pooled']['pair']['accuracy']>max(.5,g['pooled']['parent']['accuracy']))
    terrain=[v['pair']['paired_margin_mae']-v['parent']['paired_margin_mae'] for v in g['per_terrain'].values()]
    accuracy=[v['pair']['accuracy']-v['parent']['accuracy'] for v in g['per_terrain'].values()]
    bootstrap=lambda a:np.quantile([np.mean(x) for x in itertools.product(a,repeat=len(a))],[.025,.975]).tolist()
    bridge.save(HERE/'results.json',dict(features=sha(HERE/'features.json.gz'),fit=sha(__file__),new_weights=sha(HERE/'shared.npz'),changed_coefficients=int(np.count_nonzero(eff)),fixed_policy_and_u=True,inc=eff.tolist(),
        sklearn_or_other_new_candidates=0,baseline_deploy_numerical_error=maxerr,gate_pass=bool(gate),groups=groups,map_bootstrap=dict(mae_diff=bootstrap(terrain),accuracy_diff=bootstrap(accuracy)),records=obs))
if __name__=='__main__':run()
