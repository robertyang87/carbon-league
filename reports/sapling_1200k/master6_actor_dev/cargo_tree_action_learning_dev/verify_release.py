"""Verify score attribution and review later cash labels separately from policy inputs."""
import collections,hashlib,json,math,sys
import numpy as np
from collect import HERE,PARENT,BASE,FULL,ACTOR,read,save,sha,policy,CFG
from evaluate import logp
def run():
    rows=read(HERE/'rows.json.gz');evidence=read(HERE/'normal_entry_actions.json.gz');panel=read(HERE/'panel.json')
    with np.load(HERE/'public_prefix_scores.npz') as z: hidden=z['hidden'];z0=z['logits'];legal=z['legal']
    files={k:policy.load(p) for k,p in [('parent',PARENT),('candidate',HERE/'candidate.py'),('bc',HERE/'bc.py')]}
    def weight(arm,k):return files[arm]._NETS[1].w['whead.'+k]
    receipts={}
    for n in ('candidate','bc'):
        err=[];ctl=files['parent'];m=files[n]
        for v in evidence:
            o=v['observed'];i=v['index']
            if o[n]['prefix_changed_vs_parent'] or o[n]['selected_return_branch']!=o['parent']['selected_return_branch'] or o[n]['factual_logp'] is None:continue
            new=z0[i]+.25*((weight(n,'weight')-weight('parent','weight'))@hidden[i]+weight(n,'bias')-weight('parent','bias'))
            expected=logp(new,legal[i],rows[i]['target']);err.append(abs(expected-o[n]['factual_logp']))
        if not err or max(err)>1.5e-5:raise ValueError(('deployment score identity',n,max(err,default=None)))
        with np.load(BASE) as a,np.load(HERE/(n+'.npz')) as b:
            changed={k:int(np.count_nonzero(a[k]!=b[k])) for k in a.files if k!='meta' and np.any(a[k]!=b[k])}
            if set(changed)!={'whead.weight','whead.bias'}:raise ValueError('unexpected weights')
        assert all(getattr(m,a)==getattr(ctl,a) for a in ('_B1','_Q1','_M1','_HYBRID'))
        receipts[n]=dict(weights=sha(HERE/(n+'.npz')),package=sha(HERE/(n+'.py')),changed=changed,fixed_prefixes=len(err),max_abs_logp_err=max(err))
    split={}
    for sp in ('train','val','test'):
        split[sp]={}
        for task in ('bank','tree'):
            rr=[x for x in rows if x['split']==sp and x['task']==task]
            split[sp][task]=dict(roots=len(rr),positive=sum(x['positive'] for x in rr),failed=sum(x['failed'] for x in rr),maps=len({v['terrain'] for v in rr}),both_seats=sorted({v['seat'] for v in rr}),master_examples=sum(x['is_master'] for x in rr))
    if any(c['seed'] in (621758343,621758344,621758345,621758346) for c in panel['cards']):raise ValueError('excluded map')
    save('release_check.json',dict(parent=sha(PARENT),split_counts=split,arms=receipts,
                                  unique_terrains={sp:len({r['terrain'] for r in rows if r['split']==sp}) for sp in split}))
if __name__=='__main__':run()
