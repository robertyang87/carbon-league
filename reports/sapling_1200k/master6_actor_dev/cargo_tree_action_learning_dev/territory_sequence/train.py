"""Learn both factors of the deployed worker–action interaction; each horizon sees the same decisions."""
import collections
import math
import sys

import numpy as np
import torch
from torch import nn

from data import HERE,EXP,RULE,read,write,sha
from gather import policy

sys.path.insert(0,str(EXP.parent/'blended_actor_dev/serving'))
import build as pack

KEYS=('U.weight','U.bias','V.weight','V.bias')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def standard_weights(rows,y):
    labels=collections.defaultdict(list)
    groups=[(r['source'],r['role'],r['t']//48) for r in rows]
    for i,r in enumerate(rows):
        if r['split']=='train':
            labels[groups[i]].append(i)
    result=np.zeros_like(y)
    for group in set(groups):
        train=labels[group]
        if not train:raise ValueError(('unsupported normalization',group))
        v=y[train]
        location=np.median(v,axis=0)
        scale=np.maximum(1.4826*np.median(np.abs(v-location),axis=0),1.)
        obs=[i for i,g in enumerate(groups) if g==group]
        result[obs]=(1+.8*np.tanh((y[obs]-location)/scale))*np.asarray([rows[i]['w'] for i in obs])[:,None]
    return result.astype(np.float32)


def logits(z,mask):
    return z.masked_fill(~mask,-1e6).log_softmax(-1)

def run():
    orig=EXP/'candidate.npz';base=EXP/'candidate.py'
    parent=policy.load(base); W=[torch.tensor(parent._NETS[1].w[k].copy()) for k in KEYS]
    rows=read(HERE/'sequence.json.gz');pr=read(HERE/'pairs.json.gz')
    with np.load(HERE/'sequence.npz') as f:
        h,m,mask,z=[torch.from_numpy(f[k]) for k in ('h','move','mask','z')]
        w=torch.tensor(standard_weights(rows,f['y']))
    with np.load(HERE/'pairs.npz') as f:
        ph,pm,pmask,pz=[torch.from_numpy(f[k]) for k in ('h','move','mask','z')]
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    observed=torch.tensor([r['label'] for r in rows]); pi=logits(z,mask)
    ptr=[i for i,r in enumerate(pr) if r['split']=='train' and all(a['prefix'] for a in r['arms'].values())]
    assert not {pr[i]['map'] for i in ptr}&{r['map'] for r in rows if r['split']!='train'}
    assert not {pr[i]['map'] for i in ptr}&{r['map'] for r in pr if r['split']=='hold'}
    actions=torch.tensor([[r['arms'][arm]['action'] for arm in ('move','stay')] for r in pr])
    ypair=np.array([np.subtract(r['arms']['stay']['ys'],r['arms']['move']['ys']) for r in pr])
    original={k:v for k,v in np.load(orig,allow_pickle=False).items()}
    rows_train=torch.tensor(train)
    base_src=base.read_text()
    results={}
    for hi,arm in enumerate(('control','candidate')):
        dw=nn.ParameterList([nn.Parameter(torch.zeros_like(a)) for a in W])
        opt=torch.optim.Adam(dw.parameters(),lr=.002)
        def score(hh,mm,zz):
            def interaction(p):
                u=torch.nn.functional.linear(hh,*p[:2]);v=torch.nn.functional.linear(mm,*p[2:])
                return (u[:,None]*v).sum(2)
            return zz+.25*(interaction([p+dp for p,dp in zip(W,dw)])-interaction(W))
        history=[]
        pscale=max(1.,np.median(np.abs(ypair[ptr,hi]))) if ptr else 1.
        target=torch.tensor(.5+.45*np.tanh(ypair[:,hi]/pscale),dtype=torch.float32)
        for epoch in range(12):
            batches=np.array_split(np.random.default_rng(RULE['seed']+epoch).permutation(train),math.ceil(len(train)/128)); hist=[]
            for ids in batches:
                pred=logits(score(h[ids],m[ids],z[ids]),mask[ids]);nll=-pred[torch.arange(len(ids)),observed[ids]]
                loss=(w[ids,hi]*nll).sum()/w[ids,hi].sum()
                kl=(pi[ids].exp()*(pi[ids]-pred)).sum(1).mean()
                total=loss+1.2*kl+.02*sum(v.square().mean() for v in dw)
                if ptr:
                    sq=score(ph[ptr],pm[ptr],pz[ptr]).gather(1,actions[ptr])
                    total=total+.12*torch.nn.functional.binary_cross_entropy_with_logits(sq[:,1]-sq[:,0],target[ptr])
                opt.zero_grad(set_to_none=True);total.backward()
                torch.nn.utils.clip_grad_norm_(dw.parameters(),2.,error_if_nonfinite=True);opt.step()
                hist.append(float(total.detach()))
            history.append(float(np.mean(hist)))
        raw=original.copy()
        for k,d in zip(KEYS,dw):
            raw[k]=raw[k]+d.detach().numpy()
        path=HERE/(arm+'.npz');np.savez(path,**raw)
        with np.load(orig,allow_pickle=False) as f:
            embedded=pack.quantize(path,f)
        bundle=HERE/(arm+'.py');bundle.write_text(pack.substitute(base_src,embedded))
        assert pack.substitute(bundle.read_text(),pack.literal(base_src,pack.STUDENT))==base_src
        n=policy.load(bundle)
        diff={k:int(np.count_nonzero(n._NETS[1].w[k]!=parent._NETS[1].w[k])) for k in KEYS}
        results[arm]=dict(package=sha(bundle),weights=sha(path),bytes=bundle.stat().st_size,quantized_parameters_changed=diff,loss=history)
    write(HERE/'fit.json',dict(protocol=sha(HERE/'protocol.json'),parent=sha(base),parent_weights=sha(orig),outputs=results,
                                training_rows=len(train),training_terrain=len({rows[i]['map'] for i in train}),matched_train_roots=len(ptr),
                                matched_training_terrains=len({pr[i]['map'] for i in ptr}),anchors=['75% frozen it50 scores','KL from parent mixture'],
                               data_sha256={name:sha(HERE/name) for name in ('sources.json','pairs.npz','pairs.json.gz','sequence.npz','sequence.json.gz')}))

if __name__=='__main__':
    run()
