"""Screen CPU-only, full precision execution on archived actual native search tensors."""
import base64
import io
import json
import os
from pathlib import Path
from statistics import median
import sys
import time

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "rollout_readout/timing"
sys.path.insert(0,str(PANEL))
from capture import setup
plan, model=setup()
import numpy as np
import torch

stock=model._NET
value=np.load(io.BytesIO(base64.b64decode(model._VALUE)))
direct=model._TorchNet(model._POLICY,{k:value[k].astype(np.float32) for k in value.files})
def convp(x,w,b):
    Y=x.permute(0,2,3,1).contiguous()
    return torch.nn.functional.conv2d(torch.nn.functional.pad(Y,(0,0,1,1,1,1),mode="circular").permute(0,3,1,2),w,b)
def wrapw(item):
    return item.contiguous(memory_format=torch.channels_last) if item.dim()==4 else item
direct.stem=tuple(map(wrapw,direct.stem))
direct.blocks=[tuple(map(wrapw,row)) for row in direct.blocks]
direct._conv=convp

@torch.inference_mode()
def index_net(X,sid,sc=None):
    w=stock
    n=len(X)
    def gather(a,idx):return torch.index_select(a,0,idx)
    def conv(a,U,b):
        d=(w.KB @ a.flatten(1).reshape(36,-1)).reshape(36,16*n,a.shape[2])
        m=torch.bmm(d,U)
        return (w.KA @ m.reshape(36,-1)).reshape(256,n,U.shape[2]).add_(b)
    x=torch.from_numpy(X[:,:w.cin]).permute(2,3,0,1).reshape(225,n,w.cin)
    h=conv(gather(x,w.winb),*w.stem).add_(w.emb[sid]).relu_()
    for U1,b1,g,gb,U2,b2 in w.blocks:
        t=conv(gather(h,w.win),U1,b1).relu_()
        t.add_((gather(t,w.board).mean(0) @ g).add_(gb))
        h.add_(conv(gather(t,w.win),U2,b2)).relu_()
    hf=gather(h,w.board)
    lw=(hf @ w.hw[0]).add_(w.hw[1]).permute(1,0,2).contiguous().numpy()
    lb=(hf @ w.hb[0]).add_(w.hb[1]).permute(1,0,2).contiguous().numpy()
    v=None
    if sc is not None:
        s=torch.from_numpy(sc)
        z=(hf @ w.v1[0]).add_(w.v1[1]).relu_()
        z=torch.cat([z.mean(0),z.amax(0),s],1)
        v=(s[:,0]+((z @ w.v2[0].T).add_(w.v2[1]).relu_() @ w.v3[0].T+w.v3[1])[:,0]).numpy()
    return lw,lb,v
def run():
    p=json.loads((PANEL/'panel.json').read_bytes());data=np.load(PANEL/'panel.npz')
    roots=p['public_roots'];cases=[roots[0],roots[-1]]
    trials={'stock':stock,'index':index_net,'direct_nhwc':direct}
    results={}
    for root in cases:
        jobs=[]
        for batch in root['batches']:
            key=batch['key']
            jobs.append((data[key].astype(np.float32)*np.float32(.01),batch['sid'],data[key+'s'] if batch['scalar'] else None))
        result={}
        for name,fn in trials.items():
            diff=np.zeros(3);changed=np.zeros(2,dtype=np.int64)
            for args in jobs:
                orig=stock(*args);test=fn(*args)
                for i,(a,b) in enumerate(zip(orig,test)):
                    if a is not None:diff[i]=max(diff[i],float(np.abs(a-b).max()))
                    if i<2:changed[i]+=np.count_nonzero(a.argmax(-1)!=b.argmax(-1))
            result[name]=dict(errors=diff.tolist(),changed=changed.tolist(),times=[])
        for rep in range(3):
            names=list(trials);names=names[rep:]+names[:rep]
            for name in names:
                beg=time.perf_counter()
                for args in jobs:trials[name](*args)
                result[name]['times'].append(time.perf_counter()-beg)
        results[str(root['seed'])+'_'+str(root['t'])]=result
    print(json.dumps(results,indent=2))
if __name__=='__main__':run()
