"""Train only a shared linear correction to the live Sapling opening base scores."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import fit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    if os.sched_getaffinity(0)!={2} or os.getpriority(os.PRIO_PROCESS,0)<19:
        raise ValueError('use low-priority CPU2')
    torch.set_num_threads(1)
    plan=json.loads((HERE/'protocol.json').read_bytes())
    items=json.loads(gzip.decompress((HERE/'panel.json.gz').read_bytes()))['games']
    train=[g for g in items if g['split']=='train']
    assert len(train)==len({g['terrain_sha256'] for g in train})==218
    active=HERE.parent/'candidate.npz'
    net=fit.common.quantized(fit.common.model(active,plan['student_npz'])[0]).eval().requires_grad_(False)
    expert=fit.expert_model()
    data_path=HERE/'roots.npz'
    with np.load(data_path,allow_pickle=False) as z:
        indices=np.array([g['row_index'] for g in train])
        data={k:z[k][indices] for k in fit.common.trainer.FIELDS}
    assert np.array_equal(data['gid'],[g['gid'] for g in train]) and not data['step'].any()
    assert not data['carbon'].any() and not data['value'].any()
    assert np.all(data['y'][:,:10]<0) and np.all(data['y'][:,10:]>=0)
    hidden=[]; original=[]
    with torch.no_grad():
        for i in range(0,len(train),16):
            b=fit.common.batch(data,slice(i,i+16));slots=[]
            handle=net.bhead.register_forward_pre_hook(lambda module,inputs:slots.append(inputs[0].clone()))
            try:
                s=net(b)[0]
            finally:
                handle.remove()
            assert len(slots)==2
            original.append(fit.blended(expert(b)[0],s)[:,10:,:3])
            hidden.append(torch.stack(slots,1))
    x,logits=torch.cat(hidden),torch.cat(original)
    y=torch.from_numpy(data['y'][:,10:].copy()).long()
    legal=torch.from_numpy(data['legal'][:,10:,:3].copy())
    assert legal.gather(-1,y[...,None]).all() and (legal.sum(-1)>=2).all()
    counts=torch.bincount(y.flatten(),minlength=3)
    role_weights=counts.clamp(min=1).float().rsqrt()[y]
    role_weights/=role_weights.sum(1,keepdim=True)
    dw=torch.zeros(3,x.shape[-1],requires_grad=True);db=torch.zeros(3,requires_grad=True)
    cfg=plan['optimizer']
    opt=torch.optim.Adam([dw,db],lr=cfg['lr'])
    log=[];consecutive=0
    for i in range(cfg['max_steps']):
        z=(logits+.25*F.linear(x,dw,db)).masked_fill(~legal,-1e9)
        nll=(F.cross_entropy(z.transpose(1,2),y,reduction='none')*role_weights).sum(1).mean()
        loss=nll+cfg['weight_l2_mean']*dw.square().mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        exact=(z.argmax(-1)==y);consecutive=consecutive+1 if exact.all() and nll.item()<.04 else 0
        log.append({'step':i,'nll':float(nll.item()),'ordered_pair_correct':int(exact.all(1).sum())})
        if consecutive>=5:
            break
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([dw,db],cfg['gradient_clip'],error_if_nonfinite=True);opt.step()
    np.savez(HERE/'opening_weights.npz',weight=dw.detach().numpy(),bias=db.detach().numpy())
    train_results=F.linear(x,dw,db).detach()
    pred=(logits+.25*train_results).masked_fill(~legal,-1e9).argmax(-1)
    fit.common.write_json(HERE/'fit.json',{'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(HERE/'protocol.json'),
        'panel_sha256':digest(HERE/'panel.json.gz'),'inputs_sha256':digest(data_path),'weight_sha256':digest(HERE/'opening_weights.npz'),
        'raw_base_role_counts':counts.tolist(),'train_maps':len(train),'trained_scalars':dw.numel()+db.numel(),
        'updates_applied':len(log)-(consecutive>=5),'train_only_early_stop_reached':consecutive>=5,
        'final_train_exact_pairs_on_recorded_prior_base':int((pred==y).all(1).sum()),'trace':log})
    print('training-only exact pairs',int((pred==y).all(1).sum()),len(train),flush=True)


if __name__=='__main__':
    run()
