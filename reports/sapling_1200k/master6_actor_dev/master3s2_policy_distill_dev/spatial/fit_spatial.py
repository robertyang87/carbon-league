"""Matched raw-soft baseline plus training-only spatial feature distillation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from extract import HERE,RAW,B
import fit as R
base=R.base

def normalized(x):
    h=x-x.mean(dim=1,keepdim=True)
    return h*torch.rsqrt(h.square().mean(dim=1,keepdim=True)+1.e-5)

def retained(net):
    board=[]
    handle=net.aux.register_forward_pre_hook(lambda _m,args: board.append(args[0]))
    return board,handle

def target_matrix(d,cards):
    rec=json.loads((HERE/'labels.json').read_bytes())
    assert rec['extract_sha256']==R.sha(HERE/'extract.py') and rec['protocol_sha256']==R.sha(HERE/'protocol.json')
    feat=[];ids={};index=[]
    for c,r in zip(cards,rec['records']):
        path=Path(rec['directory'])/(c['id']+'.npz');assert R.sha(path)==r['sha256'] and r['id']==c['id']
        with np.load(path,allow_pickle=False) as z:
            for t,h in zip(z['step'],z['board']):
                ids[(c['gid'],int(t))]=len(feat);feat.append(h)
    for g,t in zip(d['gid'],d['step']):index.append(ids[(int(g),int(t))])
    return np.asarray(feat),np.asarray(index)

def feature_test(net,proj,labels,d,needle,take,cards):
    byseat={};net.eval();rec,hook=retained(net)
    try:
        with torch.inference_mode():
            out=[]
            for start in range(0,len(take),16):
                ix=take[start:start+16];rec.clear();net(base.batch(d,ix));assert len(rec)==1
                x=normalized(proj(rec.pop()));teacher=normalized(torch.from_numpy(labels[needle[ix]].astype(np.float32)))
                err=(x-teacher).square().mean((1,2,3))
                null=(x-teacher.roll(7,-1)).square().mean((1,2,3))
                out.extend(zip(err.tolist(),null.tolist()))
        out=np.asarray(out)
        for s in (0,1):
            mask=np.asarray([s==cards[j]['seat'] for j in d['card'][take]])
            byseat[str(s)]={'roots':int(mask.sum()),'mse':float(out[mask,0].mean()),'shift7':float(out[mask,1].mean())}
    finally:hook.remove()
    return byseat

def run():
    assert os.sched_getaffinity(0)=={2};torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=json.loads((HERE/'protocol.json').read_bytes());proto=p['config']
    for k,h in p['identity'].items():assert R.sha(RAW/k)==h
    d,cards=R.read_panel();labels,needle=target_matrix(d,cards);order=R.batch_order(d,cards,proto)
    assert hashlib.sha256(order.tobytes()).hexdigest()==p['order_sha256']
    bs=json.loads((RAW/'fit.json').read_bytes());assert np.array_equal(order,np.asarray(bs['order_rows']))
    torch.manual_seed(proto['seed']);net,meta=base.model(R.PARENT,R.PARENT_HASH)
    with torch.random.fork_rng():
        torch.manual_seed(861921)
        proj=nn.Conv2d(net.aux.in_channels,labels.shape[1],1,bias=True)
    initp=copy.deepcopy(proj);initnet=copy.deepcopy(net)
    params=[]
    for k,v in net.named_parameters():
        v.requires_grad_(not k.startswith(R.NOT_POLICY))
        if v.requires_grad:params.append(v)
    kwargs=dict(lr=proto['learning_rate'],weight_decay=proto['weight_decay'])
    opt=torch.optim.AdamW(params,**kwargs);propt=torch.optim.AdamW(proj.parameters(),**kwargs)
    rec,hook=retained(net);history=[];net.train()
    for i,ix in enumerate(order):
        rec.clear();b=base.batch(d,ix);logits=net(b)[0];assert len(rec)==1
        mask=(b['y']>=0)&(b['legal'].sum(-1)>1)
        q=torch.from_numpy(d['q'][ix]);soft=-(q*logits.log_softmax(-1)).sum(-1)[mask].mean()
        target=normalized(torch.from_numpy(labels[needle[ix]].astype(np.float32)))
        space=(normalized(proj(rec.pop()))-target).square().mean()
        loss=soft+.10*space;assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);propt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,proto['clip_norm'],error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(proj.parameters(),proto['clip_norm'],error_if_nonfinite=True)
        opt.step();propt.step()
        history.append([float(soft.detach()),float(space.detach())])
    hook.remove();identity=R.sha(HERE/'protocol.json')
    w=HERE/'student.npz';assert not w.exists()
    base.trainer.export(net,str(w),dict(meta,spatial_distill_identity=identity))
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);src=tmp/'student.npz';src.write_bytes(w.read_bytes());output=tmp/'student.py';base.pack.main(str(src),str(output),2)
        (HERE/'student.py').write_bytes(output.read_bytes())
        ix=np.flatnonzero([cards[j]['split']=='held' for j in d['card']]);held={k:v[ix] for k,v in d.items()}; eval_arms={}
        chosen=[]
        for s in (0,1):
            for c in cards:
                if c['split']=='held' and c['seat']==s:
                    ids=ix[(d['card'][ix]==c['gid'])&(d['prefix_mode'][ix]==0)]
                    chosen.extend([int(ids[len(ids)//2])])
        parity=R.parity(tmp/'student_q8.npz',d,chosen)
        assert parity['max_abs']<5e-4 and parity['argmax_differences']==0
    matched,_=base.model(RAW/'student.npz');candidate,_=base.model(w)
    for k,n in [('raw_soft',matched),('spatial',candidate),('parent',initnet)]:
        eval_arms[k]=base.predict(base.quantized(n),held)
    table=R.score(d,cards,ix,eval_arms)
    assert abs(table['raw_soft']['all']['map_equal_kl']-bs['held_metric']['student']['all']['map_equal_kl'])<1.e-6
    features={}
    for sp in ('held','fit'):
        selected=np.flatnonzero([(c['split']==sp and mode==0) for c,mode in zip([cards[j] for j in d['card']],d['prefix_mode'])])
        features[sp]={}
        for name,n,adapter in [('parent_random_adapter',initnet,initp),('spatial_fitted_adapter',candidate,proj),('raw_soft_same_fitted_adapter',matched,proj)]:
            features[sp][name]=feature_test(n,adapter,labels,d,needle,selected,cards)
    np.savez_compressed(HERE/'training_adapter.npz',**{k:v.detach().numpy() for k,v in proj.state_dict().items()})
    assert candidate.state_dict().keys()==matched.state_dict().keys()
    diff={k:int((v!=matched.state_dict()[k]).sum()) for k,v in candidate.state_dict().items()}
    data=dict(protocol_sha256=identity,raw_soft_weight_sha256=R.sha(RAW/'student.npz'),raw_soft_pack_sha256=R.sha(RAW/'student.py'),weight_sha256=R.sha(w),pack_sha256=R.sha(HERE/'student.py'),adapter_sha256=R.sha(HERE/'training_adapter.npz'),fit_sha256=R.sha(Path(__file__)),matched_order_sha256=p['order_sha256'],teacher_channels=labels.shape[1],student_channels=net.aux.in_channels,roots=len(labels),n_updates=len(history),loss_history=history,changed_from_matched_raw_soft=diff,parity=parity,held=table,feature=features)
    R.dump(HERE/'fit.json',data)
    print('held KL',[(a,z['all']['map_equal_kl']) for a,z in table.items()],flush=True)
if __name__=='__main__':run()
