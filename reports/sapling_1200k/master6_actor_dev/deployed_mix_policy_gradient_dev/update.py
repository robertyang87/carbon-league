"""Differentiate the sampled real joint command mass through the deployed student's prefix."""
import base64,json,hashlib,inspect,math,sys
from pathlib import Path
import numpy as np
import torch
from study import HERE,PARENT,BASE,PLAN,arena,policy,loads,rooted,law,row_features,ACTOR
sys.path.insert(0,str(arena.ROOT/'code'))
import cnet2_model as model
import cnet2_train as trainer
trainer.DEV=torch.device('cpu');trainer.GPU_AUG=False
sys.path.insert(0,str(ACTOR/'blended_actor_dev/serving'))
import build as pack


def capture_student(root,slot):
    src=inspect.getsource(root.m._return_decode)
    hook='action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    if src.count(hook)!=1:raise ValueError('missing decoder hook')
    src=src.replace(hook,'capture(slot,scores[1].copy())\n        '+hook)
    out={};ns=dict(vars(root.m),capture=lambda i,s:out.update({i:s}));exec(src,ns)
    ns['_return_decode'](root.x,root.aux,root.enc,False)
    return out[slot]


def root_data(m,samples):
    rows=[];roots=[]
    for e in samples:
        if e.get('skip'):continue
        root,*_=rooted(m,e['seed'],e['seat'],e['t']); old=np.asarray(e['old_prob'])
        k=e['slot'];real=law(root,k)
        if max(abs(np.asarray(real['prob'])-old))>1e-8:raise ValueError('sampling mismatch')
        raw=row_features(root,k);rows.append(raw)
        ids=e['draws'];out=[e['outcomes'][e['executed'][str(a)]] for a in ids]
        returns=np.array([v['U']+.1*math.tanh(v['margin']/3000.) for v in out])
        adv=(returns-returns.mean())*len(returns)/(len(returns)-1)
        roots.append(dict(sample=e,root=root,adv=adv,old=old,numpy_z=capture_student(root,k)))
    if not roots or not any(abs(r['adv']).max()>1.e-8 for r in roots):raise ValueError('all terminal advantages constant')
    data={k:np.stack([np.asarray(r[k]) for r in rows]) for k in trainer.LOAD}
    for k,a in data.items():
        declared=trainer.FIELDS[k][0]
        # Leave live float features full precision; only the physical actor input is allowed.
        if np.dtype(declared)==np.float16:data[k]=a.astype(np.float32)
        else:data[k]=a.astype(declared)
    return trainer.to_dev(data,False),roots


def actual_check(m,samples):
    kl=[];rat=[];gain=[];lost=[]
    for z in samples:
        e=z['sample'];root,*_=rooted(m,e['seed'],e['seat'],e['t'])
        dist=root.distribution(e['slot'],np.zeros(5))
        old=z['old'];p=[];qq=[]
        for j,a in enumerate(old):
            if a<=0:continue
            order=e['executed'][str(j)];p.append(a);qq.append(dist.get(order,(0.,None))[0])
        if min(qq)<=0:
            lost.append([e['seed'],e['seat'],e['t']]);continue
        kl.append(float(np.dot(p,np.log(np.asarray(p)/np.asarray(qq)))))
        rr=np.array([dist[e['executed'][str(a)]][0]/old[a] for a in e['draws']])
        rat.extend(rr);gain.append(np.mean(rr*z['adv']))
    return dict(lost=lost,max_KL=max(kl,default=None),ratio_range=[float(min(rat)),float(max(rat))] if rat else None,
                surrogate_gain=float(np.mean(gain)) if gain else None)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    samples=arena.read(HERE/'sampling.json');m,_=loads()
    batch,roots=root_data(m,samples)
    with np.load(BASE,allow_pickle=False) as orig:
        meta=json.loads(str(orig['meta']));raw={k:np.asarray(orig[k]).copy() for k in orig.files if k!='meta'}
    with np.load(BASE,allow_pickle=False) as ref:
        if pack.quantize(BASE,ref)['_W64']!=m._W64:raise ValueError('pack student differs from FP32 parent')
    net=model.CNet2(**meta['cfg']);deq=m._NETS[1].w
    net.load_state_dict({k:torch.from_numpy(deq[k].copy()) for k in net.state_dict()})
    net.eval()
    for k,v in net.named_parameters():v.requires_grad_(not k.startswith(('aux.','value.','utility.','wdl.','margin.')))
    outputs=net(batch)[0];loss=[];errors=[]
    for i,r in enumerate(roots):
        e=r['sample'];k=e['slot'];old=r['old'];mask=old>0;ids=e['draws']
        z=outputs[i,k];np_scores=r['numpy_z'][mask];pt=z.detach().numpy()[mask]
        errors.append(float(abs((pt-pt[0])-(np_scores-np_scores[0])).max()))
        corrected=torch.where(torch.from_numpy(mask),torch.tensor(np.log(np.maximum(old,1.e-300)),dtype=z.dtype)+.25*(z-z.detach()),torch.tensor(-1.e9))
        lp=corrected.log_softmax(-1)
        if np.max(abs(lp.detach().numpy()[ids]-np.array(e['logp'])))>1.e-5:raise ValueError('logp did not roundtrip')
        adv=torch.tensor(r['adv'],dtype=z.dtype)
        loss.append(-(adv*lp[ids]).mean())
    if max(errors)>.02:raise ValueError(('Numpy/torch student legal-logit differences',max(errors)))
    torch.stack(loss).mean().backward()
    grads={k:v.grad.detach().numpy().copy() for k,v in net.named_parameters() if v.grad is not None}
    first=np.linalg.norm(grads['blocks.0.c1.weight'])
    if not np.isfinite(first) or first<=0:raise ValueError('no terminal gradient to backbone')
    attempts=[];accepted=False;source=PARENT.read_text()
    for lr in (.00015,.000075,.0000375):
        weights={k:v-lr*grads[k]/(np.abs(grads[k])+1.e-8) if k in grads else v for k,v in raw.items()}
        if any(not np.isfinite(v).all() for v in weights.values()):raise ValueError('nonfinite')
        archive=HERE/'trial_weights.npz';np.savez(archive,**weights,meta=np.array(json.dumps(meta)))
        with np.load(BASE,allow_pickle=False) as ref:embedded=pack.quantize(archive,ref)
        p=HERE/'trial_agent.py';p.write_text(pack.substitute(source,embedded));candidate=policy.load(p)
        if candidate._B1!=m._B1 or candidate._Q1!=m._Q1 or pack.substitute(source,embedded)!=p.read_text():raise ValueError('nonstudent pack drift')
        test=actual_check(candidate,roots);attempts.append(dict(lr=lr,**test))
        if (not test['lost'] and test['max_KL']<.01 and test['ratio_range'][0]>=.75 and test['ratio_range'][1]<=1.25 and test['surrogate_gain']>1.e-10):
            archive.replace(HERE/'candidate.npz');p.replace(HERE/'candidate.py');accepted=True;break
    arena.write(HERE/'update.json',dict(accepted=accepted,attempts=attempts,non_skipped_roots=len(roots),samples=sum(len(z['sample']['draws']) for z in roots),
        changed_returns_roots=sum(float(np.ptp(z['adv']))>1e-8 for z in roots),shared_gradient_norm=float(first),numpy_torch_max_legal_logit_delta=max(errors),
        trainable_gradient_parameters=sum(np.count_nonzero(g) for g in grads.values()),changed_fp32_parameters=sum(np.count_nonzero(v!=raw[k]) for k,v in weights.items()),
        model_sha256=arena.sha(HERE/'candidate.npz') if accepted else None,agent_sha256=arena.sha(HERE/'candidate.py') if accepted else None))
    print('accepted',accepted,attempts,flush=True)
if __name__=='__main__':main()
