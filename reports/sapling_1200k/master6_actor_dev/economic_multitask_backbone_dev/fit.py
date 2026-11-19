"""Train the same full policy on the same ordered BC batches, varying only future economy supervision."""
import copy,hashlib,io,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;BROAD=HERE.parent/'broader_elite_backbone_dev'
sys.path.insert(0,str(BROAD))
import train as bc
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,doc):
    p=HERE/name;raw=json.dumps(doc,indent=2,allow_nan=False).encode()+b'\n';tmp=p.with_suffix('.tmp');tmp.write_bytes(raw);os.replace(tmp,p)
def targets(data):
    if sha(HERE/'labels_25.npz')!=json.loads((HERE/'economic_labels.json').read_bytes())['labels_sha256']:raise ValueError('label integrity')
    with np.load(HERE/'labels_25.npz',allow_pickle=False) as z:
        ix={(int(g),int(t)):row for g,t,row in zip(z['gid'],z['step'],z['target'])}
        if len(ix)!=len(z['gid']):raise ValueError('nonunique econ keys')
        y=np.stack([ix[int(g),int(t)] for g,t in zip(data['gid'],data['step'])])
    return torch.from_numpy(np.arcsinh(y/np.array([200,100,100,100,100],np.float32)))
def initialize():
    p=bc.check();torch.manual_seed(p['seed']);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    oracle=bc.ref.expert_model().eval();base,meta=bc.common.model(bc.START,p['parent_npz_sha256']);base.eval().requires_grad_(False)
    net=copy.deepcopy(base);frozen=('aux.','utility.','value.','wdl.','margin.')
    for k,v in net.named_parameters():v.requires_grad_(not k.startswith(frozen))
    return p,oracle,base,net,meta
def main():
    p,oracle,base,net,meta=initialize()
    cards=bc.assemble();train=[g for g in cards if g['split']=='train']
    data=bc.ref.load(train,bc.SHARDS);yfuture=targets(data)
    if (torch.count_nonzero(yfuture,dim=0)<100).any():raise ValueError('each training outcome must be observable and nondegenerate')
    if data['value'].any() or data['carbon'].any():raise ValueError('privileged actor input')
    phases=np.minimum(3,data['step']//75);weights=np.zeros(len(phases),np.float32)
    for g in train:
        group=data['gid']==g['gid'];phaseids=np.unique(phases[group])
        for phase in phaseids:
            mask=group&(phases==phase);weights[mask]=len(weights)/(len(train)*len(phaseids)*int(mask.sum()))
    order=np.random.default_rng(p['seed']).permutation(len(weights))
    live={k:v for k,v in net.named_parameters() if v.requires_grad}
    head=nn.Linear(net.value.in_features,5)
    # Capture exactly the already-computed context used by the trained policy and scalar value.
    ctx=[];net.value.register_forward_pre_hook(lambda mod,inputs:ctx.append(inputs[0]))
    opt=torch.optim.AdamW(list(live.values())+list(head.parameters()),lr=p['lr'],weight_decay=.001)
    logs=[];t0=time.monotonic();gradinfo={}
    quantiles=np.stack([np.median(yfuture.numpy()[phases==f],axis=0) for f in range(4)])
    for pos in range(0,len(order),p['batch']):
        i=order[pos:pos+p['batch']];b=bc.common.batch(data,i);y=b['y'];mask=bc.common.actionable(b);role=b['tok_kind'][:,:y.shape[1]]
        with torch.no_grad():
            pl=bc.logits(base,b).log_softmax(-1);q=pl.exp();expert=bc.logits(oracle,b)
        ctx.clear();z=bc.logits(net,b);sl=z.log_softmax(-1)
        ce=bc.mean_roles(bc.policy_terms(z,y),mask,role);mix=bc.mean_roles(bc.policy_terms(.75*expert+.25*z,y),mask,role)
        kl=bc.mean_roles((q*(pl-sl)).sum(-1),mask,role);w=torch.from_numpy(weights[i])
        if len(ctx)!=1:raise ValueError('context hook failed')
        econ=F.smooth_l1_loss(head(ctx[0]),yfuture[i],reduction='none').mean(1)
        loss=(w*(ce+.5*mix+.16*kl+.15*econ)).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        if pos==0:
            sample=next((k for k in live if k.startswith('blocks.') and k.endswith('weight')),None)
            if sample is None:raise ValueError('no residual block')
            v=torch.autograd.grad((w*econ).mean(),live[sample],retain_graph=True)[0];g=float(v.norm())
            if not np.isfinite(g) or g<=0:raise ValueError('no auxiliary gradient to backbone')
            gradinfo=dict(shared_backbone_tensor=sample,auxiliary_gradient_l2=g)
        opt.zero_grad(set_to_none=True);loss.backward()
        m=torch.nn.utils.clip_grad_norm_(live.values(),1.,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
        logs.append([float(econ.detach().mean()),float(ce.detach().mean()),float(m)])
        if len(logs)%50==0:print(len(logs),'updates',round(time.monotonic()-t0,1),'s',np.mean(logs[-50:],axis=0),flush=True)
    out=HERE/'candidate.npz';bc.common.trainer.export(net,str(out),dict(meta,training_status='future_economy_aux_full_actor'))
    np.savez_compressed(HERE/'offline_head.npz',w=head.weight.detach().numpy(),b=head.bias.detach().numpy(),train_phase_median=quantiles)
    old,original,expert=bc.pack.validate_baseline()
    with np.load(bc.START,allow_pickle=False) as initial:values=bc.pack.quantize(out,initial)
    package=bc.pack.substitute(old,values)
    if bc.pack.substitute(package,original)!=bc.pack.substitute(old,original):raise ValueError('non-weight difference')
    (HERE/'candidate.py').write_text(package)
    reloaded,_=bc.common.model(out);changed={}
    for k,v in reloaded.named_parameters():
        if not torch.equal(v,dict(net.named_parameters())[k]):raise ValueError('reload mismatch')
        n=changed.setdefault(k.split('.')[0],0);changed[k.split('.')[0]]=n+int((v!=dict(base.named_parameters())[k]).sum())
    write('fit.json',dict(**gradinfo,train_maps=len(train),roots=len(order),steps=len(logs),elapsed=time.monotonic()-t0,
        sample_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),weights=sha(out),package=sha(HERE/'candidate.py'),
        protocol=sha(HERE/'protocol.json'),dataset=sha(HERE/'labels_25.npz'),parent=sha(bc.START),control_recipe=sha(BROAD/'recipe.json'),
        train_code_sha256=sha(Path(__file__)),changed_by_group=changed,training_only_head_parameters=sum(v.numel() for v in head.parameters()),
        first50=np.mean(logs[:50],axis=0).tolist(),last50=np.mean(logs[-50:],axis=0).tolist(),fixed_expert_sha256=expert))
    print('DONE',flush=True)
if __name__=='__main__':main()
