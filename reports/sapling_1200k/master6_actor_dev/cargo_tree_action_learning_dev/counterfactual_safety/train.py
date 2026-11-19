"""Matched, map-crossfitted student learning, standard inference re-export, no runtime auxiliary head."""
import collections,hashlib,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gather import HERE,EXP,CFG,read,sha,write,policy

sys.path.insert(0,str(EXP.parent/'blended_actor_dev/serving'))
import build as pack
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def logp(z,mask):return z.masked_fill(~mask,-1e6).log_softmax(1)
def seedorder(ids,epochs):return [np.random.default_rng(9276+e).permutation(ids) for e in range(epochs)]

class Predictor(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(d,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3))
    def forward(self,x):return self.layers(x)

def train_teacher(x,ys,legal,factual,train,targets,all_arms):
    torch.manual_seed(7782);m=Predictor(x.shape[-1]);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.004)
    center=x[train].reshape(-1,x.shape[-1]).mean(0);scale=x[train].reshape(-1,x.shape[-1]).std(0).clamp_min(.05)
    xx=((x-center)/scale).clamp(-6,6)
    for order in seedorder(train,24):
        for chunk in np.array_split(order,int(np.ceil(len(order)/64))):
            p=m(xx[chunk]); y=ys[chunk]; observed=legal[chunk] if all_arms else F.one_hot(factual[chunk],5).bool()
            val=F.binary_cross_entropy_with_logits(p[:,:,0],y[:,:,0],reduction='none')+2*(p[:,:,1:].sigmoid()*1.5-y[:,:,1:]).square().sum(2)
            loss=((val*observed).sum(1)/observed.sum(1)).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    with torch.no_grad():
        val=m(xx[targets]).sigmoid();val[:,:,1:]*=1.5
    return val

def run():
    rec=read(HERE/'dataset.json');rows=read(HERE/'roots.json.gz')
    assert sha(HERE/'pairs.npz')==rec['arrays'] and sha(HERE/'roots.json.gz')==rec['roots']
    with np.load(HERE/'pairs.npz',allow_pickle=False) as a:
        h,move,proj,logits,z,mask,y=[torch.as_tensor(a[k]) for k in ('h','move','proj','logits','mixed','mask','labels')]
    n=len(rows);assert n==len(h) and not torch.isnan(y).any()
    idx={k:np.array([i for i,r in enumerate(rows) if r['split']==k]) for k in ['train','val','test']}
    loc={k:set(rows[i]['terrain'] for i in v) for k,v in idx.items()};assert not (loc['train']&loc['val'] or loc['test']&(loc['train']|loc['val']))
    assert [len(loc[k]) for k in ['train','val','test']]==[19,4,6]
    tags=np.array([r['tag'] for r in rows]); tree=tags=='tree';cargo=torch.tensor([r['cargo'] for r in rows],dtype=torch.float32)
    fact=torch.tensor([r['parent'] for r in rows]);true=y.clone()
    # Signals are settlement components of the same acting worker. Supervision is limited to one native simultaneous transition.
    # High cargo cannot be augmented artificially by stealing from a teammate while scoring defensive survival.
    prior=cargo[:,None]
    net_asset=y[:,:,1]+y[:,:,2]
    recover=torch.where(torch.tensor(tags=='harvest')[:,None],net_asset,torch.minimum(net_asset,prior))
    norm=(prior+40.).clamp_min(40.)
    Y=torch.stack([y[:,:,0],(recover/norm).clamp(0,1.5),(y[:,:,2]/norm).clamp(0,1.5)],dim=2)
    x=torch.cat([h.unsqueeze(1).expand(-1,5,-1),move,F.one_hot(torch.arange(5),5).float().expand(n,-1,-1),
                 logits[:,1,:].unsqueeze(2),(cargo/(cargo+60.)).view(-1,1,1).expand(-1,5,-1)],2)
    orig=EXP/'candidate.npz';basefile=EXP/'candidate.py';net=policy.load(basefile)
    with np.load(orig,allow_pickle=False) as a:
        original={k:a[k] for k in a.files};assert pack.quantize(orig,a)['_W64']==net._W64
    tt=idx['train'];train_actors=tt[~tree[tt]];split_fold={v:i%3 for i,v in enumerate(sorted(loc['train']))}
    trainmap=np.array([split_fold.get(r['terrain'],-1) for r in rows])
    outputs={};predictions={};methods={'control':False,'candidate':True}
    pref_net=logp(logits[:,1],mask);pref_mix=logp(z,mask);eye=torch.arange(n)
    for arm,observed_all in methods.items():
        pred=torch.zeros((n,5,3));assigned=np.zeros(n,bool)
        for f in range(3):
            training=train_actors[trainmap[train_actors]!=f]
            destination=tt[trainmap[tt]==f];pred[destination]=train_teacher(x,Y,mask,fact,training,destination,observed_all);assigned[destination]=True
        held=np.concatenate([idx['val'],idx['test']]);pred[held]=train_teacher(x,Y,mask,fact,train_actors,held,observed_all);assigned[held]=True;assert assigned.all()
        predictions[arm]=pred.numpy()
        utility=3.5*pred[:,:,0]+2*pred[:,:,1]+1.25*pred[:,:,2]
        # Identical loss and budget in both arms; only teacher's observable alternative labels differ.
        utility[tree]=0.
        target_student=logp(pref_net+2.5*utility,mask).exp().detach()
        target_mix=logp(pref_mix+1.0*utility,mask).exp().detach()
        dw=nn.Parameter(torch.zeros_like(torch.as_tensor(original['U.weight'])));db=nn.Parameter(torch.zeros_like(torch.as_tensor(original['U.bias'])))
        opt=torch.optim.Adam([dw,db],lr=.002)
        weights=collections.Counter((r['terrain'],r['tag']) for r in rows if r['split']=='train')
        balance=torch.tensor([len(tt)/(len(weights)*weights[(r['terrain'],r['tag'])]) if r['split']=='train' else 1. for r in rows])
        for ea in seedorder(tt,12):
            for ids in np.array_split(ea,int(np.ceil(len(ea)/96))):
                delta=((h[ids]@dw.T+db)[:,None,:]*proj[ids]).sum(2)
                policy_logits=logp(logits[ids,1]+delta,mask[ids]);mix_logits=logp(z[ids]+.25*delta,mask[ids])
                loss=(-(target_student[ids]*policy_logits+target_mix[ids]*mix_logits).sum(1)*balance[ids]).mean()+.003*dw.square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([dw,db],2.,error_if_nonfinite=True);opt.step()
        raw=original.copy();raw['U.weight']=original['U.weight']+dw.detach().numpy();raw['U.bias']=original['U.bias']+db.detach().numpy()
        fn=HERE/(arm+'.npz');py=HERE/(arm+'.py');np.savez(fn,**raw)
        with np.load(orig,allow_pickle=False) as a: embedded=pack.quantize(fn,a)
        py.write_text(pack.substitute(basefile.read_text(),embedded))
        assert pack.substitute(py.read_text(),pack.literal(basefile.read_text(),pack.STUDENT))==basefile.read_text()
        new=policy.load(py);w=lambda model,k:torch.from_numpy(model._NETS[1].w[k].copy())
        diff=((h@(w(new,'U.weight')-w(net,'U.weight')).T+w(new,'U.bias')-w(net,'U.bias'))[:,None,:]*proj).sum(2)
        predictions[arm+'_student']=(logits[:,1]+diff).numpy();predictions[arm+'_mixed']=(z+.25*diff).numpy()
        breakdown={}
        for sp in ('val','test'):
            for tg in ('bank','threat','harvest'):
                ids=idx[sp][tags[idx[sp]]==tg];m=mask[ids];held_pred=pred[ids]
                loss=torch.stack([(held_pred[:,:,0]-Y[ids,:,0]).square(),(held_pred[:,:,1]-Y[ids,:,1]).abs(),(held_pred[:,:,2]-Y[ids,:,2]).abs()],2)
                breakdown[sp+'/'+tg]=dict(roots=len(ids),actions=int(m.sum()),survival_brier=float(loss[:,:,0][m].mean()),retained_mae=float(loss[:,:,1][m].mean()),deposit_mae=float(loss[:,:,2][m].mean()))
        outputs[arm]=dict(pack=sha(py),weights=sha(fn),bytes=py.stat().st_size,teacher=breakdown,max_updated=float(dw.detach().abs().max()),changed_parameters=sum(int(np.count_nonzero(raw[k]!=original[k])) for k in ('U.bias','U.weight')))
    np.savez_compressed(HERE/'crossfitted.npz',**predictions)
    stats={sp:{tg:int(np.sum(tags[ids]==tg)) for tg in set(tags)} for sp,ids in idx.items()}
    write(HERE/'fit.json',dict(parent=sha(basefile),source=sha(orig),data=sha(HERE/'dataset.json'),contract=sha(HERE/'contract.json'),crossfitted=sha(HERE/'crossfitted.npz'),samples=stats,outputs=outputs))
if __name__=='__main__':run()
