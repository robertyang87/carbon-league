"""Same-batch BC control and outcome-conditioned action learning in the exact existing head."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from collect import HERE,PARENT,BASE,CFG,arena,policy,save,sha,ACTOR
sys.path.insert(0,str(ACTOR/'blended_actor_dev/serving'));import build as pack

def peers(rows):
    tr=[(i,r) for i,r in enumerate(rows) if r['split']=='train']
    groups=collections.defaultdict(list)
    for i,r in tr:
        gj=tuple(r['group'])
        for depth in (3,2,1,0):
            key=(r['task'],r['is_master'],depth,gj[:depth]); groups[key].append((i,r))
    used=[];log=[]
    for i,r in tr:
        study=None;gj=tuple(r['group'])
        for depth in (3,2,1,0):
            others=[x for _,x in groups[r['task'],r['is_master'],depth,gj[:depth]] if x['terrain']!=r['terrain']]
            alt=[x for x in others if x['action_class']!=r['action_class']]
            if (len(alt)<8 or len({x['terrain'] for x in alt})<3 or not any(x['positive'] for x in others)
                    or not any(x['failed'] for x in others)):continue
            bymap=collections.defaultdict(list)
            for x in alt:bymap[x['terrain']].append(x['reward'])
            balanced=np.array([np.mean(a) for a in bymap.values()]);center=float(balanced.mean())
            scale=max(.35,float(np.std([x['reward'] for x in others])))
            advantage=float(np.clip((r['reward']-center)/scale,-2,2))
            study=dict(index=i,depth=depth,matched_alt=len(alt),other_maps=len(bymap),peer_reward=center,
                       advantage=advantage,peer_types=dict(collections.Counter(x['action_class'] for x in alt)))
            break
        if study:
            used.append(i);log.append(study)
    return used,log

def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=arena.read(HERE/'rows.json.gz');receipt=arena.read(HERE/'data.json')
    if sha(HERE/'rows.json.gz')!=receipt['row_sha'] or sha(HERE/'public_prefix_scores.npz')!=receipt['arrays_sha']:raise ValueError('data changed')
    ii,studies=peers(rows);save('matched_training.json.gz',studies)
    if min(len({rows[i]['terrain'] for i in ii}),len({r['terrain'] for r in rows if r['split']=='train'}))!=19:raise ValueError('lost a whole train map')
    with np.load(HERE/'public_prefix_scores.npz',allow_pickle=False) as z:
        hidden,base,legal=[torch.from_numpy(z[k][ii]) for k in ('hidden','logits','legal')]
    target=torch.tensor([rows[i]['target'] for i in ii]);adv=torch.tensor([r['advantage'] for r in studies],dtype=torch.float32)
    gate=legal[torch.arange(len(ii)),target]
    if not gate.all():raise ValueError('unsupported actual action')
    with np.load(BASE,allow_pickle=False) as bundle:orig={k:bundle[k] for k in bundle.files}
    m=policy.load(PARENT)
    with np.load(BASE,allow_pickle=False) as checker:
        if pack.quantize(BASE,checker)['_W64']!=m._W64:raise ValueError('unmatched initial weights')
    w0=torch.from_numpy(m._NETS[1].w['whead.weight'].copy());b0=torch.from_numpy(m._NETS[1].w['whead.bias'].copy())
    oldlp=base.masked_fill(~legal,-1e9).log_softmax(1);p0=oldlp.exp()
    strata=collections.Counter((rows[i]['task'],rows[i]['terrain'],rows[i]['is_master']) for i in ii)
    wt=torch.tensor([len(ii)/(len(strata)*strata[rows[i]['task'],rows[i]['terrain'],rows[i]['is_master']]) for i in ii],dtype=torch.float32)
    rng=np.random.default_rng(881735);order=np.stack([rng.permutation(len(ii)) for _ in range(8)])
    outputs={}
    for arm in ('bc','candidate'):
        dw=torch.nn.Parameter(torch.zeros_like(w0));db=torch.nn.Parameter(torch.zeros_like(b0))
        optimizer=torch.optim.Adam([dw,db],lr=.0025);journal=[]
        for epoch in order:
            vals=[]
            for j in range(0,len(epoch),128):
                batch=epoch[j:j+128];z=base[batch]+.25*(hidden[batch]@dw.T+db)
                lp=z.masked_fill(~legal[batch],-1e9).log_softmax(1)
                neg=-lp[torch.arange(len(batch)),target[batch]];kl=(p0[batch]*(oldlp[batch]-lp)).sum(1)
                signed=(1+adv[batch]) if arm=='candidate' else 1.
                loss=(wt[batch]*(signed*neg+.25*kl)).mean()+.002*(dw.square().mean()+db.square().mean())
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_([dw,db],1.,error_if_nonfinite=True);optimizer.step();vals.append(float(loss.detach()))
            journal.append(float(np.mean(vals)))
        weights=orig.copy()
        weights['whead.weight']=orig['whead.weight']+dw.detach().numpy()
        weights['whead.bias']=orig['whead.bias']+db.detach().numpy()
        archive=HERE/(arm+'.npz');np.savez(archive,**weights)
        with np.load(BASE,allow_pickle=False) as checker:embedded=pack.quantize(archive,checker)
        py=HERE/(arm+'.py');src=pack.substitute(PARENT.read_text(),embedded);py.write_text(src)
        if pack.substitute(src,pack.literal(PARENT.read_text(),pack.STUDENT))!=PARENT.read_text():raise ValueError('other deployment drift')
        outputs[arm]=dict(pack_sha256=sha(py),weights_sha256=sha(archive),lastloss=journal[-1],epoch_train_losses=journal,max_delta=float(dw.abs().max().detach()),
                          changed=sum(int(np.count_nonzero(weights[k]!=orig[k])) for k in ('whead.weight','whead.bias')))
    save('fit.json',dict(start=sha(BASE),parent=sha(PARENT),contract=sha(HERE/'contract.json'),matched=sha(HERE/'matched_training.json.gz'),
         root_rows=len(ii),map_count=len({rows[i]['terrain'] for i in ii}),by_task_role=dict(collections.Counter(rows[i]['task']+('/master' if rows[i]['is_master'] else '/sapling') for i in ii)),
         matching_depth=dict(collections.Counter(r['depth'] for r in studies)),positive_advantages=int((adv>0).sum()),negative_advantages=int((adv<0).sum()),
         gradient_updates_per_arm=len(order)*int(np.ceil(len(ii)/128)),order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
         deployment_changed_tensors=['whead.weight','whead.bias'],other_tensors_frozen=True,outputs=outputs))
if __name__=='__main__':run()
