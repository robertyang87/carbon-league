"""Isolate the effect of a projected policy vector versus its modal command in Sapling's full policy."""
from collections import Counter, defaultdict
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
tool=importlib.util.spec_from_file_location('sapling_training_components',ACTOR/'train.py')
base=importlib.util.module_from_spec(tool)
tool.loader.exec_module(base)
PARENT=base.WEIGHTS/'master6_actor_v1.npz'
PARENT_HASH='f82bfd47a904aa1fbfa4e95fbfcdd13e623c1f1eb216a8e736694a1d282cc713'
NOT_POLICY=('value.','utility.','wdl.','margin.','aux.')
MODES=('history','teacher_greedy')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,obj):
    data=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    if path.suffix=='.gz':data=gzip.compress(data,mtime=0)
    tmp=path.with_suffix('.next');tmp.write_bytes(data);tmp.replace(path)


def read_panel():
    manifest=HERE/'manifest.json'
    cards=json.loads(manifest.read_bytes())['records']
    cols=defaultdict(list)
    for j,c in enumerate(cards):
        f=HERE/c['file'];assert f.is_file() and sha(f)==c['sha256']
        with np.load(f,allow_pickle=False) as d:
            n=len(d['y'])
            assert c['split'] in ('fit','held') and c['domain'] in ('original','sapling_visited','elite')
            assert n>0 and not np.any(d['gid']!=c['gid']) and np.all(d['teacher']==2)
            y,q,legal=d['y'],d['q'],d['legal']
            mask=y>=0
            assert q.shape==legal.shape==(*y.shape,5) and y.shape[1]==12
            assert q.dtype==np.float32 and np.isfinite(q).all() and np.all(q>=0)
            assert np.all(q[~legal]==0) and np.allclose(q.sum(-1),mask,atol=1e-5,rtol=0)
            assert np.array_equal(mask,d['tok_mask'][:,:12]) and legal[np.where(mask)+(y[mask],)].all()
            assert d['src_act'].shape==y.shape and np.all((d['src_act']>=-1)&(d['src_act']<5))
            mode=d['prefix_mode']; assert mode.shape==(n,)
            steps=[int(v) for v in c['steps']]
            assert len(set(steps))==len(steps) and all(Counter(d['step'][mode==x])==Counter(steps) for x in (0,1))
            for k in (*base.trainer.FIELDS,'q','src_act','prefix_mode'):
                cols[k].append(d[k])
            cols['card'].append(np.full(n,j,dtype=np.int64))
    assert len({c['id'] for c in cards})==len(cards) and len({c['gid'] for c in cards})==len(cards)
    split={sp:{c['terrain'] for c in cards if c['split']==sp} for sp in ('fit','held')}
    assert split['fit'] and split['held'] and not split['fit'] & split['held']
    return {k:np.concatenate(v) for k,v in cols.items()},cards


def batch_order(d,cards,protocol):
    train=defaultdict(list)
    for i,j in enumerate(d['card']):
        if cards[j]['split']=='fit' and ((d['y'][i]>=0)&(d['legal'][i].sum(-1)>1)).any():
            train[cards[j]['terrain']].append(i)
    rng=np.random.default_rng(protocol['seed'])
    rows={g:list(rng.permutation(v)) for g,v in train.items()}
    order=[]
    while len(order)<protocol['updates']*protocol['roots_per_update']:
        for g in rng.permutation(sorted(train)):
            if not rows[g]:rows[g]=list(rng.permutation(train[g]))
            order.append(int(rows[g].pop()))
    order=np.asarray(order[:protocol['updates']*protocol['roots_per_update']],dtype='<i8')
    return order.reshape(protocol['updates'],protocol['roots_per_update'])


def fitted(name,d,order,proto,identity,checkpoint_dir):
    net,meta=base.model(PARENT,PARENT_HASH)
    initial={k:v.detach().clone() for k,v in net.named_parameters()}
    params=[]
    for k,v in net.named_parameters():
        v.requires_grad_(not k.startswith(NOT_POLICY))
        if v.requires_grad:params.append(v)
    opt=torch.optim.AdamW(params,lr=proto['learning_rate'],weight_decay=proto['weight_decay'])
    cp=checkpoint_dir/(name+'.pt')
    losses=[]
    if cp.is_file():
        prior=torch.load(cp,map_location='cpu',weights_only=True)
        if prior['identity']!=identity or prior['arm']!=name:raise ValueError('training checkpoint differs')
        net.load_state_dict(prior['model']);opt.load_state_dict(prior['opt']);losses=prior['losses']
    net.train()
    for step,ix in enumerate(order[len(losses):],len(losses)):
        b=base.batch(d,ix)
        logits=net(b)[0]
        if not torch.isfinite(logits).all():raise ValueError('nonfinite scores')
        mask=(b['y']>=0)&(b['legal'].sum(-1)>1)
        q=torch.from_numpy(d['q'][ix])
        log_probs=logits.log_softmax(-1)
        if name=='student':per=-(q*log_probs).sum(-1)
        else:per=-log_probs.gather(-1,q.argmax(-1,keepdim=True)).squeeze(-1)
        losses.append(float(per[mask].mean().detach()))
        loss=per[mask].mean()
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,proto['clip_norm'],error_if_nonfinite=True)
        opt.step()
        if (step+1)%8==0:
            tmp=cp.with_suffix('.tmp.pt')
            torch.save(dict(identity=identity,arm=name,model=net.state_dict(),opt=opt.state_dict(),losses=losses),tmp)
            tmp.replace(cp)
    changed={}
    for k,v in net.named_parameters():
        n=int((v.detach()!=initial[k]).sum())
        if k.startswith(NOT_POLICY):assert n==0
        name0=k.partition('.')[0]
        changed[name0]=changed.get(name0,0)+n
    assert all(changed[k]>0 for k in ('stem','blocks','attn','gru','U','whead','bhead'))
    path=HERE/(name+'.npz');tmp=HERE/(name+'.next.npz')
    base.trainer.export(net,str(tmp),dict(meta,distill=name,master3s2_soft_vs_argmax=identity))
    tmp.replace(path)
    return {'file':path.name,'sha256':sha(path),'losses':losses,'changed_per_module':changed,
            'trained_parameters':sum(p.numel() for p in params),
            'network_parameters':sum(p.numel() for p in net.parameters())}


def score(d,cards,idx,preds):
    y=d['y'][idx];src=d['src_act'][idx];q=d['q'][idx]
    valid=(y>=0)&(d['legal'][idx].sum(-1)>1)
    cids=d['card'][idx]; groups={c['terrain'] for c in cards if c['split']=='held'}
    domains=np.asarray([cards[j]['domain'] for j in cids])
    terrain=np.asarray([cards[j]['terrain'] for j in cids]);mode=d['prefix_mode'][idx]
    kind=d['tok_kind'][idx,:12]
    result={}
    for arm,p in preds.items():
        hist=np.zeros_like(q,dtype=np.float64)
        np.log(q,out=hist,where=q>0)
        kl=(q*(hist-np.log(np.maximum(p,1e-30)))).sum(-1)
        hit=p.argmax(-1)==src; agree=p.argmax(-1)==q.argmax(-1)
        def aggregate(mask):
            maps={}
            for g in sorted(groups):
                keep=mask&(terrain==g)[:,None]
                if keep.any():
                    ok=keep&(src>=0)
                    maps[g]={'slots':int(keep.sum()),'soft_kl':float(kl[keep].mean()),'target_modal_agreement':float(agree[keep].mean()),
                             'source_slots':int(ok.sum()),'source_exact':float(hit[ok].mean()) if ok.any() else None}
            if not maps:return {'slots':0,'map_count':0}
            src_metrics=[v['source_exact'] for v in maps.values() if v['source_exact'] is not None]
            return {'slots':sum(v['slots'] for v in maps.values()),'map_count':len(maps),'by_terrain':maps,
                    'map_equal_kl':float(np.mean([v['soft_kl'] for v in maps.values()])),
                    'map_equal_target_modal':float(np.mean([v['target_modal_agreement'] for v in maps.values()])),
                    'map_equal_source_agreement':float(np.mean(src_metrics)) if src_metrics else None}
        masks={'all':valid,'original_historical_prefix':valid&((domains=='original')&(mode==0))[:,None],
               **{f'{o}_prefix':valid&(mode==i)[:,None] for i,o in enumerate(MODES)},
               **{r:valid&(kind==i) for i,r in enumerate(('collector','planter','base'))},
               **{domain: valid&(domains==domain)[:,None] for domain in sorted(set(domains))}}
        result[arm]={n:aggregate(v) for n,v in masks.items()}
    return result


def parity(weights,data,idx):
    """Numpy int8-dequantized deployment math and Torch for frozen legal prefixes and identical stored features."""
    import cnet2_np
    with np.load(weights,allow_pickle=False) as z:pol=cnet2_np.Policy(dict(z))
    net,_=base.model(weights);net.eval()
    errors=[];ranks=[];nt=0
    with torch.inference_mode():
        torch_logits=net(base.batch(data,idx))[0].numpy()
    for i,g in enumerate(idx):
        x={k:(v[g].astype('float32') if v.dtype==np.float16 else v[g]) for k,v in data.items()}
        for k in ('tok_kind','tok_pos','hist_cell','hist_move'):x[k]=x[k].astype('int64')
        tokens,ctx=pol.encode(x,int(x['teacher']))
        h=pol.lin(ctx,'h0');prev=8
        for s,a in enumerate(x['y']):
            if a<0:continue
            uf,mf,legal=x['uf'][s],x['mf'][s],x['legal'][s]
            inp=pol.lin(np.concatenate([tokens[s],ctx,uf,mf.ravel()]),'din')+pol.w['aemb.weight'][prev]
            h=pol.gru(inp,h)
            logits=pol.lin(h,'whead') if s<10 else np.pad(pol.lin(h,'bhead'),(0,2))
            logits=logits+pol.lin(mf,'V') @ pol.lin(h,'U')
            actual,expected=logits[legal],torch_logits[i,s][legal]
            errors.append(float(np.max(abs(actual-expected))));ranks.append(np.argmax(actual)==np.argmax(expected));nt+=len(expected)
            prev=int(a)+(0 if s<10 else 5)
    return dict(roots=len(idx),slots=len(errors),legal_logits=nt,max_abs=max(errors),argmax_differences=int(np.count_nonzero(~np.asarray(ranks))))


def run():
    assert os.sched_getaffinity(0)=={3} and os.getpriority(os.PRIO_PROCESS,0)==19 and base.pack.QMODE=='int8'
    assert sha(PARENT)==PARENT_HASH
    proto=json.loads((HERE/'fit_protocol.json').read_bytes());data,cards=read_panel()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(proto['seed'])
    order=batch_order(data,cards,proto)
    identity={'fit_sha256':sha(Path(__file__)),'manifest_sha256':sha(HERE/'manifest.json'),'parent_sha256':PARENT_HASH,
              'training_protocol_sha256':sha(HERE/'fit_protocol.json'),'order_sha256':hashlib.sha256(order.tobytes()).hexdigest()}
    done=HERE/'fit.json'
    if done.exists():
        old=json.loads(done.read_bytes())
        assert old['identity']==identity and all(sha(HERE/v['file'])==v['sha256'] for v in old['weights'].values())
        return
    checkpoint_dir=Path('/workspace')/('sapling-distill-checkpoints-'+identity['manifest_sha256'][:12])
    checkpoint_dir.mkdir(exist_ok=True)
    netweights={}
    for arm in ('control','student'):
        torch.manual_seed(proto['seed'])
        print('optimizing',arm,flush=True)
        netweights[arm]=fitted(arm,data,order,proto,identity,checkpoint_dir)
    ix=np.flatnonzero([cards[i]['split']=='held' for i in data['card']]);held={k:v[ix] for k,v in data.items()}
    preds={}
    parities={}
    parity_ix=[]
    for terrain in sorted({c['terrain'] for c in cards if c['split']=='held'}):
        eligible=[i for i in ix if cards[data['card'][i]]['terrain']==terrain]
        for mode in (0,1):
            roots=[i for i in eligible if data['prefix_mode'][i]==mode]
            parity_ix.append(roots[len(roots)//2])
    for arm,p in [('parent',PARENT),('control',HERE/'control.npz'),('student',HERE/'student.npz')]:
        full,_=base.model(p)
        net=base.quantized(full)
        preds[arm]=base.predict(net,held)
        if arm!='parent':
            with tempfile.TemporaryDirectory() as tmp:
                stage=Path(tmp);fp=stage/p.name
                fp.write_bytes(p.read_bytes())
                pack_src=stage/(arm+'.py')
                base.pack.main(str(fp),str(pack_src),teacher=2)
                qpath=stage/(arm+'_q8.npz')
                assert pack_src.stat().st_size<10_000_000
                netweights[arm].update(pack_sha256=sha(pack_src),pack_bytes=pack_src.stat().st_size,
                                       int8_dequantized_sha256=sha(qpath))
                parities[arm]=parity(qpath,data,parity_ix)
                assert parities[arm]['max_abs']<5.e-4 and parities[arm]['argmax_differences']==0,parities[arm]
                (HERE/(arm+'.py')).write_bytes(pack_src.read_bytes())
    metric=score(data,cards,ix,preds);student=metric['student']['all'];ctl=metric['control']['all'];par=metric['parent']['all']
    positive=sum(v['soft_kl']<ctl['by_terrain'][g]['soft_kl'] for g,v in student['by_terrain'].items())
    checks={'held_soft_kl_vs_parent':student['map_equal_kl']<par['map_equal_kl'],
            'held_soft_kl_vs_control':student['map_equal_kl']<ctl['map_equal_kl'],
            'held_pos_maps_at_least_half':2*positive>=student['map_count'],
            'historical_prefix_original_agreement_within_2pp':metric['student']['original_historical_prefix']['map_equal_source_agreement']>=metric['parent']['original_historical_prefix']['map_equal_source_agreement']-.02}
    dump(done,{'identity':identity,'weights':netweights,'protocol':proto,'held_metric':metric,'held_scorer':'int8 channel quantizer, dequantized frozen tensors',
               'held_map_kl_gains_vs_control':positive,'common_prefix_gate':checks,'common_prefix_pass':all(checks.values()),
               'numpy_prefix_parity':parities,'rows_by_split':dict(Counter(cards[j]['split'] for j in data['card'])),
               'all_games':len(cards),'order_gids':[int(data['gid'][j]) for j in order.ravel()],
               'order_rows':order.tolist(),'new_native_games':0})
    print('held KL', {k:metric[k]['all']['map_equal_kl'] for k in metric},'checks',checks,flush=True)


if __name__=='__main__':
    with (HERE/'.distill.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
