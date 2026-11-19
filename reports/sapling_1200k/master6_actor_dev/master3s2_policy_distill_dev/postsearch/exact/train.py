"""Train identical Sapling policy nets on raw marginals vs search-conditional final commands."""
import os,sys,tempfile,fcntl,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import dataset
sys.path.insert(0,str(HERE.parent.parent))
import fit as shared
base=shared.base

def load():
    manifest=dataset.read(HERE/'manifest.json');cards=manifest['cards'];columns=defaultdict(list)
    keys=set(base.trainer.LOAD)|{'q','src_act','eligible','team_changed','gid','step','prefix_repaired','unit_changed','pre_act'}
    for c in cards:
        path=HERE/c['file'];assert dataset.sha(path)==c['sha256']
        with np.load(path,allow_pickle=False) as shard:
            assert np.all(shard['gid']==c['gid']) and np.array_equal(shard['step'],np.arange(c['first_step'],c['last_step']+1))
            legal=shard['legal'];e=shard['eligible'];actual=shard['src_act']
            assert np.all(legal[np.where(e)+(actual[e],)]) and np.allclose(shard['q'].sum(-1)[e],1,atol=2e-6)
            assert not np.any(shard['q'][~legal])
            for k in keys:columns[k].append(shard[k])
    return {k:np.concatenate(v) for k,v in columns.items()},cards

def draws(d,cards,plan):
    runs=[c for c in cards if c['split']=='fit'];rng=np.random.default_rng(plan['seed'])
    buckets={c['gid']:list(rng.permutation(np.flatnonzero((d['gid']==c['gid'])&d['eligible'].any(-1)))) for c in runs}
    order=[]
    while len(order)<plan['updates']*plan['batch']:
        for j in rng.permutation(list(buckets)):
            if not buckets[j]:buckets[j]=list(rng.permutation(np.flatnonzero((d['gid']==j)&d['eligible'].any(-1))))
            order.append(int(buckets[j].pop()))
    return np.asarray(order[:plan['updates']*plan['batch']],dtype='<i8').reshape(-1,plan['batch'])

def train(d,idx,plan,arm):
    torch.manual_seed(plan['seed'])
    net,meta=base.model(shared.PARENT,shared.PARENT_HASH)
    params=[p for k,p in net.named_parameters() if p.requires_grad_(not k.startswith(shared.NOT_POLICY)).requires_grad]
    opt=torch.optim.AdamW(params,lr=plan['lr'],weight_decay=.0001);losses=[];net.train()
    for batch in idx:
        b=base.batch(d,batch);logits=net(b)[0]
        mask=torch.from_numpy(d['eligible'][batch])
        q=torch.from_numpy(d['q'][batch]);actual=torch.from_numpy(d['src_act'][batch].astype(np.int64))
        logs=logits.log_softmax(-1);raw_loss=-(q*logs).sum(dim=-1)
        post_loss=-logs.gather(-1,actual.clamp_min(0)[...,None]).squeeze(-1)
        mask_replaced=d['unit_changed'][batch] if arm=='unit' else d['team_changed'][batch,None]
        mix=(0. if arm=='raw_control' else plan['residual_weight'])*torch.from_numpy(mask_replaced)
        loss=(raw_loss+mix*(post_loss-raw_loss))[mask].mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        losses.append(float(loss.detach()))
    net.eval();return net,meta,losses

def comparison(d,cards,index,prob):
    target=d['src_act'][index];eligible=d['eligible'][index];changed=d['team_changed'][index]
    chosen=prob.argmax(-1)
    p_exec=prob[np.arange(len(index))[:,None],np.arange(12),np.maximum(target,0)]
    nll=-np.log(p_exec.clip(1e-30));q=d['q'][index]
    kl=(q*(np.log(np.maximum(q,1e-30))-np.log(np.maximum(prob,1e-30)))).sum(-1)
    result={}
    for c in cards:
        valid=d['gid'][index]==c['gid']
        if not valid.any():continue
        blocks={}
        for group,filter_ in [('all',valid),('changed',valid&changed),('not_changed',valid&~changed)]:
            mask=filter_[:,None]&eligible
            valid_joint=filter_&~d['prefix_repaired'][index].any(-1)
            if not mask.any():continue
            blocks[group]=dict(slots=int(mask.sum()),roots=int(filter_.sum()),post_nll=float(nll[mask].mean()),raw_kl=float(kl[mask].mean()),
               slot_agreement=float(np.mean(chosen[mask]==target[mask])),full_joint_roots=int(valid_joint.sum()),
               joint_agreement=float(((chosen==target)|(target<0))[valid_joint].all(-1).mean()),
               by_role={name:dict(n=int(used.sum()),agreement=float(np.mean(chosen[used]==target[used]))) for i,name in enumerate(('collector','planter','base')) if (used:=mask&(d['tok_kind'][index,:12]==i)).any()})
        for kind, subset in [('replaced',d['unit_changed'][index]),('retained',~d['unit_changed'][index]),('retained_on_changed_team',(~d['unit_changed'][index])&changed[:,None])]:
            selected=valid[:,None]&eligible&subset
            blocks[kind]=dict(slots=int(selected.sum()),post_nll=float(nll[selected].mean()),raw_kl=float(kl[selected].mean()),slot_agreement=float(np.mean(chosen[selected]==target[selected])))
        result[c['gid']]=blocks
    return result

def run():
    assert os.sched_getaffinity(0)=={3} and os.getpriority(os.PRIO_PROCESS,0)==19
    torch.set_num_threads(1);torch.set_num_interop_threads(1);plan=dataset.read(HERE/'protocol.json');d,cards=load();idx=draws(d,cards,plan)
    identity=dict(protocol=dataset.sha(HERE/'protocol.json'),manifest=dataset.sha(HERE/'manifest.json'),parent=shared.PARENT_HASH,
                  training_script=dataset.sha(Path(__file__)),draws=hashlib.sha256(idx.tobytes()).hexdigest())
    if (HERE/'fit.json').exists():
        receipt=dataset.read(HERE/'fit.json');assert receipt['identity']==identity
        for arm in ('raw_control','team','unit'):
            for suffix in ('.npz','.py'):assert dataset.sha(HERE/(arm+suffix))==receipt['artifacts'][arm][suffix]
        return
    eval_idx=np.flatnonzero(np.isin(d['gid'],[c['gid'] for c in cards if c['split']!='fit']))
    test={k:v[eval_idx] for k,v in d.items()};preds={};artifacts={};losses={}
    for arm in ('parent','raw_control','team','unit'):
        print(arm,flush=True)
        if arm=='parent':
            net,_=base.model(shared.PARENT,shared.PARENT_HASH)
        else:
            net,meta,losses[arm]=train(d,idx,plan,arm)
            tmp=HERE/(arm+'.new.npz');base.trainer.export(net,str(tmp),dict(meta,postsearch_study=identity,arm=arm))
            path=HERE/(arm+'.npz');tmp.replace(path)
            with tempfile.TemporaryDirectory() as td:
                staged=Path(td)/path.name;staged.write_bytes(path.read_bytes())
                pack=Path(td)/(arm+'.py');base.pack.main(str(staged),str(pack),teacher=2)
                assert 0<pack.stat().st_size<=10_000_000
                (HERE/(arm+'.py')).write_bytes(pack.read_bytes())
            artifacts[arm]={ext:dataset.sha(HERE/(arm+ext)) for ext in ('.npz','.py')}
        prob=base.predict(base.quantized(net),test)
        preds[arm]=comparison(d,cards,eval_idx,prob)
    summary={}
    for split in ('held','diagnostic'):
        for terrain in sorted({c['terrain'] for c in cards if c['split']==split}):
            group=[c for c in cards if c['terrain']==terrain];maps={}
            for arm,values in preds.items():
                maps[arm]={kind:{m:float(np.mean([values[c['gid']][kind][m] for c in group]))
                                 for m in ('post_nll','raw_kl','slot_agreement','joint_agreement')} for kind in ('changed','not_changed','all')}
                for kind in ('replaced','retained','retained_on_changed_team'):
                    maps[arm][kind]={m:float(np.mean([values[c['gid']][kind][m] for c in group])) for m in ('post_nll','raw_kl','slot_agreement')}
            summary[group[0]['seed']]=dict(split=split,terrains=[terrain],arms=maps)
    held=[v['arms'] for v in summary.values() if v['split']=='held']
    gate=all(all(a['unit']['replaced']['post_nll']<a[v]['replaced']['post_nll'] for v in ('raw_control','team')) and
           a['unit']['retained']['slot_agreement']>=a['raw_control']['retained']['slot_agreement']-.02 for a in held)
    shared.dump(HERE/'fit.json',dict(identity=identity,fit_gids=sorted(set(d['gid'][idx].ravel().tolist())),losses=losses,
                            draws=idx.tolist(),fit_changed_roots=int(d['team_changed'][idx].sum()),metrics_by_gid=preds,summary=summary,
                            artifacts=artifacts,held_gate=bool(gate),native_matches_launched=0))
    print('held gate',gate,{s:round(v['arms']['unit']['replaced']['post_nll']-v['arms']['raw_control']['replaced']['post_nll'],5) for s,v in summary.items()},flush=True)
if __name__=='__main__':
    with (HERE/'.train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
