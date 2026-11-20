"""Fit one three-block spatial student; score frozen true large-network targets with native team sampling."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
from torch.nn import functional as F

from network import Readout, sites, OBSERVED
from capture_spatial import PROTOCOL, original, module, sha, HERE, P


def write(path,data):
    raw=(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    path.write_bytes(raw)


def pack(split,cache):
    archive=HERE/(split+'_boards.zip')
    with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=3) as dest:
        for card in PROTOCOL['games']:
            if card['split']!=split:continue
            stem=card['sha256']; file=cache/(stem+'.npz')
            with np.load(file) as raw:
                meta=json.loads(gzip.decompress(raw['meta'].tobytes()))
                dest.writestr(stem+'.json',json.dumps(meta,separators=(',',':')))
                x=raw['x'];w=raw['w'];b=raw['b'];labels=[]
                for i,r in enumerate(meta['rows']):
                    seat=1-r['me'];p=r['players'][seat]
                    labels.extend(w[i,v[0]] for v in p[2].values())
                    labels.extend(np.pad(b[i,v],(0,2)) for v in p[1].values())
                buf=io.BytesIO();np.savez(buf,x=x,labels=np.stack(labels));dest.writestr(stem+'.npz',buf.getvalue())
    return archive


def extract(m,split,archive):
    xs=[];initials=[];zs=[];classes=[];rows=[];seen=set();cards=[];ablation=np.load(OBSERVED/'weights.npz')
    begin=time.perf_counter();teachers=[];aa={k:torch.as_tensor(ablation[k]) for k in ablation.files}
    with ZipFile(archive) as files:
        for card in PROTOCOL['games']:
            if card['split']!=split:continue
            k=card['sha256'];meta=json.loads(files.read(k+'.json')); assert meta['source']==k and meta['terrain']==card['terrain']
            with np.load(io.BytesIO(files.read(k+'.npz'))) as raw:
                x=raw['x'].astype(np.float32)*np.float32(.01);labels=raw['labels'];cursor=0
            for start in range(0,len(x),16):
                sub=x[start:start+16]
                rr,pp,patch,base,shallow=sites(m._NET,sub,m._SID,True)
                lw,lb,_=m._NET(sub,m._SID)
                old=torch.cat([shallow@aa['worker_weight']+aa['worker_bias'],shallow@aa['base_weight']+aa['base_bias']],1)
                pointer={(int(r),int(p)):i for i,(r,p) in enumerate(zip(rr,pp,strict=True))}
                for i in range(len(sub)):
                    r=meta['rows'][start+i];seat=1-r['me'];p=r['players'][seat];meta_units=[]
                    for key,knd in ((2,0),(1,1)):
                        for u,v in p[key].items():
                            pos=v[0] if key==2 else v
                            meta_units.append((u,knd,pos))
                    targets=labels[cursor:cursor+len(meta_units)];cursor+=len(meta_units)
                    assert r['hash']==sha(sub[i].tobytes())
                    if r['hash'] in seen:continue
                    seen.add(r['hash'])
                    first=len(zs);maxerr=0.
                    for j,(uid,knd,pos) in enumerate(meta_units):
                        ix=pointer[(i,pos)]
                        z=targets[j];truth=(lw[i,pos] if knd==0 else lb[i,pos]); n=len(truth)
                        maxerr=max(maxerr,float(abs(z[:n]-truth).max()))
                        xs.append(patch[ix].numpy().copy());initials.append(base[ix].numpy().copy())
                        zs.append(z.copy());classes.append(knd);teachers.append(old[ix].numpy().copy())
                    assert maxerr<.008
                    rows.append(dict(**r,terrain=card['terrain'],unit_slots=meta_units,begin=first,end=len(zs),numerical_check=maxerr))
            assert cursor==len(labels)
            cards.append(dict(seed=card['seed'],seat=card['candidate_seat'],terrain=card['terrain'],elapsed=meta['wall'],calls=meta['calls']))
    tensors=[torch.from_numpy(np.stack(v)) for v in (xs,initials,zs,classes,teachers)]
    return tensors,rows,seen,cards,time.perf_counter()-begin


def fit(tensors):
    x,ini,y,role,_=tensors
    torch.manual_seed(761105); head=Readout()
    head.mean.copy_(x.mean(0));head.scale.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.AdamW(head.parameters(),lr=.0012,weight_decay=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,55,eta_min=.00012)
    log=[]
    for epoch in range(55):
        losses=[]
        for ix in torch.randperm(len(x)).split(256):
            pred=head(x[ix],ini[ix]);loss=0
            for k,sl,n in [(0,slice(0,5),5),(1,slice(5,8),3)]:
                mask=role[ix]==k
                if not mask.any():continue
                w=pred[mask,sl];target=y[ix][mask,:n]
                for T in (1.,2.):
                    loss=loss+.25*T*T*F.kl_div(F.log_softmax(w/T,1),F.softmax(target/T,1),reduction='batchmean')
                centered=w-w.mean(1,keepdim=True);gold=target-target.mean(1,keepdim=True)
                loss=loss+.0075*F.smooth_l1_loss(centered,gold)
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        sched.step()
        if epoch in (0,4,14,29,54):log.append([epoch+1,float(np.mean(losses))]);print('epoch',log[-1],flush=True)
    return head,log


def report(m,rows,y,z,method):
    bymap=defaultdict(Counter);details=[]
    for r in rows:
        scores=bymap[r['terrain']]; first=r['begin'];end=r['end'];op=1-r['me'];p=r['players'][op]
        w=np.zeros((225,5),np.float32);b=np.zeros((225,3),np.float32);w0=w.copy();b0=b.copy()
        for j,(uid,k,pos) in enumerate(r['unit_slots'],first):
            tag='worker' if k==0 else 'base';v=z[j,:5] if k==0 else z[j,5:];gold=y[j,:len(v)]
            prob=torch.softmax(gold,0);kl=(prob*(torch.log_softmax(gold,0)-torch.log_softmax(v,0))).sum().item()
            scores[tag+'_kl_sum']+=kl;scores[tag+'_n']+=1
            (w if k==0 else b)[pos]=v.numpy();(w0 if k==0 else b0)[pos]=gold.numpy()
        sample=lambda w,b:m._sample(r['players'],op,w,b,np.random.RandomState(r['seed']),r['t'])
        gold=sample(w0,b0);assert gold==r['teacher'];pred=sample(w,b);scores['teams']+=1;scores['joint_hit']+=int(gold==pred)
        for uid,k,pos in r['unit_slots']:
            ok=gold.get(uid)==pred.get(uid);scores[('worker' if k==0 else 'base')+'_hit']+=ok
            if k==0 and p[2][uid][1]>=50:
                scores['loaded_n']+=1;scores['loaded_hit']+=ok
        details.append(dict(input=r['hash'],teacher_order=gold,prediction=pred,terrain=r['terrain'],step=r['step'],seat=r['me'],depth=r['depth'],branch=r['branch']))
    s=sum(bymap.values(),Counter())
    kl={k:float(np.mean([v[k+'_kl_sum']/v[k+'_n'] for v in bymap.values()])) for k in ['worker','base']}
    return dict(method=method,sum=s,physical_maps=len(bymap),map_equal_KL=kl,by_map=bymap),details


def main():
    cli=argparse.ArgumentParser();cli.add_argument('split',choices=('train','val'));cli.add_argument('--cache',type=Path);cli.add_argument('--weights-sha');args=cli.parse_args()
    torch.set_num_threads(1);m=module(original());m._net();file=HERE/'student.npz'
    if args.split=='val':assert sha(file.read_bytes())==args.weights_sha
    ar=pack(args.split,args.cache) if args.cache else HERE/(args.split+'_boards.zip')
    tensors,rows,seen,cards,prep=extract(m,args.split,ar)
    x,initial,y,role,old=tensors
    out=dict(source=P['source'],archive_sha=sha(ar.read_bytes()),cards=cards,rows=len(rows),units=len(x),decode_seconds=prep,
             max_teacher_logit_error=max(r['numerical_check'] for r in rows),sample_keys=sorted(seen),
             capture_seconds=sum(r['elapsed'] for r in cards),search_seconds=sum(c['wall'] or 0 for r in cards for c in r['calls']))
    if args.split=='train':
        begin=time.perf_counter();head,log=fit(tensors);out['fit_seconds']=time.perf_counter()-begin;out['losses']=log
        np.savez_compressed(file,**{k:v.numpy() for k,v in head.state_dict().items()})
        print('frozen',sha(file.read_bytes()),flush=True)
    else:
        train=json.loads((HERE/'train.json').read_text())
        assert not seen.intersection(train['sample_keys'])
        assert not set(c['terrain'] for c in cards)&set(c['terrain'] for c in train['cards'])
        head=Readout()
        with np.load(file) as f:head.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files})
    with torch.inference_mode():student=head(x,initial)
    out['weights_sha256']=sha(file.read_bytes());out['trainable']=sum(v.numel() for v in head.parameters());out['methods']={}
    for method,v in [('block3_direct',initial),('block2_trained',old),('block3_spatial',student)]:
        scores,detail=report(m,rows,y,v,method);out['methods'][method]=scores
        if args.split=='val' and method=='block3_spatial':write(HERE/'held_orders.json.gz',detail)
    write(HERE/(args.split+'.json'),out)
    print(args.split,len(rows),out['methods']['block3_spatial']['sum'],out['methods']['block3_spatial']['map_equal_KL'],flush=True)


if __name__=='__main__':main()
