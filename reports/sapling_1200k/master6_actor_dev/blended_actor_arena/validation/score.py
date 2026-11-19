"""Compare same-logged-prefix mixture CE using the *deployed* quantized bytes in each pack.
The second implementation manually advances the NumPy GRU on true selected labels for every slot.
"""
import argparse
import base64
import json
import sys
import zlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared import ACTOR,CACHE,HERE,PROTO,ROOT,TRAIN,module,read,require,save,sha
sys.path.insert(0,str(ACTOR))
import train as bc

def torch_net(pack,key):
    blob,q,meta=(pack._B1,pack._Q1,pack._M1) if key=='expert' else (pack._W64,pack._QMETA,pack._WMETA)
    buf=pack._dequantize(zlib.decompress(base64.b64decode(blob)),q)
    topo=json.loads(meta)['cfg'];net=bc.CNet2(**topo)
    missed,extra=net.load_state_dict({k:torch.from_numpy(v) for k,v in buf.items()},strict=False)
    require(not extra and set(missed)<=set([p+'.'+k for p in ('utility','margin','wdl') for k in ('weight','bias')]),'unexpected packed weights')
    w=dict(buf,meta=np.array(meta))
    return net.eval(),sys.modules['cnet2_np'].Policy(w)

def np_scores(policy,d,idx):
    obs={k:(v[idx].astype('f') if v.dtype==np.float16 else v[idx]) for k,v in d.items()}
    tokens,ctx=policy.encode(obs,2);hidden=policy.lin(ctx,'h0');prev=8
    result={}
    for k,truth in enumerate(obs['y']):
        if truth<0:continue
        mf=obs['mf'][k];uf=obs['uf'][k];state=np.concatenate([tokens[k],ctx,uf,mf.reshape(-1)])
        hidden=policy.gru(policy.lin(state,'din')+policy.w['aemb.weight'][prev],hidden)
        role=policy.lin(hidden,'whead') if k<10 else np.concatenate([policy.lin(hidden,'bhead'),np.zeros(2,np.float32)])
        result[k]=role+policy.lin(mf,'V')@policy.lin(hidden,'U')
        require(obs['legal'][k,int(truth)],'illegal demonstrated action')
        prev=int(truth)+(0 if k<10 else 5)
    return result

@torch.inference_mode()
def run(arm):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=read(PROTO)
    pack_path=ROOT/protocol['arm_parent']['path'] if arm=='parent' else TRAIN/'serving'/(arm+'.py')
    if arm=='parent':require(sha(pack_path)==protocol['arm_parent']['sha256'],'parent')
    else:
        sign=read(TRAIN/'serving/packages.json')['packages'][arm]
        require(sha(pack_path)==sign['deployed_sha256'] and sha(TRAIN/(arm+'.npz'))==sign['npz_sha256'],'candidate packaging')
    split_path=ACTOR/'data_split.json';require(sha(split_path)==protocol['admission']['split_sha256'],'holdout registry')
    manifest=read(split_path)['games'];shots=read(ACTOR/'data_receipts.json.gz')['games'];sh={r['id']:r['sha256'] for r in shots}
    val=[g for g in manifest if g['split']=='val'];require(len(val)>0 and all((CACHE/(g['id']+'.npz')).is_file() for g in val),'original val shard unavailable')
    if arm!='parent':
        recorded=read(TRAIN/'panel.json')['games']
        fit=[g for g in recorded if g['split']=='train']
        require(fit and all(not({r[k] for r in val}&{r[k] for r in fit}) for k in ('id','seed','terrain_sha256')),'training overlap')
    x=module(pack_path,'blend_ce_'+arm)
    require((x._HYBRID['expert_weight'],x._HYBRID['student_weight'],x._HYBRID['style'])==(.75,.25,2),'mixture')
    expert,nen=torch_net(x,'expert');student,nsn=torch_net(x,'student')
    maps=[];par=[];examples=[];worst=0.;changing=0;active=0
    for row in val:
        d,hashes=bc.public('val',[row],CACHE,sh);b=bc.batch(d,slice(None))
        ez=expert(b)[0];sz=student(b)[0]
        z=(.75*ez+.25*sz).masked_fill(~b['legal'].bool(),-1e9)
        y=b['y'];select=(y>=0)&(b['legal'].sum(-1)>=2)
        scores=F.cross_entropy(z[select],y[select],reduction='none').numpy().astype(float)
        require(len(scores)>0 and np.all(np.isfinite(scores)),'CE nonfinite')
        maps.append({'id':row['id'],'terrain':row['terrain_sha256'],'gid':int(row['gid']),'shard_sha256':sh[row['id']],'roots':len(d['y']),'slots':len(scores),'CE':float(scores.mean()),'correct':int((z[select].argmax(-1)==y[select]).sum())})
        for i in sorted(set([0,len(y)//2,len(y)-1])):
            ze=np_scores(nen,d,i);zs=np_scores(nsn,d,i)
            for k in ze:
                legal=d['legal'][i,k];P=.75*ze[k]+.25*zs[k];T=z[i,k].numpy();a=int(np.argmax(np.where(legal,P,-np.inf)));t=int(np.argmax(T))
                err=float(np.max(np.abs(P[legal]-T[legal])));worst=max(worst,err);changing+=int(a!=t);active+=1
                if a!=t:examples.append([row['id'],int(d['step'][i]),k,a,t,err])
    require(active>0 and worst<.002 and changing==0,'Torch vs physical-prefix numpy mismatch '+str((worst,changing,examples[:2])))
    output={'arm':arm,'pack_sha256':sha(pack_path),'source_split_sha256':sha(split_path),'protocol_sha256':sha(PROTO),'weights_sha256':protocol['arm_parent']['frozen_student'] if arm=='parent' else sha(TRAIN/(arm+'.npz')),
            'equal_map_ce':float(np.mean([r['CE'] for r in maps])),'maps':maps,'slots':sum(r['slots'] for r in maps),'checked_greedy_argmax_mismatches':changing,'parity_checked_active_slots':active,'max_abs_legal_logit_error':worst}
    save(HERE/'validation'/(arm+'_ce.json'),output);print(arm,output['equal_map_ce'],output['slots'],worst,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('parent','candidate','control'));run(p.parse_args().arm)
