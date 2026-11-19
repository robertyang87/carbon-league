"""How much NLL can the unchanged parent recover without altering a single greedy decision?"""
import json
import os
import numpy as np
import torch
import fit

HERE=fit.HERE
DESIGN={'scope':'Post-training explanation only. Never substitute an actor or touch arena eligibility.',
        'training_root_pick':'For all 218 training maps take four rows without replacement using RNG 4403879; calibrate by equally weighted maps.',
        'seed':4403879,'roots_per_train_map':4,'temperature_bounds':[0.25,4.0],
        'sample_splits':['train','val','holdout'],'grouping':'Aggregate legal multi-choice demonstration slots within map, then average maps.',
        'evaluation':'Compare temperature-fitted original ensemble CE against published original/candidate; retain original raw greedy and return correction unchanged.'}


@torch.no_grad()
def source(games,data,parent,expert,indices):
    gids=[]; ys=[]; zs=[]
    for i in range(0,len(indices),16):
        ix=indices[i:i+16]
        b=fit.common.batch(data,ix)
        keep=fit.common.actionable(b)
        z=fit.blended(expert(b)[0],parent(b)[0])
        gids.append(np.broadcast_to(data['gid'][ix,None],keep.shape)[keep.numpy()])
        zs.append(z[keep].numpy())
        ys.append(b['y'][keep].numpy())
    gid=np.concatenate(gids);z=np.concatenate(zs).astype('float64');y=np.concatenate(ys)
    labels,inv,n=np.unique(gid,return_inverse=True,return_counts=True)
    balance=1./n[inv]/len(labels)
    assert len(labels)==len(games) and np.isclose(balance.sum(),1)
    return z,y,balance,labels,inv


def run():
    if os.sched_getaffinity(0)!={2} or os.getpriority(os.PRIO_PROCESS,0)<19:
        raise ValueError('use one low-priority CPU2')
    torch.set_num_threads(1)
    panel=json.loads((HERE/'panel.json').read_bytes())['games']
    original=fit.common.model(fit.PARENT,fit.PROTOCOL['parent_weights_sha256'])[0]
    parent=fit.common.quantized(original).eval(); expert=fit.expert_model()
    rng=np.random.default_rng(DESIGN['seed'])
    temperature=None; report={'design':DESIGN,'fit_sha256':fit.sha(HERE/'fit.json'),'script_sha256':fit.sha(__file_path()),'sets':{}}
    published=json.loads((HERE/'fit.json').read_bytes())
    for split in DESIGN['sample_splits']:
        games=[g for g in panel if g['split']==split]
        data=fit.load(games)
        if split=='train':
            ix=np.concatenate([rng.choice(np.where(data['gid']==g['gid'])[0],DESIGN['roots_per_train_map'],replace=False) for g in games])
        else:
            ix=np.arange(len(data['gid']))
        z,y,w,labels,inv=source(games,data,parent,expert,ix)
        margin=z-z.max(1,keepdims=True)
        chosen=margin[np.arange(len(y)),y]
        def ce(temp):
            nll=np.log(np.exp(margin/temp).sum(1))-chosen/temp
            return nll, np.dot(nll,w)
        if split=='train':
            lo,hi=np.log(DESIGN['temperature_bounds'])
            for _ in range(64):
                a,b=(2*lo+hi)/3,(lo+2*hi)/3
                if ce(np.exp(a))[1]<ce(np.exp(b))[1]:
                    hi=b
                else:
                    lo=a
            temperature=float(np.exp((lo+hi)/2))
        orig,v0=ce(1.0); new,v1=ce(temperature)
        corrected = ((z/temperature).argmax(1)!=z.argmax(1)).sum()
        if corrected:
            raise ValueError('temperature altered greedy class')
        per={int(g):{'original':float(orig[inv==j].mean()),'temperature_only':float(new[inv==j].mean())} for j,g in enumerate(labels)}
        result={'maps':len(labels),'slots':len(y),'roots':len(ix),'fitted_temperature':temperature,'original':float(v0),'temperature_only':float(v1),
                'changed_base_slot_argmax':int(corrected),'by_map':{g['id']:per[g['gid']] for g in games}}
        if split!='train':
            base=published['scores'][split]['all']['ce']
            if not np.isclose(v0,base['parent'],atol=2e-6,rtol=0):
                raise ValueError('parent reconstruction CE mismatch')
            result['trained_candidate']=base['candidate']
            result['candidate_minus_temperature_only']=float(base['candidate']-v1)
        report['sets'][split]=result
        print(split,temperature,float(v0),float(v1),flush=True)
    fit.common.write_json(HERE/'calibration.json',report)


def __file_path():
    from pathlib import Path
    return Path(__file__)

if __name__=='__main__':
    run()
