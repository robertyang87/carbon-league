"""Compare the same full policy score gradient on two previously completed game collections."""
import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
L=HERE.parent
sys.path.insert(0,str(L))
import collect as bound_collector
import optimize as o
import numpy as np
import torch

SOURCE_FIT='51648ab50f0447a282bc46e3b850eb2a47ef240e7221908f1aab13d1bdd04041'
CHILD='510c527a6f3e88f718e71e4b456e6e065abaeb1474587dc76fd11a5744d7c3d4'


def require(ok, message):
    if not ok:raise ValueError(message)


def parameters():
    model=o.package()
    net=o.torch_model(model,1)
    for name,parameter in net.named_parameters():
        parameter.requires_grad_(not name.startswith(o.VALUE))
    return model,net


def network_data(roots):
    features,rows,_=zip(*roots)
    data={k:np.stack([r[k] for r in features]) for k in features[0]}
    data.update(slot=np.array([r['slot'] for r in rows]),moves=np.array([r['moves'] for r in rows]))
    return data,rows


def gradient(net,data,rows,ids,denominator):
    """Ascent gradient of A*log p; zeros for parameters unused by signed conditional policy scores."""
    net.zero_grad(set_to_none=True)
    for ix in o.groups(len(ids)):
        part=ids[ix]
        gap=o.evaluate(net,data,part).double()
        native=torch.tensor([rows[i]['observed_gap'] for i in part],dtype=torch.float64)
        x=native+(gap-gap.detach())
        side=torch.tensor([1. if rows[i]['pick']==0 else -1. for i in part],dtype=torch.float64)
        a=torch.tensor([rows[i]['advantage'] for i in part],dtype=torch.float64)
        value=(a*torch.nn.functional.logsigmoid(side*x)).sum()/denominator
        value.backward()
    return {n:(p.grad.detach().numpy().astype(float) if p.grad is not None else np.zeros(tuple(p.shape)))
            for n,p in net.named_parameters() if p.requires_grad}


def make_reference(cache):
    require(o.sha(L/'fit.json')==SOURCE_FIT,'wrong original training update')
    fit=o.read(L/'fit.json')
    manifest=o.read(L/'collection.json');register=o.read(L/'collection_registration.json')
    require(fit['collection_sha256']==o.sha(L/'collection.json') and manifest['registration_sha256']==o.sha(L/'collection_registration.json'),'league data chain')
    model,net=parameters();teacher=o.teacher_reader(model);roots=[]
    pairs=o.archive_game_cards(manifest)
    for key,arms in sorted(pairs.items()):
        card,greedy=arms['sampled'],arms['greedy']
        roots.extend(o.assemble(card,model,teacher,register['sample_steps'],card['u']-greedy['u'],
                                register,manifest['registration_sha256']))
    data,rows=network_data(roots);require(len(rows)==fit['decisions'],'missing league draws')
    _,_,err=o.errors(net,data,rows)
    require(err['max_signed_logit_difference_error']<=o.LOGIT_TOL and err['max_probability_error']<=o.PROB_TOL,'league parity')
    result=gradient(net,data,rows,np.arange(len(rows)),36*12)
    gnorm=math.sqrt(sum(np.vdot(v,v).item() for v in result.values()))
    bias_error=float(np.max(np.abs(result['whead.bias']+np.array(fit['training']['unclipped_worker_bias_gradient']))))
    require(abs(gnorm-fit['training']['norm_before_clip'])<1e-7 and bias_error<1e-9,'reference gradient mismatch')
    for group,v in fit['training']['gradient_norm_by_module'].items():
        n=math.sqrt(sum(np.vdot(a,a).item() for k,a in result.items() if k.split('.')[0]==group))
        require(abs(n-v)<1e-7,f'reference module differs: {group}')
    np.savez(cache,**result)
    return dict(reference_gradient_norm=gnorm, reference_worker_bias_max_error=bias_error,
                reference_supported=len(rows),reference_episodes=36,normalization=432,fit_sha256=SOURCE_FIT,
                cached_asc_grad_arrays_sha256=o.sha(cache))


def compare(g,h):
    def inner(keys):
        dot=sum(np.vdot(g[k],h[k]).item() for k in keys)
        a=math.sqrt(sum(np.vdot(g[k],g[k]).item() for k in keys))
        b=math.sqrt(sum(np.vdot(h[k],h[k]).item() for k in keys))
        cos=float(np.clip(dot/(a*b),-1,1)) if a and b else None
        return dict(dot=dot,cosine=cos,angle_degrees=math.degrees(math.acos(cos)) if cos is not None else None,
                    league_norm=a,original_norm=b)
    require(g.keys()==h.keys() and all(g[k].shape==h[k].shape for k in g),'parameter mismatch')
    names=sorted(g);groups=sorted({n.split('.')[0] for n in names})
    return dict(total=inner(names),by_module={k:inner([n for n in names if n.split('.')[0]==k]) for k in groups})


def audit_original(cache,reference):
    from original_data import original_roots
    roots,receipt=original_roots();_,base=parameters();trained=o.package(L/'candidate.py');child=o.torch_model(trained,1)
    require(o.sha(L/'candidate.py')==CHILD,'wrong deployed update')
    teacher=o.teacher_reader(trained)
    data,rows=network_data(roots)
    _,_,err=o.errors(base,data,rows)
    require(err['max_signed_logit_difference_error']<=o.LOGIT_TOL and err['max_probability_error']<=o.PROB_TOL,'native training parity on original')
    with torch.inference_mode():
        torch_child=np.concatenate([o.evaluate(child,data,ids).numpy().astype(float) for ids in o.groups(len(rows))])
    # Recompute the deployed child projection with actual frozen partner/safety mask and the logged pre-action prefix.
    log_tickets=defaultdict(list);readback=[]
    for i,(_,r,public) in enumerate(roots):
        x,aux,uids=public
        feat=sys.modules['cnet2_feat'].Feat(r['seat']);feat.aux,feat.own_uid=aux,uids
        enc,_,_=o.decode_root(trained,feat,x)
        prefix=feat.labels(r['parent_order'])
        fixed=np.zeros_like(data['residual'][i]);combined=np.zeros_like(fixed)
        teacher(x,aux,enc,r['route'],prefix,fixed,combined,r['slot']+1)
        scores=combined[r['slot'],r['moves']].astype(float)
        child_gap=float(scores[0]-scores[1]);require(abs(child_gap-torch_child[i])<o.LOGIT_TOL,'deployed child gap reconstruction failed')
        sign=1 if r['pick']==0 else -1
        log0=-float(np.logaddexp(0,-sign*r['observed_gap']));log1=-float(np.logaddexp(0,-sign*child_gap))
        require(abs(log0-r['old_logp'])<1e-10,'signed log probability disagreement')
        value=r['advantage']*(log1-log0)
        log_tickets[r['seed'],r['seat']].append(value)
        readback.append(dict(seed=r['seed'],seat=r['seat'],step=r['step'],advantage=r['advantage'],old_logp=log0,
                             new_projected_logp=log1,weighted_delta=value))
    seeds=sorted({r['seed'] for r in rows});require(len(seeds)==4 and len(log_tickets)==8,'missing retired maps')
    base_g=gradient(base,data,rows,np.arange(len(rows)),64)
    with np.load(cache,allow_pickle=False) as z:league={k:z[k] for k in z.files}
    total=compare(league,base_g)
    parent_params=dict(base.named_parameters());child_params=dict(child.named_parameters())
    delta={k:(child_params[k].detach().cpu().numpy().astype(float)-parent_params[k].detach().cpu().numpy().astype(float)) for k in base_g}
    # Compare the actually deployed quantized parameter displacement, in addition to the raw score gradient.
    delta_by_module={k:sum(np.vdot(base_g[n],delta[n]).item() for n in base_g if n.split('.')[0]==k)
                     for k in sorted({n.split('.')[0] for n in base_g})}
    loo=[];by_map=[]
    for seed in seeds:
        for leave,out in ((True,loo),(False,by_map)):
            indices=np.array([i for i,r in enumerate(rows) if (r['seed']!=seed)==leave])
            n=48 if leave else 16
            calculated=gradient(base,data,rows,indices,n)
            out.append(dict(seed=seed,denominator=n,supported=len(indices),comparison=compare(league,calculated),actual_update_first_order_dot=sum(np.vdot(calculated[k],delta[k]).item() for k in delta),
                            weighted_logp_delta=sum(v for (s,_),values in log_tickets.items() if (s!=seed)==leave for v in values)/n))
    require(o.sha(L/'fit.json')==SOURCE_FIT and reference['cached_asc_grad_arrays_sha256']==o.sha(cache),'broken reference cache')
    return dict(operator='full policy ascent before clipping or Adam, same mixed-logit softmax on recorded two-order support',
                league_reference=reference,source_loader_sha256=o.sha(HERE/'original_data.py'),analysis_sha256=o.sha(Path(__file__)),
                original_inputs=receipt,original_parity=err,comparison=total,leave_one_original_map_out=loo,individual_maps=by_map,
                gradient_dims=sum(v.size for v in base_g.values()),policy_update_sha256=CHILD,
                actual_deployed_delta_l2=math.sqrt(sum(np.vdot(v,v).item() for v in delta.values())),
                original_gradient_dot_actual_deployed_delta=sum(delta_by_module.values()),
                original_gradient_dot_actual_deployed_delta_by_module=delta_by_module,
                weighted_logp_delta_per_offered_turn=sum(v for values in log_tickets.values() for v in values)/64,
                weighted_logp_delta_by_ticket=[dict(seed=s,seat=p,sum=sum(values),per_offered_turn=sum(values)/8)
                                              for (s,p),values in sorted(log_tickets.items())],
                decisions=readback,
                limitation='This compares noisy offline gradient estimates, not strength. Original advantages use matched historical maps/seats and different natural examiner samples and wall clocks. League and examiner have different scheduled decision times. Candidate likelihood uses the recorded two joint orders, original route and original prefix, not a newly selected free menu. Unsampled legal decisions have no fitted term. No recent natural-original outcomes or held maps were read.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['reference','finish'])
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    descriptor=args.cache.with_suffix('.json')
    if args.mode=='reference':
        o.save(descriptor,make_reference(args.cache));print('league reference verified',flush=True)
    else:
        report=audit_original(args.cache,o.read(descriptor));o.save(HERE/'alignment.json',report)
        print(report['comparison']['total'],report['weighted_logp_delta_per_offered_turn'],flush=True)
