"""Apply one preregistered gradient step and audit the two exact deployed policies."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import alignment as a
from original_data import original_roots
import numpy as np
import torch
o=a.o
L=a.L
PROTOCOL='69254a9c3ab0068bbd465315a5c8959990a5e236f6200aaa7f1f6a11c6a12033'


def verify(revision):
    p=HERE/'protocol.json'; rel=p.relative_to(o.ROOT).as_posix()
    require=a.require
    require(o.sha(p)==PROTOCOL and len(revision)==40 and p.read_bytes()==subprocess.check_output(
        ['git','-C',str(o.ROOT),'show',f'{revision}:{rel}']),'registered Git file missing or changed')
    plan=o.read(p)
    paths={'league_fit_sha256':L/'fit.json','historical_alignment_sha256':HERE.parent/'alignment.json',
           'league_training_register_sha256':L/'collection_registration.json','league_collection_sha256':L/'collection.json',
           'gradient_implementation_sha256':HERE.parent/'alignment.py','original_root_loader_sha256':HERE.parent/'original_data.py',
           'historical_8_protocol_sha256':o.EXP/'onpolicy_native/protocol.json','historical_8_scores_sha256':o.EXP/'onpolicy_native/scores.json'}
    require(all(o.sha(path)==plan[k] for k,path in paths.items()),'training source drift')
    require(o.sha(o.PACK)==plan['parent_sha256'] and plan['nominal_pre_export_l2']==.10,'step or parent drift')
    return plan


def league_roots(model):
    reg=o.read(L/'collection_registration.json');sha=o.sha(L/'collection_registration.json')
    manifest=o.read(L/'collection.json')
    teacher=o.teacher_reader(model)
    roots=[]
    for key,arms in sorted(o.archive_game_cards(manifest).items()):
        card,base=arms['sampled'],arms['greedy']
        roots.extend(o.assemble(card,model,teacher,reg['sample_steps'],card['u']-base['u'],reg,sha))
    return roots


def norm(values):
    return math.sqrt(sum(np.vdot(v,v).item() for v in values.values()))


def diagnostic(roots,data,rows,child,denom):
    checked={};teacher=o.teacher_reader(child);receipts=[]
    model=o.torch_model(child,1)
    with torch.inference_mode():
        approx=np.concatenate([o.evaluate(model,data,ix).numpy().astype(float) for ix in o.groups(len(rows))])
    for i,(_,r,public) in enumerate(roots):
        x,aux,uids=public
        feat=sys.modules['cnet2_feat'].Feat(r['seat']);feat.aux,feat.own_uid=aux,uids
        encoded,labels,child_route=o.decode_root(child,feat,x)
        a.require(np.array_equal(labels,child._mixed_actions(x,aux)),'repacked free policy inconsistent')
        cmd=feat.to_actions(labels)
        valid=o.actual_features(feat,x,cmd,child)
        a.require(valid['cut']==len(labels) and feat.to_actions(feat.labels(cmd))==cmd,'free command failed physical legality')
        fixed=np.zeros_like(data['residual'][i]);mix=np.zeros_like(fixed)
        teacher(x,aux,encoded,r['route'],feat.labels(r['parent_order']),fixed,mix,r['slot']+1)
        scores=mix[r['slot'],r['moves']].astype(float)
        gap=float(scores[0]-scores[1])
        a.require(abs(gap-approx[i])<=o.LOGIT_TOL,'quantized Torch and NumPy readback disagree')
        g0=r['observed_gap']; sign=1 if r['pick']==0 else -1
        l0=lambda g: -np.logaddexp(0.,-g)
        sign0=float(l0(sign*g0));sign1=float(l0(sign*gap))
        a.require(abs(sign0-r['old_logp'])<1e-10,'old log-probability drift')
        p=r['observed_p0'];kl=float(p*(l0(g0)-l0(gap))+(1-p)*(l0(-g0)-l0(-gap)))
        a.require(kl>=-1e-14 and math.isfinite(kl),'nonfinite binary KL')
        receipts.append(dict(seed=r['seed'],seat=r['seat'],step=r['step'],slot=r['slot'],moves=r['moves'],pick=r['pick'],
                             advantage=r['advantage'],parent_order=r['parent_order'],free_child=cmd,old_route=r['route'],
                             new_free_route=child_route,changed_free=cmd!=r['parent_order'],old_gap=g0,projected_gap=gap,
                             old_logp=sign0,projected_logp=sign1,weighted_logp_delta=(sign1-sign0)*r['advantage'],
                             kl_parent_to_child=max(kl,0.),torch_gap=approx[i]))
    changed=[sum(r['seat']==s and r['changed_free'] for r in receipts) for s in (0,1)]
    total_kl=sum(r['kl_parent_to_child'] for r in receipts)
    weighted=sum(r['weighted_logp_delta'] for r in receipts)/denom
    return dict(scheduled=denom,supported=len(receipts),changed_free_by_seat=changed,
                weighted_logp_delta_per_scheduled=weighted,mean_binary_kl_supported=total_kl/len(receipts),
                mean_binary_kl_scheduled=total_kl/denom,
                max_torch_native_error=max(abs(r['torch_gap']-r['projected_gap']) for r in receipts),
                qualifies=bool(weighted>0 and min(changed)>0)),receipts


def fit(revision):
    plan=verify(revision);base,net=a.parameters()
    original,source=original_roots()
    corpora={'original':original,'league':league_roots(base)}
    normal={'original':64,'league':432}
    data={};rows={};grad={};parity={}
    for name in corpora:
        roots=corpora[name]
        data[name],rows[name]=a.network_data(roots)
        _,_,parity[name]=o.errors(net,data[name],rows[name])
        a.require(parity[name]['max_probability_error']<=o.PROB_TOL and parity[name]['max_signed_logit_difference_error']<=o.LOGIT_TOL,'old policy parity')
        grad[name]=a.gradient(net,data[name],rows[name],np.arange(len(roots)),normal[name])
    measured={k:norm(g) for k,g in grad.items()}
    expected=o.read(HERE.parent/'alignment.json')['comparison']['total']
    a.require(all(abs(measured[k]-expected[k+'_norm'])<1e-8 for k in measured),'original gradient not reproduced')
    direction={k:grad['original'][k]/measured['original']+grad['league'][k]/measured['league'] for k in grad['league']}
    size=norm(direction);a.require(size>0 and math.isfinite(size),'unit directions canceled')
    old={k:p.detach().numpy().copy() for k,p in net.named_parameters() if k in direction}
    with torch.no_grad():
        for k,p in net.named_parameters():
            if k in direction: p.add_(torch.from_numpy((plan['nominal_pre_export_l2']/size)*direction[k]).to(p.dtype))
    real={k:p.detach().numpy().astype(float)-old[k].astype(float) for k,p in net.named_parameters() if k in direction}
    a.require(abs(norm(real)-.10)<1e-6,'floating parameter step outside roundoff')
    with np.load(o.BASE,allow_pickle=False) as ref:
        a.require(o.repack.quantize(o.BASE,ref)==o.repack.literal(o.PACK.read_text(),o.repack.STUDENT),
                  'parent editable weights do not reproduce deployment')
    released=o.export(net,base,'candidate',HERE)
    a.require(released['quantized_changed']>0 and released['package_bytes']<=10_000_000,'invalid package')
    child=o.package(HERE/'candidate.py');cn=o.torch_model(child,1)
    quantized={k:p.detach().numpy().astype(float)-old[k].astype(float) for k,p in cn.named_parameters() if k in old}
    reports={};decisions={}
    for name in corpora:
        reports[name],decisions[name]=diagnostic(corpora[name],data[name],rows[name],child,normal[name])
        reports[name]['first_order_continuous_delta']=sum(np.vdot(grad[name][k],real[k]).item() for k in real)
        reports[name]['first_order_quantized_delta']=sum(np.vdot(grad[name][k],quantized[k]).item() for k in real)
    decision_file=HERE/'decision_audit.json.gz';o.save(decision_file,decisions)
    result=dict(protocol_sha256=PROTOCOL,registered_git_revision=revision,fit_script_sha256=o.sha(Path(__file__)),
                parent_sha256=plan['parent_sha256'],candidate_sha256=o.sha(HERE/'candidate.py'),weights_sha256=o.sha(HERE/'candidate.npz'),
                gradient_norms=measured,sum_of_unit_gradients_norm=size,nominal_l2=.10,actual_floating_parameter_l2=norm(real),
                deployed_parameter_l2=norm(quantized),parent_behavior_parity=parity,decision_audit_sha256=o.sha(decision_file),
                export=released,cohorts=reports,passes_training_mechanism=all(v['qualifies'] for v in reports.values()),
                source_original_protocol=source['source_protocol_sha256'] if 'source_protocol_sha256' in source else plan['historical_8_protocol_sha256'])
    o.save(HERE/'fit.json',result)
    print(result['candidate_sha256'],result['passes_training_mechanism'],reports,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registration-rev',required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    fit(args.registration_rev)
