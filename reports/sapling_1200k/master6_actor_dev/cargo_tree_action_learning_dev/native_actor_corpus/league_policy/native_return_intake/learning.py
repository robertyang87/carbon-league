"""Fit one full-policy score direction from 12 pairs of independent 851c games against the original."""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
HERE=Path(__file__).resolve().parent
LEAGUE=HERE.parent
sys.path[:0]=[str(LEAGUE/'native_gradient'),str(LEAGUE/'natural_original')]
import alignment as a
import audit_menu as menu
import numpy as np
import torch
o=a.o
c=a.bound_collector
BEHAVIOR=LEAGUE/'natural_original/candidate_sampled.py'
BASE_NPZ=LEAGUE/'candidate.npz'
PLAN_SHA='d348f38843eb899886b6643527dc57ba40d8c551ce8e2afd7f2640f63c56e75a'
check=a.require


def signed_plan(revision):
    path=HERE/'learning.json'
    check(len(revision)==40 and o.sha(path)==PLAN_SHA and path.read_bytes()==subprocess.check_output(
        ['git','-C',str(o.ROOT),'show',revision+':'+path.relative_to(o.ROOT).as_posix()]),'unregistered learner')
    plan=o.read(path)
    for f,k in [(BEHAVIOR,'deployed_behavior'),(LEAGUE/'candidate.py','dequantized_backbone_source_pack'),
                (BASE_NPZ,'editable_source_npz'),(LEAGUE/'natural_original/export_behavior.py','base_behavior_export_sha256'),
                (LEAGUE/'natural_original/audit_menu.py','base_menu_auditor_sha256')]:
        check(o.sha(f)==plan[k],'source identity '+k)
    return plan


def compact(doc):return json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def build_features(base,teacher,feat,x,event,adv,seed,seat):
    slot=event['slot']; moves=event['actions']; opts=event['options']; pick=event['choice']
    check(len(moves)==len(set(moves))==len(opts)==2 and opts[0]!=opts[1] and opts[pick]==event['issued'],'recorded full order')
    enc,labels,route=o.decode_root(base,feat,x)
    check(route==event['route'] and feat.to_actions(labels)==opts[0] and labels[:slot].tolist()==event['prefix'],'parent route and prefix')
    for move,cmd in zip(moves,opts):
        y=feat.labels(cmd)
        check(int(y[slot])==move and y[:slot].tolist()==event['prefix'] and feat.to_actions(y)==cmd,'physical menu mismatch')
    row=o.actual_features(feat,x,opts[0],base)
    check(row.pop('cut')>slot and row['eligible'][slot] and row['legal'][slot].tolist()==event['legal']
          and all(row['legal'][slot,moves]),'real prefix legality')
    fixed=np.zeros_like(row['residual']);mix=np.zeros_like(fixed)
    teacher(x,feat.aux,enc,route,labels,fixed,mix,slot+1);row['residual']=fixed
    gap=float(np.subtract(*np.asarray(event['native_logits'],float)))
    native=float(np.subtract(*mix[slot,moves].astype(float)))
    p=float(event['probabilities'][0]); observed=math.log(event['probabilities'][pick])
    check(abs(native-gap)<=o.LOGIT_TOL and abs(p-1/(1+math.exp(-gap)))<=1e-12 and
          abs(observed-event['logp'])<=1e-12,'acted mixed native probability')
    meta=dict(seed=seed,seat=seat,step=event['step'],slot=slot,moves=moves,pick=pick,advantage=adv,
              parent_order=opts[0],observed_gap=gap,native_gap=native,observed_p0=p,old_logp=observed,route=route)
    return row,meta


def replay_episode(ep,model,teacher,plan):
    seat,seed=ep['seat'],ep['seed']; path=o.ROOT/ep['archive_path']
    check(o.sha(path)==ep['archive_sha256'],'training archive changed')
    schedule=set(plan['scheduled_turns']);events={e['step']:e for e in ep['events']}
    check(len(events)==len(ep['events']),'duplicate opportunities')
    roots=[];sites=[]
    with zipfile.ZipFile(path) as z, TemporaryDirectory() as temp:
        reg=Path(temp)/'registration.json';reg.write_bytes(z.read(ep['registration_member']))
        receipt=json.loads(z.read(ep['receipt_member']));tape=z.read(ep['turns_member'])
        check(sha_bytes(tape)==receipt['turns_sha256'] and receipt.get('official_scores_by_seat')==ep['official_scores_by_seat'],'unsigned game tape')
        live=menu.open_live(reg); env,cfg=live.official()
        check(sha_bytes(bytes(env.states[0].observation.full_carbon))==ep['terrain'],'opening terrain')
        actor=c.Player(BEHAVIOR);F=sys.modules['cnet2_feat'];feat=F.Feat(seat);bank=[5.,5.]
        frames=[json.loads(s) for s in tape.splitlines()]
        for t,frame in enumerate(frames):
            check(not env.done and t==frame['step']==env.states[0].observation.step and all(q['status']=='ok' for q in frame['calls']),'training completed actions missing')
            pub=live.observe(env,seat,bank[seat]);h=sha_bytes(live.encoded(pub))
            check(h==frame['public_observation_sha256'][seat] and 'full_carbon' not in pub,'public observation drift')
            cmds=[q['response']['action'] for q in frame['calls']]
            actual=menu.per_order(actor,'candidate',pub,cfg,cmds[seat])
            x=feat.observe(pub)
            if t in schedule:
                e=events[t]
                check(e['seat']==seat and pub==e['public_observation'],'physical public state or seat changed')
                if e['supported']:
                    role=pub['players'][seat][2][e['uid']][2]
                    check(str(feat.own_uid[e['slot']])==e['uid'] and e['role']==int(role=='PLANTER'),'source worker role or uid differs')
                check(actual is not None and all(k in e and e[k]==v for k,v in actual.items()) and
                      e['public_observation_sha256']==h,'independent natural menu versus emitted event')
                public=(copy.deepcopy(x),copy.deepcopy(feat.aux),list(feat.own_uid))
                if actual['supported']:
                    f,r=build_features(model,teacher,feat,x,actual,ep['advantage'],seed,seat)
                    roots.append((f,r,public))
                sites.append(dict(seed=seed,seat=seat,arm=ep['arm'],step=t,event=actual,public=public))
            else:check(actual is None,'unregistered binary choice')
            bank=list(frame['overage_remaining_seconds']);env.step(cmds)
        rewards=[state.reward for state in env.states]
        check(env.done and [v.status for v in env.states]==['DONE','DONE'] and rewards==ep['official_scores_by_seat']
              and set(events)=={t for t in schedule if t<len(frames)},'incorrect episode or event census')
        _,u=o.payoff(rewards,seat)
        check(abs(u-ep['u'])<1e-12,'wrong terminal utility')
    return roots,sites


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def training_manifest(plan,path):
    m=o.read(path)
    check(m['training_eligible'] and m['learning_sha256']==PLAN_SHA and m['normalization']==288 and m['policy_sha256']==plan['deployed_behavior']
          and m['settlement_sha256']==o.sha(path.parent/'settlement.json') and
          m['protocol_sha256']==o.sha(HERE/'protocol.json'),'unsigned training intake')
    eps=m['episodes']; groups=defaultdict(dict)
    for ep in eps:
        pair=groups[ep['seed'],ep['seat']]
        check(ep['arm'] in ('draw_a','draw_b') and ep['arm'] not in pair,'duplicate training arm')
        pair[ep['arm']]=ep
    seeds={k[0] for k in groups}
    game_plan=o.read(HERE/'protocol.json')
    check(seeds==set(game_plan['seeds']) and all(ep['episode_verified'] and ep['terrain']==game_plan['physical_boards'][str(ep['seed'])] for ep in eps),'wrong registered map or signed episode')
    check(len(eps)==24 and len(seeds)==6 and set(groups)=={(s,t) for s in seeds for t in (0,1)} and
          len({ep['terrain'] for ep in eps})==6 and all(len({q['terrain'] for q in eps if q['seed']==s})==1 for s in seeds),'invalid 12 paired units')
    for key,pair in groups.items():
        check(set(pair)=={'draw_a','draw_b'},'unpaired')
        for k,ep in pair.items():
            partner=pair['draw_b' if k=='draw_a' else 'draw_a'];u=o.payoff(ep['official_scores_by_seat'],ep['seat'])[1]
            check(ep['partner_arm']==partner['arm'] and abs(ep['baseline_u']-partner['u'])<1e-12 and
                  abs(ep['u']-u)<1e-12 and abs(ep['advantage']-(u-partner['u']))<1e-12,'sibling baseline mismatch')
    return m,eps


def export_trial(path,base,gradient,step,size):
    net=o.torch_model(base,1);init=base._NETS[1].w;source=BEHAVIOR.read_text();old=o.repack.literal(source,o.repack.STUDENT)
    with torch.no_grad():
        for k,p in net.named_parameters():
            if k in gradient:p.add_(torch.from_numpy(gradient[k]*(step/size)).to(p.dtype))
    with np.load(BASE_NPZ,allow_pickle=False) as ref:
        check(o.repack.quantize(BASE_NPZ,ref)==old,'editable 510c file does not reproduce serving 851c student')
        raw={k:ref[k]+(v.detach().numpy()-init[k]) for k,v in net.state_dict().items()}
        out=path/'candidate.npz';np.savez(out,**raw,meta=ref['meta'])
        fields=o.repack.quantize(out,ref)
    target=path/'candidate.py';target.write_text(o.repack.substitute(source,fields))
    check(target.stat().st_size<=10_000_000 and o.repack.substitute(target.read_text(),old)==source,'natural wrapper or partner changed')
    fresh=o.package(target)
    for k in net.state_dict():
        if k not in gradient:check(np.array_equal(init[k],fresh._NETS[1].w[k]),'frozen tensor changed '+k)
    floating=sum(np.square(net.state_dict()[k].detach().numpy().astype(float)-init[k]).sum() for k in gradient)**.5
    physical=sum(np.square(fresh._NETS[1].w[k].astype(float)-init[k]).sum() for k in gradient)**.5
    check(abs(floating-step)<1e-6,'floating step drift')
    return fresh,dict(step=step,float_step=float(floating),quantized_step=float(physical),pack_sha256=o.sha(target),weights_sha256=o.sha(out),
                      pack_bytes=target.stat().st_size)


def true_free(child,site):
    x,aux,uids=site['public'];feat=sys.modules['cnet2_feat'].Feat(site['seat']);feat.aux,feat.own_uid=aux,uids
    child._FEAT=feat;child._obs_key=(site['step'],site['seat']);child._behavior_rng=menu.MenuDraw()
    initial=feat.to_actions(child._mixed_actions(x,aux))
    raw=child._behavior_rng.menu
    options=[initial] if raw is None else raw['options']
    probs=[1.] if raw is None else raw['probabilities']
    check(options[0]==initial and len({compact(c) for c in options})==len(options),'malformed child menu')
    for cmd in options:
        y=feat.labels(cmd);f=o.actual_features(feat,x,cmd,child)
        check(feat.to_actions(y)==cmd and f['cut']==len(y),'invalid actual physical child option')
    return options,probs


def evaluate_trial(child,roots,sites,data,rows):
    teacher=o.teacher_reader(child);network=o.torch_model(child,1)
    with torch.inference_mode():torch_gaps=np.concatenate([o.evaluate(network,data,indices).numpy() for indices in o.groups(len(rows))])
    shifted=[];weighted=0.;kls=[];max_error=0.
    for i,(feature,r,public) in enumerate(roots):
        x,aux,uids=public;feat=sys.modules['cnet2_feat'].Feat(r['seat']);feat.aux,feat.own_uid=aux,uids
        enc,_,_=o.decode_root(child,feat,x)
        fixed=np.zeros_like(feature['residual']);mix=np.zeros_like(fixed)
        teacher(x,aux,enc,r['route'],feat.labels(r['parent_order']),fixed,mix,r['slot']+1)
        gap=float(np.subtract(*mix[r['slot'],r['moves']].astype(float)))
        max_error=max(max_error,abs(gap-float(torch_gaps[i])))
        p=r['observed_p0'];g0=r['observed_gap'];sign=1 if r['pick']==0 else -1
        lp=lambda g:-float(np.logaddexp(0.,-g))
        dy=lp(sign*gap)-r['old_logp']
        kl=p*(lp(g0)-lp(gap))+(1-p)*(lp(-g0)-lp(-gap))
        check(kl>=-1e-13 and math.isfinite(dy),'invalid physical divergence')
        kls.append(max(0.,kl));weighted+=r['advantage']*dy
        shifted.append(dict(seed=r['seed'],seat=r['seat'],step=r['step'],advantage=r['advantage'],pick=r['pick'],old_logp=r['old_logp'],
                            chosen_logp_shift=dy,old_gap=g0,projected_gap=gap,kl=kl))
    check(max_error<=o.LOGIT_TOL,'torch and actual quantized child projection mismatch')
    free=[];counts=[Counter(),Counter()]
    for site in sites:
        old=site['event'];cmds,p=true_free(child,site);sd=site['seat']
        prior={compact(c):q for c,q in zip(old['options'],old['probabilities'])}
        new={compact(c):q for c,q in zip(cmds,p)}
        maximum=max(abs(prior.get(k,0)-new.get(k,0)) for k in new.keys()|prior.keys())
        new_menu=prior.keys()!=new.keys();flip=old['options'][0]!=cmds[0]
        stats=counts[sd];stats['visited']+=1;stats['free_greedy_flip']+=int(flip);stats['physical_support_change']+=int(new_menu)
        stats['pure_probability_shift']+=int(not new_menu and maximum>=1e-6)
        stats['changed_order_or_probability']+=int(new_menu or maximum>=1e-6)
        missing=compact(old['issued']) not in new;stats['logged_full_order_lost']+=int(missing)
        free.append(dict(seed=site['seed'],seat=sd,arm=site['arm'],step=site['step'],old=old['options'],child=cmds,
                         q0=old['probabilities'],q1=p,max_physical_probability_shift=maximum,logged_order_lost=missing))
    weighted/=288;mean_kl=float(np.mean(kls))
    result=dict(conditional_kl_supported=mean_kl,conditional_kl_scheduled=sum(kls)/288,
                realized_weighted_logp_delta_per_scheduled=weighted,ordinary_free_by_seat=counts,max_projection_gap_error=max_error,
                qualifies=weighted>0 and mean_kl<=.005 and all(v['changed_order_or_probability'] for v in counts))
    return result,dict(projected_decisions=shifted,actual_menus=free)


def map_stability(net,data,rows,seeds,pooled):
    # Each terrain has two seats and two sibling trajectories per seat. All four
    # trajectories stay in a fold; retaining 48 as the denominator counts missed draws as zero.
    pieces={}
    for seed in sorted(seeds):
        index=np.array([ix for ix,r in enumerate(rows) if r['seed']==seed],dtype=int)
        pieces[seed]=a.gradient(net,data,rows,index,48)
    fold=[];sq_error=0.
    for k in pooled:
        avg=sum(g[k] for g in pieces.values())/len(pieces)
        sq_error+=np.vdot(avg-pooled[k],avg-pooled[k]).item()
    for seed,g in pieces.items():
        held_norm=math.sqrt(sum(np.vdot(v,v).item() for v in g.values()))
        vec={k:sum(other[k] for other_seed,other in pieces.items() if other_seed!=seed)/5 for k in g}
        kept_norm=math.sqrt(sum(np.vdot(v,v).item() for v in vec.values()))
        dot=sum(np.vdot(vec[k],g[k]).item() for k in g)
        cos=float(np.clip(dot/(kept_norm*held_norm),-1,1)) if kept_norm and held_norm else None
        group=[r for r in rows if r['seed']==seed]
        fold.append(dict(held_terrain_seed=seed,held_total_scheduled=48,other_total_scheduled=240,
            held_supported=len(group),held_gradient_l2=held_norm,other_five_gradient_l2=kept_norm,
            held_dot_other_five=dot,cosine=cos,angle_degrees=math.degrees(math.acos(cos)) if cos is not None else None))
    check(math.sqrt(sq_error)<1e-5,'mean of map gradients does not reconstruct the all-map direction')
    return dict(folds=fold,mean_of_six_vs_pooled_gradient_error_l2=math.sqrt(sq_error),
                interpretation='Descriptive cross-map training-gradient agreement. The held map and the other five use disjoint episodes and the same frozen policy. This diagnostic does not affect step selection.')


def execute(rev,manifest):
    plan=signed_plan(rev);m,eps=training_manifest(plan,manifest)
    base=o.package(BEHAVIOR);teacher=o.teacher_reader(base);roots=[];sites=[]
    for e in eps:
        r,s=replay_episode(e,base,teacher,plan);roots.extend(r);sites.extend(s)
    summary=dict(learning_sha256=PLAN_SHA,registration_rev=rev,learn_script_sha256=o.sha(Path(__file__)),
                 archive_manifest_sha256=o.sha(manifest),parent_sha256=plan['deployed_behavior'],total_trajectories=24,
                 paired_map_seats=12,physical_maps=6,supported=len(roots),scheduled_reached=len(sites),planned=288,trials=[])
    if not roots:
        o.save(HERE/'fit.json',dict(**summary,status='no_supported_decisions'));return
    data,rows=a.network_data(roots);net=o.torch_model(base,1)
    for k,v in net.named_parameters():v.requires_grad_(not k.startswith(o.VALUE))
    _,_,parity=o.errors(net,data,rows);summary['parent_behavior_replay']=parity
    check(parity['max_signed_logit_difference_error']<=o.LOGIT_TOL and parity['max_probability_error']<=o.PROB_TOL,'851c is not reproduced')
    grad=a.gradient(net,data,rows,np.arange(len(rows)),288);size=math.sqrt(sum(np.vdot(v,v).item() for v in grad.values()))
    check(math.isfinite(size),'nonfinite gradient');summary['gradient_l2']=size
    summary['leave_one_physical_map_out']=map_stability(net,data,rows,{ep['seed'] for ep in eps},grad)
    if size<=plan['epsilon_gradient_l2']:
        o.save(HERE/'fit.json',dict(**summary,status='zero_gradient'));return
    winner=None;audits={}
    with TemporaryDirectory(dir=HERE,prefix='.learning_tmp_') as temp:
        for step in plan['steps']:
            folder=Path(temp)/str(step);folder.mkdir()
            child,g=export_trial(folder,base,grad,step,size)
            rr,aa=evaluate_trial(child,roots,sites,data,rows);g.update(rr)
            summary['trials'].append(g);audits[str(step)]=aa
            if winner is None and g['qualifies']:winner=(folder,g)
            print('trial',step,g['realized_weighted_logp_delta_per_scheduled'],g['conditional_kl_supported'],g['qualifies'],flush=True)
        audit=HERE/'fit_decisions.json.gz';o.save(audit,audits);summary['decision_audit_sha256']=o.sha(audit)
        if winner is None:summary['status']='no_eligible_registered_step'
        else:
            chosen,g=winner
            for name,key in [('candidate.py','pack_sha256'),('candidate.npz','weights_sha256')]:
                target=HERE/name
                if target.exists():check(o.sha(target)==g[key],'existing release differs')
                else:
                    stage=HERE/('.'+name+'.tmp');shutil.copyfile(chosen/name,stage);stage.replace(target)
                check(o.sha(target)==g[key],'published candidate changed')
            summary.update(status='training_mechanism_pass',selected_step=g['step'],selected_package_sha256=g['pack_sha256'],selected_weights_sha256=g['weights_sha256'])
    o.save(HERE/'fit.json',summary)


if __name__=='__main__':
    z=argparse.ArgumentParser(description=__doc__);z.add_argument('--learning-rev',required=True)
    z.add_argument('--manifest',type=Path,default=HERE/'hosted/training_manifest.json');args=z.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    execute(args.learning_rev,args.manifest)
