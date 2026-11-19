"""One nonlinear execution-policy step subject to complete-support, per-root trust gates."""
from collections import Counter, defaultdict
import math
import numpy as np

SETTINGS={'draws':6,'u_rms_floor':.04,'margin_rms_floor':750.,'margin_weight':.10,'map_gradient_norm_limit':.15,
          'first_step_l2':.4,'max_halvings':10,'clip':.2,'max_root_kl':.02,'min_combined_gain':1e-10,
          'min_positive_maps':4}


def _distribution(fn,root,delta):
    z=fn(root,delta);out={}
    for key,(p,grad) in z.items():
        prob=float(p);g=np.asarray(grad,dtype=np.float64)
        if not isinstance(key,str) or not math.isfinite(prob) or prob<0 or prob>1+1e-12 or g.shape!=(15,) or not np.isfinite(g).all():
            raise ValueError('invalid complete whole-command probability or gradient')
        if prob>0:out[key]=(prob,g)
    if abs(math.fsum(v[0] for v in out.values())-1.)>2e-6:
        raise ValueError('whole-command probabilities do not sum to one')
    return out


def _loo(v):
    n=len(v)
    if np.all(v==v[0]):return np.zeros_like(v)
    return (n*v-math.fsum(v))/(n-1)


def _ppo(ratio,adv):
    c=SETTINGS['clip']
    return float(np.mean(np.minimum(ratio*adv,np.clip(ratio,1-c,1+c)*adv)-adv))


def optimize(roots,distribution):
    zero=np.zeros(15,np.float32)
    grouped=defaultdict(list);seen=set();aliases={};baseline=[]
    for r in roots:
        terrain=str(r['terrain']);seed=int(r['seed']);seat=int(r['seat']);step=int(r['t'])
        if not terrain or seat not in (0,1) or step<0 or (terrain,seat,step) in seen or aliases.setdefault(seed,terrain)!=terrain:
            raise ValueError('duplicate or inconsistent physical root')
        seen.add((terrain,seat,step))
        samples=r['samples']
        if len(samples)!=SETTINGS['draws']:raise ValueError('expected exactly six with-replacement draws')
        ref=_distribution(distribution,r,zero)
        keys=[s['key'] for s in samples];u=np.array([s['u'] for s in samples],float);margin=np.array([s['margin'] for s in samples],float)
        if not np.isfinite(u).all() or np.any((u<0)|(u>1)) or not np.isfinite(margin).all():
            raise ValueError('invalid terminal targets')
        for k,s in zip(keys,samples):
            p=float(s['old_p'])
            if k not in ref or not math.isfinite(p) or p<=0 or not math.isclose(p,ref[k][0],rel_tol=1e-5,abs_tol=1e-12):
                raise ValueError('executed-command propensity mismatch')
        au,am=_loo(u),_loo(margin)
        su=max(float(np.sqrt(np.mean(au**2))),SETTINGS['u_rms_floor'])
        sm=max(float(np.sqrt(np.mean(am**2))),SETTINGS['margin_rms_floor'])
        aux=am/sm
        ac=au/su+SETTINGS['margin_weight']*aux
        score=np.stack([ref[k][1]/ref[k][0] for k in keys])
        gradient=np.mean(score*ac[:,None],axis=0)
        uid={'seed':seed,'seat':seat,'terrain':terrain,'t':step}
        grouped[terrain].append(dict(source=r,id=uid,keys=keys,ref=ref,u=au,combined=ac,raw_m=am,gradient=gradient))
        baseline.append(dict(uid,old_support={k:p for k,(p,_) in sorted(ref.items())},u_rms_denominator=su,
            margin_rms_denominator=sm,draws=[dict(key=k,u=float(v),margin=float(m),old_p=ref[k][0],loo_u=float(a),
            loo_margin=float(b),combined_adv=float(c)) for k,v,m,a,b,c in zip(keys,u,margin,au,am,ac)]))
    if not grouped:raise ValueError('no training maps')
    gradients=[];map_weights={}
    for terrain,rows in sorted(grouped.items()):
        g=np.mean([r['gradient'] for r in rows],axis=0);norm=float(np.linalg.norm(g))
        capped=g*(min(1.,SETTINGS['map_gradient_norm_limit']/norm) if norm else 1.)
        if not np.isfinite(capped).all():raise ValueError('non-finite normalized map gradient')
        gradients.append(capped);map_weights[terrain]=dict(roots=len(rows),raw_gradient=g.tolist(),raw_l2=norm,capped_gradient=capped.tolist())
    direction=np.mean(gradients,axis=0);norm=float(np.linalg.norm(direction))
    out={'settings':SETTINGS.copy(),'training_roots':len(seen),'training_maps':len(grouped),'sample_draws':len(seen)*SETTINGS['draws'],
         'map_gradients':map_weights,'capped_gradient_average':direction.tolist(),'baseline':baseline,
         'trials':[],'decision':'KEEP_ZERO','delta':zero.tolist()}
    if norm<=1e-14:
        out['reason']='zero_balanced_direction';return out
    if len(grouped)<SETTINGS['min_positive_maps']:
        out['reason']='fewer_than_required_positive_maps';return out
    for h in range(SETTINGS['max_halvings']+1):
        delta=np.asarray(SETTINGS['first_step_l2']*(2.**-h)*direction/norm,dtype=np.float32)
        map_results={};missing=False;max_kl=0.;ratios=[]
        for terrain,rows in sorted(grouped.items()):
            receipt=[]
            for row in rows:
                ref=row['ref'];now=_distribution(distribution,row['source'],delta)
                lost=sorted(ref.keys()-now.keys());missing|=bool(lost)
                by_key={k:(now[k][0]/v[0] if k in now else 0.) for k,v in ref.items()}
                rho=np.array([by_key[k] for k in row['keys']]);ratios.extend(rho.tolist())
                kl=None if lost else math.fsum(p*(math.log(p)-math.log(now[k][0])) for k,(p,_) in ref.items())
                if kl is not None:
                    if kl < -3e-6:raise ValueError('computed negative forward KL')
                    max_kl=max(max_kl,kl)
                receipt.append(dict(row['id'],old_support_lost=lost,kl_old_new=kl,
                    new_support={k:p for k,(p,_) in sorted(now.items())},new_only_mass=math.fsum(p for k,(p,_) in now.items() if k not in ref),
                    old_support_ratios=by_key,sampled_ratios=rho.tolist(),sample_clip_fraction=float(np.mean((rho<.8)|(rho>1.2))),
                    combined_gain=_ppo(rho,row['combined']),u_gain=_ppo(rho,row['u']),margin_gain=_ppo(rho,row['raw_m'])))
            map_results[terrain]={f:float(np.mean([q[f] for q in receipt])) for f in ('combined_gain','u_gain','margin_gain','sample_clip_fraction')}
            map_results[terrain]['roots']=receipt
        macro={f:float(np.mean([d[f] for d in map_results.values()])) for f in ('combined_gain','u_gain','margin_gain','sample_clip_fraction')}
        positive=sum(d['combined_gain']>0 for d in map_results.values())
        reasons=[]
        if missing:reasons.append('old_support_lost')
        if max_kl>SETTINGS['max_root_kl']:reasons.append('per_root_kl')
        if macro['combined_gain']<=SETTINGS['min_combined_gain']:reasons.append('combined_ppo_gain')
        if macro['u_gain']<=0:reasons.append('raw_official_u_ppo_gain')
        if positive<SETTINGS['min_positive_maps']:reasons.append('insufficient_positive_maps')
        out['trials'].append(dict(halvings=h,delta=delta.tolist(),max_root_kl=None if missing else max_kl,positive_maps=positive,
            sample_ratio_min=min(ratios),sample_ratio_max=max(ratios),maps=map_results,macro=macro,rejections=reasons))
        if not reasons:
            out.update(decision='UPDATE',delta=delta.tolist(),reason='first_feasible_predeclared_step')
            return out
    out['reason']='all_predeclared_steps_rejected'
    return out
