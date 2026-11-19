"""Unclipped trajectory IS for six held-out physical maps and independently sampled siblings."""
from math import exp, fsum, isclose, isfinite, log, sqrt
from statistics import stdev

ARMS=('draw_a','draw_b')
SEEDS=tuple(range(621758460,621758466))


def need(ok,message):
    if not ok:
        raise ValueError(message)


def pnumber(x):
    need(type(x) is not bool and isfinite(float(x)) and 0.<=float(x)<=1.,'Invalid physical option probability')
    return float(x)


def weights_ess(weights):
    total=fsum(weights);squares=fsum(x*x for x in weights)
    return total*total/squares if squares else 0.


def score_trajectory(episode,predictions,consumed):
    seed,seat,arm=episode['seed'],episode['seat'],episode['arm']
    need(episode['episode_verified'] is True and episode['status']=='complete','Missing verified official trajectory')
    expected=list(range(12,min(277,episode['official_transitions']),24))
    need([e['step'] for e in episode['events']]==expected,'Not all registered draw sites appear exactly once')
    log_terms=[];has_zero=False;draws=0;extremes=[]
    for event in episode['events']:
        k=(seed,seat,arm,event['step']);options=event['options'];ch=event['choice']
        need(type(ch) is int and 0<=ch<len(options) and options[ch]==event['issued'],'Wrong actually sampled physical team order')
        if not event['supported']:
            need(len(options)==1 and event['probabilities']==[1.] and ch==0 and event['logp']==0.,'Non-unit probability outside binary menu')
            if k in predictions:
                need(pnumber(predictions[k])==1.,'Changed a deterministic menu outside common support')
                consumed.add(k)
            continue
        need(k in predictions and len(options)==2 and options[0]!=options[1],'Missing held-out physical choice or degenerate support')
        consumed.add(k);draws+=1
        source=list(map(pnumber,event['probabilities']))
        need(len(source)==2 and all(0.<p<1. for p in source) and isclose(sum(source),1.,abs_tol=1e-10,rel_tol=0.),'Invalid behavior policy')
        need(isclose(log(source[ch]),float(event['logp']),abs_tol=1e-9,rel_tol=0.),'Behavior likelihood disagrees with issued order')
        q0=pnumber(predictions[k]);p=source[ch]
        # Two numerically normalized softmax entries can differ from (p0,1-p0)
        # by an ulp. The target is specified by p0 alone; an exact identity
        # must contribute a ratio of one on both actually sampled branches.
        q=p if q0==source[0] else (q0,1.-q0)[ch]
        if q==0.:has_zero=True
        else:log_terms.append(log(q)-log(p))
        extremes.append((q/p,event['step']))
    w=0. if has_zero else exp(fsum(log_terms))
    need(isfinite(w),'Nonfinite unclipped trajectory likelihood ratio')
    u=pnumber(episode['u'])
    return dict(seed=seed,seat=seat,arm=arm,terrain=episode['terrain'],u=u,logged_decisions=draws,
                w=w,log_w=None if has_zero else fsum(log_terms),zero_trajectory_weight=has_zero,
                min_realized_ratio_and_step=min(extremes) if extremes else None,
                max_realized_ratio_and_step=max(extremes) if extremes else None)


def compute(episodes,predictions):
    """predictions[(seed,seat,arm,step)] = held-map target probability of logged physical option zero."""
    episodes=list(episodes)
    wanted={(s,t,a) for s in SEEDS for t in (0,1) for a in ARMS}
    keys={(e['seed'],e['seat'],e['arm']) for e in episodes}
    need(len(episodes)==24 and keys==wanted,'Incorrect map/seat/arm population')
    consumed=set();tr={};pairs=[];maps=[];known_boards=set()
    episode_by_key={(e['seed'],e['seat'],e['arm']):e for e in episodes}
    for k,e in episode_by_key.items():
        tr[k]=score_trajectory(e,predictions,consumed)
    need(consumed==set(predictions),'Orphan or stale target likelihood')
    for seed in SEEDS:
        by_seat=[]
        board={tr[seed,seat,arm]['terrain'] for seat in (0,1) for arm in ARMS}
        need(len(board)==1 and not board&known_boards,'Physical training/validation groups ambiguous')
        known_boards.update(board)
        for seat in (0,1):
            a,b=(tr[seed,seat,n] for n in ARMS)
            for name,other in zip(ARMS,reversed(ARMS)):
                e=episode_by_key[seed,seat,name];other_u=tr[seed,seat,other]['u']
                need(e['partner_arm']==other and e['paired_seed_seat']==[seed,seat]
                     and isclose(e['baseline_u'],other_u,abs_tol=1e-10,rel_tol=0.),
                     'Actual independently sampled sibling return or pair identity disagrees')
            delta=(a['w']-b['w'])*(a['u']-b['u'])/2.
            # Direct two-trajectory estimator using the partner's outcome as its baseline.
            direct=fsum([(a['w']-1.)*(a['u']-b['u']),(b['w']-1.)*(b['u']-a['u'])])/2.
            need(isclose(delta,direct,rel_tol=1e-10,abs_tol=1e-12),'Wrong paired baseline or fixed divisor')
            r=dict(seed=seed,seat=seat,delta_u=delta,w_a=a['w'],w_b=b['w'],u_a=a['u'],u_b=b['u'])
            pairs.append(r);by_seat.append(r)
        local=[tr[seed,seat,arm]['w'] for seat in (0,1) for arm in ARMS]
        maps.append(dict(held_out_seed=seed,terrain=next(iter(board)),delta_u=fsum(x['delta_u'] for x in by_seat)/2.,
                         seat_delta_u=[x['delta_u'] for x in by_seat],mean_w=fsum(local)/4.,ess=weights_ess(local)))
    rows=[tr[k] for k in sorted(tr)]
    w=[r['w'] for r in rows];d=[r['delta_u'] for r in maps]
    point=fsum(d)/6.;positive=sum(k>0 for k in d)
    deleted={str(m['held_out_seed']):fsum(z['delta_u'] for z in maps if z is not m)/5. for m in maps}
    return dict(schema='sapling-paired-native-sixfold-is/1',expected_trajectories=24,paired_map_seats=12,physical_maps=6,
                mean_delta_u=point,positive_maps=positive,
                gate=point>0. and positive>=4,trajectories=rows,pairs=pairs,per_heldout_map=maps,
                by_physical_seat=[dict(seat=s,mean_delta_u=fsum(r['delta_u'] for r in pairs if r['seat']==s)/6.,
                                  positive_map_seats=sum(r['delta_u']>0 for r in pairs if r['seat']==s),
                                  min_delta_u=min(r['delta_u'] for r in pairs if r['seat']==s),
                                  max_delta_u=max(r['delta_u'] for r in pairs if r['seat']==s)) for s in (0,1)],
                adverse_pairs=[r for r in pairs if r['delta_u']<0.],nonpositive_maps=[m['held_out_seed'] for m in maps if m['delta_u']<=0.],
                weights=dict(sum=fsum(w),mean=fsum(w)/24.,min=min(w),max=max(w),ess=weights_ess(w),expected_count=24),
                map_sensitivity=dict(delete_one_map_mean_delta_u=deleted,range=[min(deleted.values()),max(deleted.values())],
                                     descriptive_map_sd=stdev(d),descriptive_map_se=stdev(d)/sqrt(6.)),
                estimator="Mean over 6 physical maps and 2 seats of (W_a-W_b)(U_a-U_b)/2. W is the product of target/behavior probabilities for actually issued orders with no clipping, self-normalization or ignored losses.",
                limitations="Six cross-fitted policies share training maps. Deletion range and map SE describe instability; they do not establish independent validation of the eventual all-map fit, population win rate, or a gain in future games. Valid only with both target orders entirely on the behavior support, other dispatch steps unchanged and a predictor trained before seeing the held-out map.")


def self_test():
    episodes=[];q={}
    for s in SEEDS:
        for seat in (0,1):
            for i,arm in enumerate(ARMS):
                value=(.7,.3)[i]
                ops=[{'k':'UP'},{'k':'DOWN'}]
                ev=dict(step=12,choice=i,options=ops,issued=ops[i],probabilities=[.5,.5],logp=log(.5),supported=True)
                episodes.append(dict(seed=s,seat=seat,arm=arm,terrain=str(s),u=value,partner_arm=ARMS[1-i],
                                     baseline_u=(.7,.3)[1-i],paired_seed_seat=[s,seat],episode_verified=True,
                                     status='complete',official_transitions=14,events=[ev]))
                q[s,seat,arm,12]=.5
    same=compute(episodes,q)
    assert same['mean_delta_u']==0. and same['weights']['ess']==24. and not same['gate']
    q[SEEDS[0],0,'draw_a',12]=.75;q[SEEDS[0],0,'draw_b',12]=.75
    change=compute(episodes,q)
    assert isclose(change['pairs'][0]['delta_u'],.2) and isclose(change['mean_delta_u'],.2/12.) and change['positive_maps']==1
    assert isclose(change['weights']['max'],1.5) and not change['gate']
    four=dict(q)
    for s in SEEDS[1:4]:
        four[s,0,'draw_a',12]=four[s,0,'draw_b',12]=.75
    supported=compute(episodes,four)
    assert supported['gate'] and supported['positive_maps']==4 and supported['mean_delta_u']>0
    for s in SEEDS[4:]:
        for seat in (0,1):
            four[s,seat,'draw_a',12]=four[s,seat,'draw_b',12]=0.
    net_loss=compute(episodes,four)
    assert not net_loss['gate'] and net_loss['positive_maps']==4 and net_loss['mean_delta_u']<0
    q[SEEDS[0],0,'draw_b',12]=1.
    zero=compute(episodes,q)
    assert zero['trajectories'][1]['w']==0. and isclose(zero['pairs'][0]['delta_u'],.3)
    bad=dict(q);bad.pop((SEEDS[0],0,'draw_a',12))
    try:compute(episodes,bad)
    except ValueError:pass
    else:raise AssertionError('Missing choice passed')
    return 'Same-policy zero, known two-arm effect, zero target likelihood, and incomplete support tests passed.'


def files_run(directory):
    import hashlib
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from rerank import export_pack
    here=Path(directory).resolve()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=dict(manifest=here.parent/"hosted/training_manifest.json",plan=here/"protocol.json",
               weights=here/"weights.json",predictions=here/"fold_predictions.json",full_package=here/"research_candidate.py")
    docs={k:json.loads(p.read_bytes()) for k,p in paths.items() if k!="full_package"}
    manifest,pr,weights=docs['manifest'],docs['predictions'],docs['weights']
    need(manifest['training_eligible'] is True and weights['prereg_sha256']==digest(paths['plan'])
         and weights['source_manifest_sha256']==digest(paths['manifest'])
         and manifest['policy_sha256']==weights['source_package_sha256']==pr['parent_package_sha256']==docs['plan']['base_pack_sha256'],
         'Source training register, behavior package, or complete returns changed')
    need(pr['full_package_sha256']==digest(paths['full_package']), 'All-map research package differs')
    need(set(pr['held_fold_package_sha256'])==set(weights['folds'])==set(weights['fit_maps'])==set(map(str,SEEDS)),
         'Incomplete signed out-of-fold inventory')
    with TemporaryDirectory() as tmp:
        for seed in SEEDS:
            key=str(seed)
            need(weights['fit_maps'][key]==sorted(set(SEEDS)-{seed}), 'Held terrain appears in training')
            rebuilt=export_pack(weights['folds'][key],Path(tmp)/(key+'.py'))
            need(rebuilt==pr['held_fold_package_sha256'][key], 'Held fold weights do not recreate its physical policy')
    probs={}
    for ep in pr['episodes']:
        for step,q in ep['q0'].items():
            k=ep['seed'],ep['seat'],ep['arm'],int(step)
            need(k not in probs,'Duplicate held prediction')
            probs[k]=q
    es=manifest['episodes']
    old={(e['seed'],e['seat'],e['arm'],v['step']):v['probabilities'][0] for e in es for v in e['events'] if v['supported']}
    ident=compute(es,old)
    need(ident['mean_delta_u']==0. and ident['positive_maps']==0 and ident['weights']['min']==ident['weights']['max']==1.,'Behavior identity placebo failed')
    result=compute(es,probs)
    result.update(source_sha256={k:digest(v) for k,v in paths.items()},estimator_sha256=digest(Path(__file__)),
                  source_workflow_run=manifest['run_id'],source_supported_events=len(old),same_policy_delta=ident['mean_delta_u'],
                  source_win_draw_loss=[sum(e['wdl']==i for e in es) for i in (1.,.5,0.)],
                  out_of_fold_packages=pr['held_fold_package_sha256'],target_replay_required=True)
    out=here/'ope.json';data=json.dumps(result,sort_keys=True,indent=2,allow_nan=False,ensure_ascii=False)+'\n'
    if out.exists():need(out.read_text()==data,'Prior independent output differs')
    else:
        tmp=out.with_suffix('.tmp');tmp.write_text(data);tmp.replace(out)
    return result


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--read-back',action='store_true')
    args=parser.parse_args();print(self_test())
    if args.read_back:
        r=files_run(Path(__file__).parent)
        print(r['mean_delta_u'],r['positive_maps'],r['gate'],r['weights'])
