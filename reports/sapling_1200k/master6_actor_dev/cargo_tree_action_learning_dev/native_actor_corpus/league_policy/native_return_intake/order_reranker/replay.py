"""Audit a held-map reranking package on real public requests without generating new contests."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
ROOT = next(p for p in HERE.parents if (p/'.git').exists())
sys.path.insert(0,str(INTAKE.parent/'natural_original'))
import audit_menu as parent

TRAINING = '63daa51e127aafea36401ebdce6400de60faa772d3929a1803ce768e121be5b4'
POLICY = '851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee'
STEPS = set(range(12,277,24))


def need(yes,why):
    if not yes:
        raise ValueError(why)


def filehash(path):
    return sha256(path.read_bytes()).hexdigest()


class ActualDraw:
    """Trace the probability actually used; forward the deployed independent random draw."""
    def __init__(self,original):
        self.original=original
        self.event=None

    def random(self):
        need(self.event is None,'More than one draw at a public step')
        site=sys._getframe(1).f_locals
        slot=int(site['slot']);best=int(site['best']);second=int(site['second'])
        probs=list(map(float,site['probs']))
        need(len(probs)==2 and all(math.isfinite(p) and 0<p<1 for p in probs) and abs(sum(probs)-1)<1.e-12,'Invalid physical draw')
        u=float(self.original.random())
        self.event=dict(slot=slot,actions=[best,second],legal=site['trace'][slot][2].tolist(),route=bool(site['route']),
                        prefix=site['baseline'][:slot].tolist(),options=[dict(site['base_cmd']),dict(site['alt_cmd'])],
                        probabilities=probs,pick=int(u>=probs[0]))
        return u


def game(ep,package,expected_p):
    seat,seed=ep['seat'],ep['seed']
    arm=ep['arm'];zpath=ROOT/ep['archive_path']
    need(ep['episode_verified'] and ep['policy_sha256']==POLICY and filehash(zpath)==ep['archive_sha256'],'Input training source changed')
    sites={e['step']:e for e in ep['events']}
    need(set(sites)==STEPS and len(sites)==len(ep['events']),'Irregular opportunity count')
    actor=parent.collect.Player(package)
    counts=Counter()
    values=[];maxerr=0.
    with zipfile.ZipFile(zpath) as z,TemporaryDirectory() as folder:
        reg=Path(folder)/'registration.json'
        reg.write_bytes(z.read(ep['registration_member']))
        receipt=json.loads(z.read(ep['receipt_member']))
        tape=z.read(ep['turns_member'])
        need(filehash(reg)==ep['registration_sha256'] and sha256(tape).hexdigest()==ep['tape_sha256']==receipt['turns_sha256'],'Original tape binding')
        live=parent.open_live(reg)
        env,cfg=live.official();bank=[5.,5.]
        need(sha256(bytes(env.states[0].observation.full_carbon)).hexdigest()==ep['terrain'],'Wrong map')
        draw=None
        for step,line in enumerate(tape.splitlines()):
            frame=json.loads(line)
            need(not env.done and frame['step']==step and frame['calls'][seat]['status']=='ok','Incomplete source segment')
            both=[live.observe(env,s,bank[s]) for s in (0,1)]
            hashes=[sha256(live.encoded(o)).hexdigest() for o in both]
            need(frame['public_observation_sha256']==hashes,'Signed observation diverged')
            public=both[seat];old=frame['calls'][seat]['response']['action']
            if step==1:
                draw=ActualDraw(actor.mod._behavior_rng)
                actor.mod._behavior_rng=draw
            if draw is not None:
                need(actor.mod._behavior_rng is draw,'RNG identity changed inside episode')
                draw.event=None
            issued=actor.act(public,cfg)
            fresh=draw.event if draw is not None else None
            counts['public_calls']+=1
            if step in STEPS:
                historic=sites[step]
                need(historic['public_observation_sha256']==hashes[seat] and historic['issued']==old and
                     historic['options'][historic['choice']]==old,'Historical choice or menu observation changed')
                if historic['supported']:
                    need(fresh is not None,'Binary support collapsed')
                    for k in ('options','actions','route','slot','prefix','legal'):
                        need(fresh[k]==historic[k],f'Physical support differs at {seed}/{seat}/{arm}/{step}: {k}')
                    need(fresh['options'][fresh['pick']]==issued,'Actual package output mismatches selected physical order')
                    expected=expected_p(ep,historic)
                    err=abs(fresh['probabilities'][0]-expected)
                    need(err<=2.e-6,'Served score does not match held-out model')
                    maxerr=max(maxerr,err)
                    delta=fresh['probabilities'][0]-historic['probabilities'][0]
                    values.append(dict(step=step,p_base=historic['probabilities'][0],p_fold=fresh['probabilities'][0],
                                       p_fold_reference=expected,actual_new_pick=fresh['pick'],historical_pick=historic['choice']))
                    counts.update({'supported':1,'probability_changed':int(abs(delta)>1.e-6),
                                   'historical_order_in_new_support':1,'natural_alternate_draws':fresh['pick']})
                else:
                    need(fresh is None and issued==old==historic['options'][0] and historic['probabilities']==[1.],
                         'No-support site changed')
                    counts['scheduled_singleton']+=1
            else:
                need(fresh is None and issued==old,'Ordinary physical order or random-call schedule changed')
                counts['ordinary']+=1
            bank=frame['overage_remaining_seconds']
            env.step([c['response']['action'] for c in frame['calls']])
        need(env.done and [p.reward for p in env.states]==ep['official_scores_by_seat'] and
             counts['public_calls']==ep['official_transitions'],'Physical history mismatch')
    actor.diagnostics()
    return dict(seed=seed,seat=seat,arm=arm,terrain=ep['terrain'],counts=dict(counts),served_sha256=filehash(package),
                max_p_error=maxerr,roots=values)


def scored_replay(task):
    episode,pack,prediction=task
    expected={str(v['step']) for v in episode['events'] if v['supported']}
    need(set(prediction)==expected,'Unexpected or missing prediction roots')
    record=game(episode,pack,lambda ep,h:float(prediction[str(h['step'])]))
    def ratio(root):
        p,q=root['p_base'],root['p_fold']
        return q/p if root['historical_pick']==0 else (1-q)/(1-p)
    record['uncapped_logged_trajectory_weight']=math.exp(math.fsum(math.log(ratio(v)) for v in record['roots']))
    return record


def compute(workers):
    import rerank
    manifest=INTAKE/'hosted/training_manifest.json'
    need(filehash(manifest)==TRAINING,'Unexpected training input')
    source=json.loads(manifest.read_bytes())
    path_weights,path_pred=HERE/'weights.json',HERE/'fold_predictions.json'
    weights=json.loads(path_weights.read_bytes())
    prediction_card=json.loads(path_pred.read_bytes())
    need(prediction_card['parent_package_sha256']==POLICY,'Wrong served parent')
    pred=prediction_card['episodes']
    bykey={(q['seed'],q['seat'],q['arm']):q['q0'] for q in pred}
    episodes=source['episodes'];seeds=sorted({e['seed'] for e in episodes})
    tickets={(s,p,a) for s in seeds for p in (0,1) for a in ('draw_a','draw_b')}
    need(len(seeds)==6 and len(episodes)==len(pred)==len(tickets)==24 and
         {(e['seed'],e['seat'],e['arm']) for e in episodes}==tickets==set(bykey) and
         {str(s) for s in seeds}==set(weights['folds']),'Missing or extra out-of-fold prediction')
    audit=[]; folds={}; fit_returns={}
    with TemporaryDirectory() as tmp:
        base=Path(tmp)
        tasks=[]
        for s in seeds:
            pack=base/f'held_{s}.py'
            rerank.export_pack(weights['folds'][str(s)],pack)
            need(pack.stat().st_size<=10_000_000,'Oversized fold')
            folds[str(s)]=filehash(pack)
            need(folds[str(s)]==prediction_card['held_fold_package_sha256'][str(s)],'Wrong pre-frozen holdout file')
            for e in [v for v in episodes if v['seed']==s]:
                key=(s,e['seat'],e['arm'])
                tasks.append((e,pack,bykey[key]))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            audit=list(pool.map(scored_replay,tasks))
        for (ep,_,_),g in zip(tasks,audit):
            fit_returns[ep['seed'],ep['seat'],ep['arm']]=(g['uncapped_logged_trajectory_weight'],ep['u'])
        full=base/'all_training_maps.py'
        rerank.export_pack(weights['full'],full)
        full_sha=filehash(full)
        need(full_sha==prediction_card['full_package_sha256'],'Wrong all-map documentary file')
        need(full.stat().st_size<=10_000_000,'Oversized all-six documentary export')
    counts=Counter()
    for g in audit:
        counts.update(g['counts'])
    need(counts['public_calls']==7176 and counts['ordinary']==6888 and counts['supported']==275 and
         counts['scheduled_singleton']==13 and counts['historical_order_in_new_support']==275,'Wrong whole-corpus census')
    pairs=[]
    for s in seeds:
        for seat in (0,1):
            wa,ua=fit_returns[s,seat,'draw_a'];wb,ub=fit_returns[s,seat,'draw_b']
            pairs.append(dict(seed=s,seat=seat,actual_order_paired_estimate=(wa-wb)*(ua-ub)/2))
    n=len(pairs)
    maps={str(s):math.fsum(v['actual_order_paired_estimate'] for v in pairs if v['seed']==s)/2 for s in seeds}
    ws=[w for w,_ in fit_returns.values()]
    return dict(schema='sapling-rerank-physical-command-audit/1',scope='Out-of-map-fold, sequential replay of already exposed training observations. Native draws are newly sampled from the held-map policy. Physical transitions and scores here reproduce the archived original actions.',
                source_training_manifest_sha256=TRAINING,source_policy_sha256=POLICY,prereg_sha256=filehash(HERE/'protocol.json'),
                weights_sha256=filehash(path_weights),offline_predictions_sha256=filehash(path_pred),exporter_sha256=filehash(Path(rerank.__file__)),
                auditor_sha256=filehash(Path(__file__)),fold_package_sha256=folds,all_training_package_sha256=full_sha,
                totals=dict(counts),new_native_games=0,independent_development_games=0,
                max_served_vs_fit_p0_difference=max(v['max_p_error'] for v in audit),
                actual_order_offline_estimator=dict(pairs=pairs,by_map=maps,mean=math.fsum(maps.values())/len(maps),
                    positive_maps=sum(v>0 for v in maps.values()),trajectory_ess=math.fsum(ws)**2/math.fsum(w*w for w in ws)),
                games=audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'audit.json')
    parser.add_argument('--workers',choices=(1,2,3),type=int,default=2)
    args=parser.parse_args()
    data=compute(args.workers)
    if args.out.exists():
        old=json.loads(args.out.read_bytes())
        need(all(old[k]==data[k] for k in ('weights_sha256','fold_package_sha256','actual_order_offline_estimator',
                                           'source_training_manifest_sha256','max_served_vs_fit_p0_difference')),
             'Existing release describes a different replay')
    else:
        temporary=args.out.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        temporary.replace(args.out)
    print(args.out,filehash(args.out),data['totals'],data['actual_order_offline_estimator']['mean'])
