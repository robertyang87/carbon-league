"""Apply frozen return models to new native-examiner trajectories with explicit public and outcome clocks."""
import argparse
from collections import deque
from copy import deepcopy
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
import sys
import zipfile

HERE=Path(__file__).resolve().parent
REP=HERE.parents[1]/'replication'
sys.path.insert(0,str(HERE.parent))
import analyze as historical
from menu import threats
from zerosum_env import make

SEEDS=(621758510,621758511,621758512,621758513)
ARMS=('parent','bc')
MODELS=('geometry','neural')


def require(ok,reason):
    if not ok:raise ValueError(reason)


def read(p):
    return json.loads(p.read_bytes())


def hfile(p):
    return sha256(p.read_bytes()).hexdigest()


def pick_target(observation):
    t=observation['step']
    targets=threats(observation) if 40<=t<=240 and t%10==0 else []
    return targets[0] if targets else None


def target_label(events,start,uid,enemy,completed):
    """Match the training first-trip boundary; signed terminal windows may be shorter than ten."""
    subset=events[start:start+10]
    if len(subset)<10 and not completed:
        return dict(observed=len(subset),eligible=False,reason='right_censored_in_incomplete_game')
    deposits=[];boundary=None
    for row in subset:
        paid=[e for e in row['payments'] if e[3]==uid]
        for e in paid:
            if e[0]=='own_delivery':
                require(e[1]==enemy and e[2]>=0,'Own deposit beneficiary differs from selected enemy')
                deposits.append([row['t'],e[2]])
        if paid or uid in row['dead']:
            boundary=row['t']
            break
    amount=math.fsum(v[1] for v in deposits)
    return dict(observed=len(subset),eligible=True,first_trip_ends_at=boundary,ordinary_own_deposits=deposits,
                amount=amount,positive=int(amount>0))


def verify_trajectory(game,archive,plan,model):
    import numpy as np
    seed,seat,arm=game['seed'],game['seat'],game['arm']
    with zipfile.ZipFile(archive) as z:
        rec_bytes=z.read(arm+'/receipt.json');receipt=json.loads(rec_bytes)
        reg=json.loads(z.read(arm+'/registration.json'))
        require(sha256(rec_bytes).hexdigest()==game['receipt_sha256'] and
                reg['seed']==seed and reg['candidate_seat']==seat and reg['arm']==arm and
                reg['physical_map_sha256']==plan['physical_boards'][str(seed)]==game['terrain'] and
                reg['files']['candidate']['sha256']==plan['files'][arm]['sha256'] and
                reg['files']['opponent']['sha256']==plan['files']['opponent']['sha256'] and
                receipt['status']==game['status'],'Unbound signed game')
        data=z.read(arm+'/turns.jsonl') if arm+'/turns.jsonl' in z.namelist() else b''
        require(not data and not receipt.get('turns_sha256') or sha256(data).hexdigest()==receipt.get('turns_sha256'),
                'Turn tape and official receipt differ')
    o=historical.NATIVE.parent/'direct_original/games/candidate/raw_621758346_1.json.gz'
    cfg={**historical.read(o)['game_configuration'],'agentTimeout':20,'actTimeout':1,'runTimeout':1500}
    np.random.seed(seed)
    env=make('carbon',configuration={**cfg,'randomSeed':seed})
    sim=historical.FastCarbon(cfg).reset(seed=seed)
    require(historical.sha(bytes(map(int,sim.carbon)))==game['terrain']==historical.sha(bytes(env.states[0].observation.full_carbon)),
            'Opening engines or physical maps differ')
    sim.cash=historical.CashEvents(sim.cash)
    history=deque(maxlen=11);roots=[];events=[];commands=[];banks=[5.,5.];seen_slots=0
    for t,line in enumerate(data.splitlines()):
        if t>=game['official_transitions']:break
        obs=sim.obs(seat)
        full=env.states[0].observation
        require(obs['step']==t and obs['players']==full.players and obs['trees']==full.trees and
                obs['carbon']==list(env.states[seat].observation.carbon),'Official input and student public observer differ')
        history.append(deepcopy(obs))
        target=pick_target(obs)
        if t in range(40,241,10):seen_slots+=1
        if target:
            uid=target['target_uid']
            require(len(history)==11 and [v['step'] for v in history]==list(range(t-10,t+1)),'Noncontiguous observer history')
            estimates=model.predict(list(history),uid)
            require(all(isinstance(estimates.get(k),(int,float,np.floating)) and math.isfinite(estimates[k])
                        for k in (*MODELS,'geometry_cash','neural_cash')) and
                    all(0<=estimates[name]<=1 and estimates[name+'_cash']>=0 for name in MODELS),'Invalid frozen predictions')
            roots.append(dict(seed=seed,seat=seat,arm=arm,terrain=game['terrain'],step=t,target_uid=uid,
                              cargo=target['target_cargo'],home_distance=target['target_home_distance'],
                              public_sha256=historical.digest(obs),prefix_sha256=historical.digest(commands),
                              history_sha256=historical.digest(list(history)),
                              predictions={k:float(estimates[k]) for k in (*MODELS,'geometry_cash','neural_cash')}))
        # The selected UID and both predictions are sealed above before loading this joint action.
        tick=json.loads(line)
        require(tick['step']==t and all(c['status']=='ok' for c in tick['calls']),'Unapplied state transition')
        public=dict(step=full.step,player=seat,carbon=env.states[seat].observation.carbon,players=full.players,
                    trees=full.trees,remainingOverageTime=banks[seat])
        # Match the live observer's public field order, compact JSON and terminating newline.
        actual=historical.sha((json.dumps(public,separators=(',',':'),allow_nan=False)+'\n').encode())
        require(actual==tick['public_observation_sha256'][seat],'Observer input or physical seat time budget mismatch')
        issued=[c['response']['action'] for c in tick['calls']]
        commands.append(issued);sim.cash.events.clear();sim.step(*issued);env.step(issued)
        require(list(sim.cash)==[p[0] for p in env.states[0].observation.players],'Two engines disagree on cash')
        events.append(dict(t=t,payments=[list(e) for e in sim.cash.events
             if e[0] in ('own_delivery','opponent_worker_delivery_to_this_base')],
             dead={a[0] for a in sim.last_collisions}))
        banks=tick['overage_remaining_seconds']
    require(len(events)==game['official_transitions'],'Wrong number of applied transitions')
    done=game['status']=='complete'
    if done:
        require(env.done and [s.reward for s in env.states]==game['official_scores_by_seat'],'Official terminal differs')
    for r in roots:
        r['label']=target_label(events,r['step'],r['target_uid'],1-seat,done)
    ticket=dict(seed=seed,seat=seat,arm=arm,game_status=game['status'],official_transitions=len(events),
                scheduled_slots=seen_slots,roots_with_threat=len(roots),receipt_sha256=game['receipt_sha256'],
                eligible_labels=sum(r['label']['eligible'] for r in roots))
    return ticket,roots


def unique(records):
    first={};duplicates=[]
    for row in records:
        if not row['label']['eligible']:continue
        key=(row['terrain'],row['seat'],row['step'],row['prefix_sha256'],row['public_sha256'])
        if key in first:
            base=first[key]
            require(row['target_uid']==base['target_uid'] and row['history_sha256']==base['history_sha256'] and
                    row['predictions']==base['predictions'],'Same public prefix produced different target or forecast')
            duplicates.append(dict(seed=row['seed'],seat=row['seat'],step=row['step'],kept_arm=base['arm'],extra_arm=row['arm'],
                 conflicting_labels=row['label']['positive']!=base['label']['positive']))
        else:first[key]=row
    return list(first.values()),duplicates


def summary(records):
    rows,extra=unique(records)
    result=dict(stored_occurrences=len(records),n=len(rows),duplicates=len(extra),
                duplicate_label_conflicts=sum(x['conflicting_labels'] for x in extra),
                positives=sum(r['label']['positive'] for r in rows))
    for model in MODELS:
        groups=[[] for _ in range(5)]
        losses=[];brier=[];cash=[];correct=0
        for row in rows:
            p=row['predictions'][model];y=row['label']['positive'];groups[min(4,int(5*p))].append((p,y))
            brier.append((p-y)**2)
            losses.append(-math.log(max(1e-12,p if y else 1-p)));correct+=int((p>=.5)==bool(y))
            cash.append(abs(row['predictions'][model+'_cash']-row['label']['amount']))
        result[model]=dict(brier=mean(brier) if rows else None,nll=mean(losses) if rows else None,
            cash_mae=mean(cash) if rows else None,threshold_correct=correct,
            calibration=[dict(lower=i/5,upper=(i+1)/5,n=len(g),positives=sum(y for _,y in g),
                              predicted=mean([p for p,_ in g]) if g else None,observed=mean([y for _,y in g]) if g else None)
                         for i,g in enumerate(groups)])
    result['geometry_right_neural_wrong']=sum((r['predictions']['geometry']>=.5)==bool(r['label']['positive']) and
                                            (r['predictions']['neural']>=.5)!=bool(r['label']['positive']) for r in rows)
    result['neural_right_geometry_wrong']=sum((r['predictions']['neural']>=.5)==bool(r['label']['positive']) and
                                            (r['predictions']['geometry']>=.5)!=bool(r['label']['positive']) for r in rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settlement',type=Path,default=REP/'hosted/settlement.json');args=parser.parse_args()
    from predict import PublicReturn
    import numpy as np
    from metrics import evaluate
    require(hfile(HERE/'release.json')=='f540b4abc48a5a147a4361f219b8b7bb07ed7e4c13a92a9a9d65f6bfe36b4596','Different preregistered prediction model')
    model=PublicReturn()
    freeze={f:hfile(HERE/f) for f in ('weights.npz','predict.py','release.json','protocol.json')}
    expected=read(HERE/'protocol.json')
    plan=read(REP/'protocol.json');receipt=read(args.settlement)
    require(tuple(plan['seeds'])==SEEDS and tuple(plan['arms'])==ARMS and
            len(set(plan['physical_boards'].values()))==4 and
            not set(plan['physical_boards'].values())&(set(expected['held_terrains'])|set(expected['training_terrains'])) and
            receipt['protocol_sha256']==hfile(REP/'protocol.json') and receipt['registered_games']==16,
            'Wrong settlement or overlapping calibration/training maps')
    games={(g['seed'],g['seat'],g['arm']):g for g in receipt['games']}
    require(len(games)==len(receipt['games'])==16 and set(games)=={(s,p,a) for s in SEEDS for p in (0,1) for a in ARMS},
            'Wrong external ticket denominator')
    archive_paths={hfile(p):p for p in args.settlement.parent.glob('*.zip')}
    archives={(a['seed'],a['seat']):archive_paths.get(a['sha256']) for a in receipt['archives']}
    rows=[];tickets=[]
    for seed in SEEDS:
        for seat in (0,1):
            for arm in ARMS:
                key=seed,seat,arm;g=games[key];path=archives.get((seed,seat))
                if path is None or g['official_transitions'] is None:
                    tickets.append(dict(seed=seed,seat=seat,arm=arm,game_status=g['status'],evaluated=False,
                                        reason='Signed archive or verified turn count unavailable'))
                    continue
                ticket,group=verify_trajectory(g,path,plan,model);tickets.append(dict(ticket,evaluated=True));rows.extend(group)
    require(freeze=={f:hfile(HERE/f) for f in freeze},'Model inputs changed during evaluation')
    maps={str(s):summary([r for r in rows if r['seed']==s]) for s in SEEDS}
    per_ticket={f'{s}_{p}_{a}':summary([r for r in rows if (r['seed'],r['seat'],r['arm'])==(s,p,a)]) for s,p,a in games}
    error_metrics=('brier','nll','cash_mae')
    equal={model:{k:mean(z[model][k] for z in maps.values()) if all(z[model][k] is not None for z in maps.values()) else None
                  for k in error_metrics} for model in MODELS}
    unique_rows,dups=unique(rows)
    registered_metrics=None
    if unique_rows and len({r['seed'] for r in unique_rows})==len(SEEDS):
        values=[dict(amount=r['label']['amount'],terrain=r['terrain'],seed=r['seed']) for r in unique_rows]
        predictions={k:(np.array([r['predictions'][k] for r in unique_rows]),
                        np.array([r['predictions'][k+'_cash'] for r in unique_rows])) for k in MODELS}
        registered_metrics=evaluate(values,predictions)
        require(all(abs(registered_metrics['summary'][k][metric]-equal[k][metric])<1e-10
                    for k in MODELS for metric in error_metrics),'Map-equal metrics disagree')
    record=historical.write(HERE/'external_roots.json.gz',dict(rows=rows),compressed=True)
    result=dict(scope='Frozen public probability of the selected native opponent carrying worker making its first ordinary own-base deposit over the next 10 turns. External-only evaluation, no fitting or game action changes.',
        frozen_inputs=freeze,source_sha256=hfile(args.settlement),evaluation_sha256=hfile(Path(__file__)),
        workflow_run=receipt['workflow_run'],registered_games=16,tickets=tickets,
        physical_maps=4,roots_sha256=record,raw_root_occurrences=len(rows),distinct_eligible_roots=len(unique_rows),
        pooled_duplicates=dups,source_game_files=[{'seed':a['seed'],'seat':a['seat'],'sha256':a['sha256']} for a in receipt['archives']],
        pooled=summary(rows),map_equal=equal,by_map=maps,
        by_arm={a:summary([r for r in rows if r['arm']==a]) for a in ARMS},
        by_seat={str(p):summary([r for r in rows if r['seat']==p]) for p in (0,1)},
        by_map_arm={f'{s}_{a}':summary([r for r in rows if r['arm']==a and r['seed']==s]) for s in SEEDS for a in ARMS},
        by_ticket=per_ticket,nll_clip=1e-12,registered_map_evaluation=registered_metrics,
        metric_program_sha256=hfile(HERE/'metrics.py'),
        deduplication='Within each reported cohort keep its first eligible occurrence in registered (map,seat,parent/bc) order, keyed by terrain, seat, step, preceding joint actions and public frame; preserve all occurrences and report any conflicting outcomes.',
        right_censoring='Normal official game end closes the horizon at its recorded last turn. In incomplete matches the final fewer-than-10-step windows are enumerated and excluded.',
        target_selection='t=40,50,...,240. Active enemy collector cargo>=100 and distance 2..10 to nearest enemy base; maximum cargo/(1+distance), UID tiebreaker; no current orders used.')
    historical.write(HERE/'external.json',result)
    print(json.dumps({'tickets':len(tickets),'row_count':len(rows),'unique':len(unique_rows),'macro':equal},ensure_ascii=False))


if __name__=='__main__':
    main()
