"""Consume the signed intake once, audit held-map decisions and run the conditional native comparison."""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
T=HERE.parent
SUPER=T/'supplement_01/completion.json'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(a,b):
    if not math.isclose(a,b,abs_tol=1.e-7,rel_tol=1.e-10):
        raise ValueError(f'independent numeric check differs: {a} versus {b}')


def fit_lstsq(rows,spec):
    import numpy as np
    group=Counter(r['board'] for r in rows)
    w=np.sqrt([1/(len(group)*group[r['board']]) for r in rows])
    feats=np.array([r['x'] for r in rows])
    x=np.column_stack((np.ones(len(rows)),feats))*w[:,None]
    y=np.array([r['y'] for r in rows])*w
    L=np.diag(np.sqrt([spec['intercept_penalty']]+[spec['slope_penalty']]*feats.shape[1]))
    beta=np.linalg.lstsq(np.vstack([x,L]),np.concatenate([y,np.zeros(L.shape[0])]),rcond=None)[0]
    constant=sum(r['y']/(len(group)*group[r['board']]) for r in rows)/(1+spec['intercept_penalty'])
    return beta,constant,feats.min(axis=0),feats.max(axis=0)


def decision(row,beta,lo,hi,spec):
    x=row['x'];parent='pay' if row['pay_is_v1'] else 'defer'
    if any(v<a-spec['support_pad'] or v>b+spec['support_pad'] for v,a,b in zip(x,lo,hi)):return parent
    pred=float(beta[0]+sum(w*v for w,v in zip(beta[1:],x)))
    if parent=='pay' and pred<=-spec['minimum_predicted_departure']:return 'defer'
    if parent=='defer' and pred>=spec['minimum_predicted_departure']:return 'pay'
    return parent


def gain(row,choice):
    selected=row['pay_margin'] if choice=='pay' else row['defer_margin']
    initial=row['pay_margin'] if row['pay_is_v1'] else row['defer_margin']
    return selected-initial


def verify(status,rows,evaluation,model,reg):
    from scipy.stats import t
    boards=sorted({r['board'] for r in rows});paid=[r for r in rows if r['filled']]
    if len({(r['board'],r['seat'],r['t'],r['uid']) for r in rows})!=len(rows):raise ValueError('duplicate paired observation')
    counts={'paid_pairs':len(paid),'paid_terrains':len({r['board'] for r in paid}),
            'paid_positive_terrains':len({r['board'] for r in paid if r['y']>0}),'paid_negative_terrains':len({r['board'] for r in paid if r['y']<0})}
    if counts!=status['support']['counts'] or status['support']['sufficient']!=all(counts[k]>=v for k,v in reg['minimum'].items()):
        raise ValueError('incorrect fitted population')
    if not status['support']['sufficient']:
        if evaluation is not None or model is not None or status['eligible_for_development']:raise ValueError('premature fitted output')
        return {'counts':counts,'passed':False,'reason':'insufficient_realized_support'}
    if set(boards)!={r['board'] for r in evaluation['folds']} or len(boards)!=len(evaluation['folds']):
        raise ValueError('missing or repeated held map')
    spec=reg['ridge'];fmap={f['board']:f for f in evaluation['folds']};result=[]
    for b in boards:
        kept=[r for r in paid if r['board']!=b]
        beta,constant,lo,hi=fit_lstsq(kept,spec);fold=fmap[b]
        if set(fold['fold_weights']['training_terrains'])!={r['board'] for r in kept}:raise ValueError('leakage or omitted training map')
        for actual,stored in zip(beta,fold['fold_weights']['beta']):assert_close(actual,stored)
        items=[r for r in rows if r['board']==b]
        idx={(r['seat'],r['t'],r['uid']):r for r in fold['roots']}
        if len(idx)!=len(items):raise ValueError('missing held intentions')
        gs=[];cs=[];changed=0
        for row in items:
            choice=decision(row,beta,lo,hi,spec);ref=decision(row,[constant]+[0.]*(len(beta)-1),lo,hi,spec)
            stored=idx[row['seat'],row['t'],row['uid']]
            if (stored['choice'],stored['constant_choice'],stored['root_hash'])!=(choice,ref,row['root_hash']):
                raise ValueError('out-of-terrain policy action differs')
            g=gain(row,choice);c=gain(row,ref);assert_close(g,stored['gain']);assert_close(c,stored['constant_gain'])
            gs.append(g);cs.append(c);changed+=choice!=('pay' if row['pay_is_v1'] else 'defer')
        result.append({'board':b,'intent':len(items),'paid':sum(r['filled'] for r in items),'changed':changed,
                       'gain':statistics.mean(gs),'constant':statistics.mean(cs)})
    rates=[r['gain'] for r in result];avg=statistics.mean(rates);avg_c=statistics.mean(r['constant'] for r in result)
    lower=avg-float(t.ppf(.95,len(rates)-1))*statistics.stdev(rates)/math.sqrt(len(rates))
    gates=reg['development_gate']
    checks={'mean_gain':avg>=gates['minimum_all_intent_mean_gain'],'lower':lower>gates['one_sided_t_lower_must_exceed'],
            'changed_maps':sum(r['changed']>0 for r in result)>=gates['minimum_changed_terrains'],
            'positive_map_fraction':sum(g>0 for g in rates)/len(rates)>=gates['minimum_positive_terrain_fraction'],'beat_constant':avg>avg_c}
    for actual,stored in zip(fit_lstsq(paid,spec)[0],model['beta']):assert_close(actual,stored)
    assert_close(avg,evaluation['equal_map_means']['gain']);assert_close(lower,evaluation['lower_one_sided_t95'])
    if checks!=evaluation['checks'] or all(checks.values())!=status['eligible_for_development']:raise ValueError('development gate differs')
    return {'counts':counts,'maps':result,'equal_terrain_mean_gain':avg,'constant_gain':avg_c,'t_lower_one_sided':lower,
            'checks':checks,'passed':all(checks.values()),'reason':status['stage']}


def final():
    import train
    sup=read(SUPER)
    if sup['phase'] not in ('registered_count_target_reached','all_registered_blocks_exhausted'):
        raise ValueError('intake series has not settled')
    if sup['registration_sha256']!=digest(T/'supplement_01/registration.json'):raise ValueError('other intake registration')
    for batch in sup['counts']['source_intakes']:
        if digest(T/batch['dir']/'training_labels.json')!=batch['labels_sha256']:raise ValueError('changed training batch')
    status=train.publish(T,HERE)
    rows=read(HERE/'samples.json.gz')
    if (len(rows),sum(r['filled'] for r in rows))!=(sup['counts']['sealed_pairs'],sup['counts']['actual_paid_pairs']):
        raise ValueError('research aggregate and independently collected rows differ')
    have=status['support']['sufficient']
    finding=verify(status,rows,read(HERE/'evaluation.json') if have else None,read(HERE/'weights.json') if have else None,read(HERE/'protocol.json'))
    finding.update({'completion_sha256':digest(SUPER),'status_sha256':digest(HERE/'status.json'),'sample_tape_sha256':digest(HERE/'samples.json.gz'),
                    'conditional_native_source':'development_runs/comparison.json' if finding['passed'] else None})
    from source_games import trial
    trial.frozen_file(HERE/'independent_acceptance.json',finding)
    if finding['passed']:
        if (T/'.continuation.lock').exists():
            with (T/'.continuation.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX)
        import development
        while True:
            try:
                result=development.play();break
            except BlockingIOError:time.sleep(8)
        if result['games']!=8 or result['formal']:raise ValueError('development is incomplete')
        print(result['verdict'],flush=True)
    print(json.dumps({'support':finding['counts'],'held_terrain_pass':finding['passed'],'reason':finding['reason']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('wait','final'));command=parser.parse_args().command
    if command=='final':
        if not SUPER.exists():raise SystemExit('complete all registered native training blocks or reach their count target first')
        sys.path.insert(0,str(T))
        # Bare module names of the audited trainer take precedence over the upstream intake helpers.
        sys.path.insert(0,str(HERE))
        final();return
    with (HERE/'.handoff-owner.lock').open('a') as fd:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('head supervisor already owns the data handoff');return
        while not SUPER.exists():time.sleep(20)
        subprocess.run([sys.executable,'-u',str(Path(__file__)),'final'],env=ENV,check=True)


if __name__=='__main__':main()
