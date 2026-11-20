"""Grade fixed choices against archived, genuinely responding terminal branches, one map per ticket."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
LOCK = '7fc66af456ef7458aae2a53c2c0cdd86db90b4e192931eb540f3d74c5481998f'
CONFIG = HERE/'protocol.json'
REP = 10000
EPS = 1e-8


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(p):
    data = p.read_bytes()
    if p.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


def labels(t):
    g, s, tick = t['group'], t['seat'], t['step']
    stem = f"{t['seed']}_{s}_{tick}"
    close = g == 'close'
    if close:
        base = D/'return_response_learning/close_games'
        file = base/'cases'/f"{t['close_ordinal']}.json"
        root = read(file)
        assert (root['seed'], root['seat'], root['step']) == (t['seed'], s, tick)
        assert root['menu_sha256'] == t['root_sha']
        base = base/'evidence'
        names = ['A', 'B']
    else:
        base = D/'action_rank_data'/g
        file = base/(stem+'_pair.json')
        root = read(file)
        assert root['status'] == 'complete' and root['root_sha256'] == t['root_sha']
        assert all(t[k] == root['ticket'][k] for k in ('seed','seat','step','terrain'))
        names = ['first','second']
    receipt, y = [], []
    for name in names:
        arm = root['arms'][name]
        archive = base/(stem+'_'+name+'.json.gz')
        assert sha(archive.read_bytes()) == arm['evidence_sha256']
        trace = read(archive)
        done = trace['outcome']
        assert done['scores'] == arm['scores'] and done['statuses'] == arm['statuses'] == ['DONE','DONE']
        m = arm['scores'][s] - arm['scores'][1-s]
        assert abs(m-arm['margin']) < 1e-7
        turns = trace['rows']
        assert [r['t'] for r in turns] == list(range(tick, tick+len(turns)))
        assert len(turns) == arm['transitions'] and len(turns) >= 12
        bank = [sum(r['flow'][side].get('own_delivery',0.) for r in turns[:12]) for side in (0,1)]
        net = bank[s] - bank[1-s]
        if close:
            assert np.allclose(arm['total_first12_own_delivery_by_seat'], bank, atol=1e-7)
            assert abs(arm['first12_net_delivery']-net) < 1e-7
        y.append(dict(scores=arm['scores'], margin=m, win=int(m>EPS), draw=int(abs(m)<=EPS),
                      loss=int(m < -EPS), net12=net, deposits_by_seat=bank))
        receipt.append(dict(path=str(archive.relative_to(D)),sha256=arm['evidence_sha256']))
    return dict(outcomes=y, receipts=receipt, pair_sha256=sha(file.read_bytes()))


def evaluate(c, p):
    assert all(p[k] == c[k] for k in ('group','seed','seat','step','terrain','root_sha','source_sha','parent'))
    assert p['draw_seeds'] == c['draw_seeds'] and len(set(p['draw_seeds'])) == 8
    rec = dict(group=c['group'],seed=c['seed'],seat=c['seat'],step=c['step'],terrain=c['terrain'],status=p['status'])
    if c['status'] == 'skipped':
        assert p['status'] == 'historical_skip'
        return rec
    rec.update(labels(c))
    if p['status'] != 'complete':
        assert p['status']=='reconstruction_failed'
        rec['reason']=p['reason']
        return rec
    original, baseline = np.array(p['native_values']), np.array(p['native_reproduced'])
    draws = np.asarray(p['draw_values'],dtype=float)
    assert draws.shape == (8,2) and np.all(np.isfinite(draws)) and np.max(abs(original-baseline)) <= 2e-4
    delta = draws[:,1]-draws[:,0]
    dif = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    native = int(original[1] > original[0])
    mean = int(dif>0) if dif != 0 else native
    single = int(delta[0]>0) if delta[0] != 0 else native
    assert (native, mean, single) == (p['choice_native'], p['choice_mean'], p['choice_fresh_single'])
    t7 = 2.36462425
    interval = [dif - t7*se, dif + t7*se]
    assert np.allclose(interval,p['conditional_t7_95'],rtol=1e-9,atol=1e-12)
    rec.update(choose={'native':native,'mean8':mean,'fresh1':single}, mc_conditional=dict(
            delta_B_minus_A=dif,draw_delta=delta.tolist(),se=se, t7_95=interval,
            n_B_preferred=int(np.count_nonzero(delta>0)),n_equal=int(np.count_nonzero(delta==0)),
            n_A_preferred=int(np.count_nonzero(delta<0))))
    return rec


def summarize(rows):
    rows=[r for r in rows if r['status']=='complete']
    if not rows:
        return dict(n=0)
    assert len(rows) == len(set(r['terrain'] for r in rows))
    n=len(rows)
    samples=np.random.default_rng(712908).integers(0,n,(REP,n))
    def summary(v):
        a=np.asarray(v,dtype=float)
        return dict(mean=float(a.mean()),map_bootstrap_95=np.quantile(a[samples].mean(axis=1),[.025,.975]).tolist())
    def accuracy(c, truth):
        c=np.asarray(c)
        usable=np.abs(truth)>EPS
        correct=usable & (c==(truth>0))
        n_ok=int(usable.sum())
        result=dict(correct=int(correct.sum()),different=n_ok,tied=n-n_ok,accuracy=None, map_bootstrap_95=None)
        if n_ok:
            num=correct[samples].sum(axis=1);den=usable[samples].sum(axis=1)
            result.update(accuracy=float(correct.sum()/n_ok),map_bootstrap_95=np.quantile(num[den>0]/den[den>0],[.025,.975]).tolist())
        return result
    fields={k:np.array([[a[k] for a in r['outcomes']] for r in rows]) for k in ('margin','net12','win','draw','loss')}
    choice={k:np.array([x['choose'][k] for x in rows]) for k in ('native','mean8','fresh1')}
    overall={}
    for name, local in choice.items():
        orig=choice['native'];group={}
        for metric, a in fields.items():
            v=a[np.arange(n),local]
            ref=a[np.arange(n),orig]
            group[metric]=dict(selected=summary(v),gain_over_native=summary(v-ref))
            if metric in ('margin','net12'):
                group[metric]['higher_choice']=accuracy(local,a[:,1]-a[:,0])
            else:
                group[metric]['count']=int(v.sum())
        group['switches_from_native']=int(np.count_nonzero(local!=orig))
        overall[name]=group
    return dict(n=n,by_policy=overall,
                actual_wdl_changed_between_arms=int(np.count_nonzero(
                        (fields['win'][:,0]!=fields['win'][:,1]) | (fields['draw'][:,0]!=fields['draw'][:,1]))),
                conditional_mc_ci_excludes_zero=sum(r['mc_conditional']['t7_95'][0]>0 or r['mc_conditional']['t7_95'][1]<0 for r in rows))


def main(audit_only=False):
    lock=sha(CONFIG.read_bytes())
    assert lock == LOCK
    tickets=read(CONFIG)['tickets']
    assert len({t['terrain'] for t in tickets})==len(tickets)==16
    if audit_only:
        ledger=[labels(t) for t in tickets if t['status']!='skipped']
        print('Verified',len(ledger),'existing official pairs and',len(ledger)*2,'suffix receipts')
        return
    records=[]
    for t in tickets:
        path=HERE/'predictions'/f"{t['group']}_{t['seed']}_{t['seat']}_{t['step']}.json"
        pred=read(path)
        assert pred['protocol'] == lock
        records.append(evaluate(t,pred) | {'prediction_sha256':sha(path.read_bytes())})
    groups={g:summarize([r for r in records if r['group']==g]) for g in ('train','held','close')}
    pooled=summarize([r for r in records if r['group'] in ('held','close')])
    groups['held_plus_close']=pooled
    complete=all(groups[g]['n']==d for g,d in [('train',7),('held',4),('close',4)])
    def margin(g, policy, field):
        return groups[g]['by_policy'][policy]['margin'][field]
    passed=(complete and all(margin(g,'mean8','gain_over_native')['mean']>0 for g in ('held','close','held_plus_close'))
            and margin('held_plus_close','mean8','higher_choice')['correct']>margin('held_plus_close','native','higher_choice')['correct'])
    doc=dict(protocol_sha256=lock,observed_status_counts=dict(Counter(r['status'] for r in records)),groups=groups,
             positive_pilot=bool(passed),all_eligible_reconstructed=complete,records=records,
             uncertainty='Map percentile bootstrap, 10000 replicates seeded 712908. T7 within-root interval estimates only simulation variation over eight independent seeds conditional on a frozen model. Each recorded real arm has one responding trajectory; natural per-root outcome uncertainty is unestimated. All four hold and four close maps were previously exposed; academic retrospective only.')
    raw=(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    path=HERE/'score.json'
    if path.exists():
        assert path.read_bytes()==raw
    else:
        tmp=path.with_suffix('.tmp')
        tmp.write_bytes(raw)
        tmp.replace(path)
    print('Complete',sha(raw),'positive_pilot',passed, flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audit-labels-only',action='store_true')
    main(p.parse_args().audit_labels_only)
