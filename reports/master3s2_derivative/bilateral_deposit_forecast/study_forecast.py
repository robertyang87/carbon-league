"""Train and verify a ten-step score model for both sides' actual credited deposits."""

import argparse
from collections import Counter, defaultdict
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

from features import NAMES, raw

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
ROOT=FAMILY.parents[1]
sys.path.insert(0,str(FAMILY/'v2_original_cash_paths'))
from study import EXAMINER, LEDGER, PARENT, select, sha

H=10
STEPS=tuple(range(40,290,3))
HELD={'native_speed_dev','switch_margin_dev'}
TRAFFIC=('own_delivery','opponent_worker_delivery_to_this_base')


def save(name,data):
    content=(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    if name.endswith('.gz'):
        content=gzip.compress(content,mtime=0)
    path=HERE/name
    tmp=path.with_suffix('.partial')
    tmp.write_bytes(content)
    tmp.replace(path)
    return sha(content)


def key(t):
    return (t['panel'],t['game']['terrain'],t['game']['seat'])


def plan(tickets):
    return dict(schema='bilateral-actual-deposit-forecast/1', source=sha(Path(__file__).read_bytes()),
                feature_sha256=sha((HERE/'features.py').read_bytes()), packages=[PARENT,EXAMINER],
                horizon=H, steps=list(STEPS), held_panels=sorted(HELD), features=NAMES,
                features_baseline=NAMES[:2], ridge=.05, danger_threshold=-250,
                target='Realized credited deposit cash difference over the following ten official joint moves. '
                       'No immediate commands, future cash, hidden carbon, or map-id features.',
                tickets=[dict(identity=key(t),archive=t['archive_sha256'],receipt=t['game']['receipt_sha256'],
                              registration=t['registration_sha256']) for t in tickets])


def collect(tickets,split):
    spec=importlib.util.spec_from_file_location('cash_event_replayer',LEDGER)
    ledger=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    cfg=ledger.config()
    results=[]
    for t in tickets:
        if (split=='held') != (t['panel'] in HELD):
            continue
        game=t['game']
        prefix=game['arm']+'/'
        seat=game['seat']
        file=ROOT/t['archive']
        assert sha(file.read_bytes())==t['archive_sha256']
        with zipfile.ZipFile(file) as z:
            book=ledger.replay(game,z,cfg)
            views=gzip.decompress(z.read(prefix+'observations.jsonl.gz')).splitlines()
            assert sha(z.read(prefix+'registration.json'))==t['registration_sha256']
        turns=book['transitions']
        assert len(views)==len(turns)==299
        for step in STEPS:
            row=json.loads(views[step])
            obs=row['observations'][seat]
            assert row['step']==obs['step']==step and obs['player']==seat and 'full_carbon' not in obs
            p=obs['players']
            assert row['public_observation_sha256']==turns[step]['public_observation_sha256']
            assert sha((json.dumps(obs,separators=(',',':'))+'\n').encode())==row['public_observation_sha256'][seat]
            x=[a-b for a,b in zip(raw(p,seat,step),raw(p,1-seat,step),strict=True)]
            y=[sum(t['signed_cash_by_seat'][i].get(k,0.) for k in TRAFFIC for t in turns[step:step+H]) for i in (0,1)]
            results.append(dict(terrain=game['terrain'],seed=game['seed'],seat=seat,step=step,
                                public_sha256=row['public_observation_sha256'][seat], x=x, df=y[seat]-y[1-seat],
                                credited=y, official_margin=book['official_margin']))
        print(split,game['seed'],seat,flush=True)
    return results


def fit(rows,cols,lam):
    import numpy as np
    x=np.array([r['x'][:cols] for r in rows])
    y=np.array([r['df'] for r in rows])
    counts=Counter(r['terrain'] for r in rows)
    w=np.array([1/counts[r['terrain']] for r in rows]); w/=w.sum()
    scale=np.maximum(1e-6,np.sqrt(w@np.square(x)))
    z=x/scale
    adjusted=np.linalg.solve((z.T*w)@z+lam*np.eye(cols),(z.T*w)@y)
    return (adjusted/scale).tolist()


def review(rows,weights,limit):
    from statistics import mean
    groups=defaultdict(list)
    for r in rows:
        r['predicted']={tag:sum(w*x for w,x in zip(beta,r['x'])) for tag,beta in weights.items()}
        groups[r['terrain']].append(r)
    result={}
    for name,group in groups.items():
        local={}
        for key in weights:
            y=[r['df'] for r in group]
            pred=[r['predicted'][key] for r in group]
            local[key]=dict(mae=mean(abs(a-b) for a,b in zip(y,pred,strict=True)),
                            alert=sum(b<=limit for b in pred), positives=sum(a<=limit for a in y),
                            true_alert=sum(a<=limit and b<=limit for a,b in zip(y,pred,strict=True)),
                            false_alert=sum(a>limit and b<=limit for a,b in zip(y,pred,strict=True)))
        result[name]=dict(rows=len(group), models=local)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['register','fit','held'])
    ap.add_argument('--cpu',type=int,required=True)
    a=ap.parse_args()
    os.sched_setaffinity(0,{a.cpu})
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[k]='1'
    tickets=select()
    protocol=plan(tickets)
    path=HERE/'protocol.json'
    if a.mode=='register':
        assert not path.exists() or json.loads(path.read_bytes())==json.loads(json.dumps(protocol))
        print(save('protocol.json',protocol),flush=True)
        return
    assert json.loads(path.read_bytes())==json.loads(json.dumps(protocol))
    if a.mode=='held':
        rec=json.loads((HERE/'weights.json').read_bytes())
        assert rec['protocol_sha256']==sha(path.read_bytes())
        assert rec['train_rows_sha256']==sha((HERE/'fit_rows.json.gz').read_bytes())
        weights=rec['coefficients']
    rows=collect(tickets,a.mode)
    if a.mode=='fit':
        weights=dict(totals=fit(rows,2,protocol['ridge']), routes=fit(rows,len(NAMES),protocol['ridge']))
    else:
        assert set(rec['train_maps']).isdisjoint({r['terrain'] for r in rows})
    maps=review(rows,weights,protocol['danger_threshold'])
    digest=save(a.mode+'_rows.json.gz',rows)
    if a.mode=='fit':
        save('weights.json',dict(coefficients=weights, train_rows_sha256=digest,
                                 train_maps=sorted(maps), protocol_sha256=sha(path.read_bytes())))
    save(a.mode+'.json',dict(rows=len(rows), maps=maps, input_digest=digest, weights_sha256=sha((HERE/'weights.json').read_bytes())))


if __name__=='__main__':
    main()
