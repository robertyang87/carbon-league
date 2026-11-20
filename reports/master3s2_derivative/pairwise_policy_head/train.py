"""Learn a residual on original search preferences from paired official continuations."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import types
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
REPEATED = FAMILY/'terminal_pair_replication'
HELD = FAMILY/'original_action_pairs'
PARENT = FAMILY/'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
REGULARIZATION = 1.0
CORRECTION_LIMIT = 0.04


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(data):
    return (json.dumps(data,indent=2,allow_nan=False)+'\n').encode()


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def freeze(path, data):
    raw=encode(data)
    assert not path.exists() or path.read_bytes()==raw
    path.write_bytes(raw)
    return sha(raw)


def protocol():
    files = sorted(REPEATED.glob('part*/rows.json.gz'))
    assert len(files)==4
    return dict(schema='pairwise-neural-root-preference/1',parent=PARENT_SHA,
                train={str(p.relative_to(FAMILY)):sha(p.read_bytes()) for p in files},
                held_protocol=sha((HELD/'protocol.json').read_bytes()),
                features='Acting-seat root: original frozen 96 activations at entities, accumulated by action type and averaged by team size. Difference between action orders. Separate worker and base blocks.',
                fit='Equal terrain weights. Regress (mean official first-minus-second margin)/3000 minus native score difference. No intercept. Scale all columns by one common weighted RMS; ridge 1.0.',
                decision='Add correction capped at ±0.04 to the native difference. In a candidate, use a per-order ±0.02 cap. Compare the same two complete native finalists, first on ties.',
                promotion='Positive map-equal official gain on the held panel with at least two changed, improved physical maps. No tune or refit based on held.',
                interpretation='Training uses two true responses per root; held has one. Conditional suffixes only, using archived signed observation prefixes and no future or hidden features.')


def read_history():
    src=FAMILY/'root_rotation_policy/native/probe.py'
    method=next(n for n in ast.parse(src.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='each_history')
    ns=dict(D=FAMILY,gzip=gzip,json=json,sha=sha,zipfile=zipfile)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(src),'exec'),ns)
    return ns['each_history']


def model():
    raw=PARENT.read_bytes()
    assert sha(raw)==PARENT_SHA
    end=b'        return lw.T, lb.T'
    assert raw.count(end)==1
    m=types.ModuleType('frozen_pairwise_features')
    exec(compile(raw.replace(end,b'        return hf.T, lw.T, lb.T'),str(PARENT),'exec'),m.__dict__)
    m._torch.set_num_threads(1)
    assert m._SEARCH_OK and m._net() is not None and m._TEMP==1 and m._S_ABORT==.58
    return m


def descriptor(h, orders, players, seat, model):
    bases, workers=players[seat][1:3]
    size=h.shape[1]
    w=np.zeros((5,size),dtype=np.float64)
    b=np.zeros((3,size),dtype=np.float64)
    for uid, (pos,*_) in workers.items():
        w[model.WIDX[orders.get(uid,'NONE')]]+=h[pos]
    for uid,pos in bases.items():
        b[model.BIDX[orders.get(uid,'NONE')]]+=h[pos]
    return np.concatenate((w.ravel()/max(1,len(workers)),b.ravel()/max(1,len(bases))))


def features(ticket, orders, m, history):
    tracker=m.BeliefTracker(m._Bel,ticket['seat'])
    prior=None
    for current, issued, digest in history(ticket):
        step,seat=current['step'],ticket['seat']
        estimate,known=tracker.step(current['players'],step,current['carbon'])
        if step==ticket['step']:
            assert digest==ticket['own_public_sha256']
            x=np.clip(np.rint(m.encode(current['players'],step,seat,estimate,known,prior)*100.),0,255)*.01
            h,_,_=m._POLICY.forward(x,m._SID)
            return [descriptor(h,a,current['players'],seat,m) for a in orders]
        tracker.set_actions(current['players'],issued)
        prior=current['players']
    raise ValueError('Inaccessible signed root')


def training(m, spec, history):
    data=[]
    base_protocol=(REPEATED/'protocol.json').read_bytes()
    for key,digest in spec['train'].items():
        path=FAMILY/key
        assert sha(path.read_bytes())==digest
        stored=read(path)
        assert stored['protocol']==sha(base_protocol)
        for row in stored['rows']:
            t=row['ticket']
            if not row['branches']:
                continue
            indices=[row['choices'][arm] for arm in ('first','second')]
            actions=[row['roots']['menu'][i] for i in indices]
            leaf=[row['roots']['leaf_scores'][i] for i in indices]
            observed=[]
            for r in row['branches']:
                for arm,order in zip(('first','second'),actions,strict=True):
                    rec=(path.parent/f"{r['stream']}_{t['seed']}_{t['seat']}_{t['step']}_{arm}.json.gz").read_bytes()
                    assert sha(rec)==r['arms'][arm]['evidence_sha256']
                    tape=json.loads(gzip.decompress(rec))
                    assert tape['rows'][0]['orders'][t['seat']]==order and tape['outcome']['statuses']==['DONE','DONE']
                observed.append(r['arms']['first']['margin']-r['arms']['second']['margin'])
            x=features(t,actions,m,history)
            data.append(dict(ticket=t, x=(x[0]-x[1]).tolist(), prior=leaf[0]-leaf[1], observed=observed))
    assert len(data)==16 and len({i['ticket']['terrain'] for i in data})==8
    return data


def fit(data):
    x=np.array([i['x'] for i in data],dtype=np.float64)
    y=np.array([np.mean(i['observed'])/3000.-i['prior'] for i in data])
    terrains=Counter(i['ticket']['terrain'] for i in data)
    w=np.array([1/terrains[i['ticket']['terrain']] for i in data])
    w /= w.sum()
    scale=np.sqrt(w@np.sum(x*x,axis=1))
    design=x*np.sqrt(w[:,None])/scale
    head=design.T@np.linalg.solve(design@design.T+REGULARIZATION*np.eye(len(x)),np.sqrt(w)*y)/scale
    assert head.shape==(768,) and np.isfinite(head).all()
    return head


def measured(m, spec, head, histories):
    raw=(HELD/'protocol.json').read_bytes()
    assert sha(raw)==spec['held_protocol']
    rows=[]
    for t in json.loads(raw)['tickets']:
        name=f"{t['seed']}_{t['seat']}_{t['step']}"
        d=HELD/str(t['seed'])
        r=read(d/f'{name}_result.json')
        z=(d/f'{name}_root.json.gz').read_bytes()
        assert r['root_sha256']==sha(z) and r['protocol_sha256']==sha(raw)
        if 'outcome' not in r:
            rows.append(dict(ticket=t,status=r['status']))
            continue
        root=json.loads(gzip.decompress(z))
        signs=[]
        root_orders=[arm['order'] for arm in root['arms']]
        for key,order in zip(('first','second'),root_orders,strict=True):
            outcome=r['outcome']['arms'][key]
            record=(d/f'{name}_{key}.json.gz').read_bytes()
            assert sha(record)==outcome['evidence_sha256']
            tape=json.loads(gzip.decompress(record))
            assert tape['rows'][0]['orders'][t['seat']]==order and tape['outcome']['statuses']==['DONE','DONE']
            signs.append(outcome['margin'])
        x=features(r['ticket'],root_orders,m,histories)
        corrections=[float(np.clip(a@head,-CORRECTION_LIMIT/2,CORRECTION_LIMIT/2)) for a in x]
        native=root['arms'][0]['original_value']-root['arms'][1]['original_value']
        new=native+corrections[0]-corrections[1]
        rows.append(dict(ticket=t,status='complete', native=native, learned=new, corrections=corrections,
                         terminal_margins=signs, picked='second' if new<0 else 'first',
                         gain=signs[1]-signs[0] if new<0 else 0.0))
    complete=[r for r in rows if r['status']=='complete']
    maps={t['terrain'] for t in json.loads(raw)['tickets']}
    gains={k:float(np.mean([r['gain'] for r in complete if r['ticket']['terrain']==k])) for k in sorted(maps)}
    return dict(rows=rows,complete=len(complete),changed=sum(r['picked']=='second' for r in complete), map_gains=gains,
                mean_gain=float(np.mean(list(gains.values()))),qualified=sum(n>0 for n in gains.values())>=2 and np.mean(list(gains.values()))>0)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('stage',choices=('register','fit','evaluate'))
    args=cli.parse_args()
    lock=HERE/'protocol.json'
    if args.stage=='register':
        print(freeze(lock,protocol()))
    else:
        assert lock.read_bytes()==encode(protocol())
        os.environ['CARBON_DEBUG']='1'
        m=model()
        history=read_history()
        weights = HERE/'weights.json'
        spec=read(lock)
        if args.stage=='fit':
            data=training(m,spec,history)
            head=fit(data)
            fit_sha=freeze(HERE/'training_rows.json',{'rows':data})
            print(freeze(weights,dict(protocol=sha(lock.read_bytes()),training=fit_sha,
                                     cap=CORRECTION_LIMIT,weights=head.tolist())),np.linalg.norm(head))
        else:
            saved=read(weights)
            assert saved['protocol']==sha(lock.read_bytes()) and saved['training']==sha((HERE/'training_rows.json').read_bytes())
            result=measured(m,spec,np.array(saved['weights']),history)
            print(freeze(HERE/'held.json',result|dict(weights_sha256=sha(weights.read_bytes()))),
                  {k:v for k,v in result.items() if k!='rows'})
