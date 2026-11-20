"""Fit a small signed residual on measured paired terminal returns, by physical map."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from statistics import mean

import numpy as np

from collect import HERE, encode, sha, write


def protocol():
    return dict(schema='terminal-pair-sixteen-ridge/1', collection=sha((HERE/'protocol.json').read_bytes()),
                lambda_ridge=.1, label='mean over two paired differences of exact native terminal utility',
                input=['cash_margin','cargo_margin','near_cashable_margin','tree_count_margin'],
                scoring='No intercept on candidate-first minus second differences. Keep native score as residual offset. '
                        'Use both repeated original responses for label; score each stream separately as well.',
                test='Report original and learned terminal utility error, action direction and actual official margin, '
                     'held supervision by physical map. Also report both streams sign agreement.')


def read(split):
    data=[]
    expected=json.loads((HERE/'protocol.json').read_bytes())['tickets']
    seen=set()
    for folder in sorted(HERE.glob('part*')):
        raw=(folder/'rows.json.gz').read_bytes()
        part=json.loads(gzip.decompress(raw))
        assert part['protocol']==sha((HERE/'protocol.json').read_bytes())
        for row in part['rows']:
            t=row['ticket']
            if t['split']!=split:
                continue
            ident=(t['seed'],t['seat'])
            assert ident not in seen and t in expected
            seen.add(ident)
            if not row['branches']:
                data.append(dict(ticket=t, skipped=True))
                continue
            indices=row['choices']
            a,b=[str(indices[k]) for k in ('first','second')]
            roots=row['roots']
            actual=[]
            for pair in row['branches']:
                vals=[]
                for arm in ('first','second'):
                    out=pair['arms'][arm]
                    name=f"{pair['stream']}_{t['seed']}_{t['seat']}_{t['step']}_{arm}.json.gz"
                    packet=(folder/name).read_bytes()
                    assert sha(packet)==out['evidence_sha256']
                    event=json.loads(gzip.decompress(packet))
                    assert event['rows'][0]['orders'][t['seat']]==roots['menu'][indices[arm]]
                    assert event['rows'][0]['orders'][1-t['seat']]==event['root_shared_original_order']
                    assert [x['t'] for x in event['rows']]==list(range(t['step'],299))
                    assert out['statuses']==['DONE','DONE']
                    vals.append(dict(margin=out['margin'],u=terminal(out['margin']),sha=out['evidence_sha256'],
                                     root_enemy=event['root_shared_original_order']))
                assert vals[0]['root_enemy']==vals[1]['root_enemy']
                actual.append(dict(margin=vals[0]['margin']-vals[1]['margin'],u=vals[0]['u']-vals[1]['u'],arms=vals))
            assert len(actual)==2
            data.append(dict(ticket=t,skipped=False, native=roots['leaf_scores'][int(a)]-roots['leaf_scores'][int(b)],
                             x=[u-v for u,v in zip(row['leaves'][a],row['leaves'][b],strict=True)],
                             streams=actual, observed=mean(x['u'] for x in actual)))
    assert seen=={(t['seed'],t['seat']) for t in expected if t['split']==split}
    return data


def terminal(margin):
    return float(np.clip(margin/3000,-2.5,2.5)+.5*np.sign(margin))


def fit(rows,lock):
    rows=[r for r in rows if not r['skipped']]
    x=np.asarray([r['x'] for r in rows])
    n=Counter(r['ticket']['terrain'] for r in rows)
    w=np.asarray([1/n[r['ticket']['terrain']] for r in rows]);w/=w.sum()
    y=np.asarray([r['observed']-r['native'] for r in rows])
    norm=np.maximum(1e-9,np.sqrt(w@np.square(x)))
    z=x/norm
    ridge=lock['lambda_ridge']
    correction=np.linalg.solve((z.T*w)@z+ridge*np.eye(z.shape[1]),(z.T*w)@y)/norm
    return correction.tolist()


def measure(rows,weights):
    groups=defaultdict(list)
    for r in rows:
        if r['skipped']:
            continue
        e=r['native']
        r['predicted']=dict(native=e, learned=e+float(np.dot(weights,r['x'])), cash_only=r['x'][0]/3000)
        r['same_direction_both_streams']=bool(np.sign(r['streams'][0]['margin'])==np.sign(r['streams'][1]['margin']))
        groups[r['ticket']['terrain']].append(r)
    metrics={}
    for terrain,group in groups.items():
        metrics[terrain]={}
        for label in ('native','learned','cash_only'):
            values=[(r['predicted'][label],r['observed'],mean(a['margin'] for a in r['streams'])) for r in group]
            metrics[terrain][label]={'mae':mean(abs(a-b) for a,b,_ in values),
                                    'correct':sum(int(np.sign(a)==np.sign(b)) for a,b,_ in values),
                                    'incremental_margin':mean(-m if a<0 else 0 for a,_,m in values)}
    return metrics


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('stage',choices=['register','fit','held'])
    a=cli.parse_args()
    lock=protocol()
    reg=HERE/'training_protocol.json'
    if a.stage=='register':
        assert not reg.exists() or reg.read_bytes()==encode(lock)
        print(write(reg,lock))
    else:
        assert reg.read_bytes()==encode(lock)
        data=read(a.stage)
        target=HERE/'head.json'
        if a.stage=='fit':
            weights=fit(data,lock)
            write(target,dict(weights=weights, protocol=sha(reg.read_bytes()),
                              training_map=sorted({r['ticket']['terrain'] for r in data})))
        else:
            saved=json.loads(target.read_bytes())
            assert saved['protocol']==sha(reg.read_bytes())
            assert set(saved['training_map']).isdisjoint({r['ticket']['terrain'] for r in data})
            weights=saved['weights']
        metrics=measure(data,weights)
        print(write(HERE/(a.stage+'.json'),dict(weights=sha(target.read_bytes()),by_map=metrics,rows=data)))
