"""Train a one-coefficient late-game value mixture on official action pairs."""
import argparse
import gzip
import json
from statistics import mean

from collect import HERE, FAMILY, encode, sha

OLD = FAMILY/'action_rank_data'


def distance(a, b):
    row, col = divmod(a, 15)
    other_row, other_col = divmod(b, 15)
    dr, dc = abs(row-other_row), abs(col-other_col)
    return min(dr, 15-dr) + min(dc, 15-dc)


def cash_plus_collectible(leaf, own):
    """Uncontested upper bound on conversion of currently carried carbon."""
    payout = []
    remaining = 299-leaf['leaf_step']
    for cash, bases, workers, _ in leaf['players']:
        bankable = sum(
            cargo for position, cargo, role in workers.values()
            if role == 'COLLECTOR' and remaining > 0
            and any(distance(position, base) <= remaining for base in bases.values())
        )
        payout.append(cash+bankable)
    return payout[own]-payout[1-own]


def examples(root, outcome):
    seat = root['ticket']['seat']
    assert [a['name'] for a in root['arms']]==['first','second']
    assert all(len(a['leaves'])==1 and not a['leaves'][0]['exact_terminal'] for a in root['arms'])
    native = [3000*a['original_value'] for a in root['arms']]
    flow = [cash_plus_collectible(a['leaves'][0],seat) for a in root['arms']]
    labels = [outcome[k]['margin'] for k in ('first','second')]
    assert all(a['statuses']==['DONE','DONE'] for a in outcome.values())
    return dict(seed=root['ticket']['seed'], seat=seat, terrain=root['ticket']['terrain'],
                native=native[0]-native[1], flow=flow[0]-flow[1],
                actual=labels[0]-labels[1], scores=labels)


def training():
    rows = []
    for split in ('train','held'):
        for p in sorted((OLD/split).glob('*_pair.json')):
            raw = p.read_bytes()
            labels = json.loads(raw)
            if labels['status']!='complete':
                continue
            q = p.with_name(p.name.replace('_pair.json','_root.json.gz'))
            root = q.read_bytes()
            assert sha(root)==labels['root_sha256']
            rows.append(examples(json.loads(gzip.decompress(root)),labels['arms']) | dict(
                root_sha256=sha(root), label_sha256=sha(raw), root_path=str(q.relative_to(FAMILY)), labels_path=str(p.relative_to(FAMILY))))
    assert len(rows)==len({r['terrain'] for r in rows})==11
    return rows


def train():
    rows = training()
    delta = [r['flow']-r['native'] for r in rows]
    target = [r['actual']-r['native'] for r in rows]
    # Prior shrinkage equal to one root's expected squared correction.
    scale = mean(d*d for d in delta)
    coefficient = max(0.0, min(1.0, sum(d*t for d,t in zip(delta,target,strict=True))/(sum(d*d for d in delta)+scale)))
    return dict(schema='official-cash-conversion-mixture/1', coefficient=coefficient, training=rows,
                holdout_protocol_sha256=sha((HERE/'protocol.json').read_bytes()),
                goal='Predict official first-minus-second margin. Convex blend on paired differences. '
                     'Re-use all eleven labeled historical terrains for fit. The new panel is disjoint and outcomes are not opened by fit.')


def assess(fit):
    protocol_raw = (HERE/'protocol.json').read_bytes()
    assert fit['holdout_protocol_sha256']==sha(protocol_raw)
    rows=[]
    for t in json.loads(protocol_raw)['tickets']:
        name=f"{t['seed']}_{t['seat']}_{t['step']}"
        p=HERE/str(t['seed'])
        r=json.loads((p/f'{name}_result.json').read_bytes())
        z=(p/f'{name}_root.json.gz').read_bytes()
        assert r['root_sha256']==sha(z) and r['protocol_sha256']==sha(protocol_raw)
        assert all(r['ticket'][k]==t[k] for k in t)
        if 'outcome' not in r:
            rows.append(dict(ticket=t, status=r['status']))
            continue
        root=json.loads(gzip.decompress(z))
        for label, arm in r['outcome']['arms'].items():
            evidence=(p/f'{name}_{label}.json.gz').read_bytes()
            assert sha(evidence)==arm['evidence_sha256'] and arm['transitions']==25
            tape=json.loads(gzip.decompress(evidence))
            assert tape['rows'][0]['orders'][1-t['seat']]==tape['root_shared_original_order']
            assert tape['rows'][0]['orders'][t['seat']]==next(a['order'] for a in root['arms'] if a['name']==label)
        assert sha((p/f'{name}_leaf.npz').read_bytes())==r['leaf_sha256']
        row=examples(root,r['outcome']['arms'])
        assert row['terrain'] not in {old['terrain'] for old in fit['training']}
        predicted = row['native'] + fit['coefficient']*(row['flow']-row['native'])
        rows.append(row | dict(status='paired_eligible', predicted=predicted,
                               picked='second' if predicted<0 else 'first',
                               gain=0.0 if predicted>=0 else -row['actual']))
    scored=[r for r in rows if r['status']=='paired_eligible']
    terrain_ids=sorted({r['terrain'] for r in scored})
    return dict(rows=rows, complete=len(scored), registered=len(rows), changed=sum(r['picked']=='second' for r in scored),
                mean_gain_by_map=mean(mean(r['gain'] for r in scored if r['terrain']==t) for t in terrain_ids) if terrain_ids else None)


if __name__ == '__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('stage',choices=('fit','evaluate'))
    args=cli.parse_args()
    path=HERE/'weights.json'
    if args.stage=='fit':
        data=encode(train())
        assert not path.exists() or path.read_bytes()==data
        path.write_bytes(data)
        print(sha(data), json.loads(data)['coefficient'])
    else:
        raw=path.read_bytes()
        assert raw==encode(train())
        data=assess(json.loads(raw)) | dict(frozen_weights=sha(raw))
        (HERE/'held.json').write_bytes(encode(data))
        print({k:v for k,v in data.items() if k!='rows'})
