"""Score the actual joint opponent commands from the pre-registered development roots."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import zipfile

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def actual(ticket):
    archive=FAMILY/ticket['archive']
    assert sha(archive.read_bytes())==ticket['archive_sha256']
    t=ticket['step']
    prefix=ticket['arm']+'/'
    with zipfile.ZipFile(archive) as z:
        reg=z.read(prefix+'registration.json')
        receipt=json.loads(z.read(prefix+'receipt.json'))
        tape=z.read(prefix+'turns.jsonl')
        views=gzip.decompress(z.read(prefix+'observations.jsonl.gz'))
    assert sha(reg)==ticket['registration_sha256']==receipt['protocol_sha256']
    assert sha(tape)==receipt['turns_sha256']
    row=json.loads(tape.splitlines()[t])
    view=json.loads(views.splitlines()[t])
    assert row['step']==view['step']==t
    assert row['public_observation_sha256']==view['public_observation_sha256']
    seat=1-ticket['seat']
    own=view['observations'][ticket['seat']]['players'][seat]
    return row['calls'][seat]['response']['action'],set(own[1]),set(own[2])


def equal_work(row):
    a,b=row['equal_work']['v4'],row['equal_work']['critic']
    selected=a[0]
    keys=('candidates','depth','rollouts','sorted_depths','menu','act0')
    return bool(selected['depth']) and all(x[k]==selected[k] for x in a+b for k in keys)


def one(row):
    order,bases,workers=actual(row['ticket'])
    result={}
    for tag,name in (('v4','sample'),('critic','modal')):
        roots=row['equal_work'][tag]
        p=roots[0]['opponent_samples']
        assert len(p)==1 and roots[1]['opponent_samples']==p
        p=p[0]
        hits=lambda ids:sum(p.get(i,'NONE')==order.get(i,'NONE') for i in ids)
        result[name]=dict(order=p,exact=p==order,workers=hits(workers),bases=hits(bases))
    work=equal_work(row)
    changed=all((row['live_change'],row['repeat'],row['same_issued'],row['recovered_parent'],work))
    return dict(ticket=row['ticket'],actual=order,workers=len(workers),bases=len(bases),same_work=work,
                stable_own_change=changed,predictions=result)


def main():
    src=(HERE/'results.json.gz').read_bytes()
    rows=[one(row) for row in json.loads(gzip.decompress(src))['rows']]
    pool=defaultdict(list)
    for row in rows:
        pool[row['ticket']['seed']].append(row)
    stats={}
    for label in ('sample','modal'):
        summary={k:sum(r['predictions'][label][k] for r in rows) for k in ('exact','workers','bases')}
        summary['map_equal_worker_accuracy']=mean(sum(a['predictions'][label]['workers'] for a in group)/sum(a['workers'] for a in group)
                                                  for group in pool.values())
        stats[label]=summary
    out=dict(source_sha256=sha(src),rows=rows,totals=stats,worker_actions=sum(r['workers'] for r in rows),
             base_actions=sum(r['bases'] for r in rows),same_work=sum(r['same_work'] for r in rows),
             stable_own_change=sum(r['stable_own_change'] for r in rows))
    (HERE/'predictions.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
