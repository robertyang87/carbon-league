"""Evaluate a frozen original value representation against signed whole-game rewards."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import sys
import types
import zipfile

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
ROOT=FAMILY.parents[1]
sys.path.insert(0,str(FAMILY/'v2_original_cash_paths'))
from study import PARENT, EXAMINER, select, sha

SAMPLE=tuple(range(25,266,20))
TEST_PANELS={'native_speed_dev','switch_margin_dev'}


def save(name,obj):
    raw=(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    if name.endswith('.gz'):
        raw=gzip.compress(raw,mtime=0)
    dest=HERE/name
    temp=dest.with_suffix('.partial')
    temp.write_bytes(raw);temp.replace(dest)
    return sha(raw)


def protocol(tickets):
    return dict(schema='midgame-terminal-critic/1',candidate=PARENT,original=EXAMINER,steps=SAMPLE,
                source=sha(Path(__file__).read_bytes()), test_panels=sorted(TEST_PANELS),
                games=[dict(map=t['game']['terrain'],seat=t['game']['seat'],panel=t['panel'],
                            receipt=t['game']['receipt_sha256'],archive=t['archive_sha256']) for t in tickets],
                target='Exact V2-perspective official terminal _terminal applied to uncontaminated signed result. '
                       'Each physical map equal total weight. V2-only public views and own previous commands.',
                designs={'native':'Original frozen output','phase':'Affine calibration of original value and turn',
                         'representation':'Add ridge residual from frozen value layer and two phase interactions',
                         'cash':'Current cash mapped by original terminal function'},
                ridge=.4)


def package():
    src=(FAMILY/'ladder/prediction_feedback/package/candidate.py').read_bytes()
    assert sha(src)==PARENT
    source=src.decode().replace('\r\n','\n')
    markers=(
        'z = _torch.cat([z.mean(2), z.amax(2), s], 1)',
        'z = _torch.cat([z.mean(0), z.amax(0), s], 1)',
    )
    for k in markers:
        assert source.count(k)==1
        source=source.replace(k,k+'\n                self._calibration_z = z.numpy().copy()')
    m=types.ModuleType('signed_parent_calibration')
    exec(compile(source,'signed-parent-critic-feature','exec'),m.__dict__)
    assert m._net() is not None and m._S_ABORT==.58 and m._TEMP==1
    return m


def collect(tickets,split,m):
    import numpy as np
    rows=[]
    for t in tickets:
        if (split=='held')!=(t['panel'] in TEST_PANELS):
            continue
        g=t['game'];seat=g['seat'];member=g['arm']+'/'
        archive=ROOT/t['archive']
        assert sha(archive.read_bytes())==t['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            data=z.read(member+'receipt.json')
            receipt=json.loads(data)
            assert sha(data)==g['receipt_sha256'] and receipt['counted'] and receipt['official_transitions']==299
            assert receipt['official_scores_by_seat']==g['official_scores_by_seat']
            actual=receipt['initialization']
            assert [actual[i]['response']['sha256'] for i in (seat,1-seat)]==[PARENT,EXAMINER]
            assert actual[1-seat]['response']['original']==dict(_S_ABORT=.58,_TEMP=1.,_SEED=None,_SEARCH_OK=True)
            tape=z.read(member+'turns.jsonl')
            assert sha(tape)==receipt['turns_sha256']
            pub=gzip.decompress(z.read(member+'observations.jsonl.gz'))
        tracker=m.BeliefTracker(m._Bel,seat)
        prev=None
        x=[];sc=[];digests=[]
        for step,(o_json,t_json) in enumerate(zip(pub.splitlines(),tape.splitlines(),strict=True)):
            data=json.loads(o_json); turn=json.loads(t_json); ob=data['observations'][seat]
            assert ob['player']==seat and ob['step']==data['step']==turn['step']==step and 'full_carbon' not in ob
            est,known=tracker.step(ob['players'],step,ob['carbon'])
            if step in SAMPLE:
                digest=sha((json.dumps(ob,separators=(',',':'))+'\n').encode())
                assert digest==data['public_observation_sha256'][seat]==turn['public_observation_sha256'][seat]
                sc.append(m._scalars(ob['players'],step,seat))
                x.append(m._quant(m.encode(ob['players'],step,seat,est,known,prev)))
                digests.append(digest)
            tracker.set_actions(ob['players'],turn['calls'][seat]['response']['action'])
            prev=ob['players']
            if step==SAMPLE[-1]:
                break
        assert len(x)==len(SAMPLE)
        ans=m._NET(np.asarray(x,dtype=np.float32),m._SID,np.asarray(sc,dtype=np.float32))[2]
        acts=m._NET._calibration_z
        assert len(ans)==len(acts)==len(SAMPLE) and np.isfinite(acts).all()
        scores=receipt['official_scores_by_seat']
        margin=scores[seat]-scores[1-seat]
        rows.extend(dict(terrain=g['terrain'], seed=g['seed'], seat=seat, step=step, public=d,panel=t['panel'],
                         native=float(v), gold=m._terminal(margin), final_margin=margin,
                         value_features=activations.tolist(), cash=m._terminal(float(values[0])*3000.))
                    for step,d,v,activations,values in zip(SAMPLE,digests,ans,acts,sc,strict=True))
        print(split,g['seed'],seat,flush=True)
    return rows


def xrow(r,tag):
    fraction=(r['step']-145)/120.
    v=r['native']
    cols=[1.,fraction,v,v*fraction]
    if tag=='representation':
        z=r['value_features']
        cols+=z+[z[-5]*fraction,z[-4]*fraction]
    return cols


def fit(rows,tag,lam):
    import numpy as np
    x=np.asarray([xrow(r,tag) for r in rows])
    y=np.asarray([r['gold']-r['native'] for r in rows])
    n=Counter(r['terrain'] for r in rows)
    w=np.asarray([1/n[r['terrain']] for r in rows]); w/=w.sum()
    rms=np.maximum(np.sqrt(w@np.square(x)),1e-6)
    z=x/rms
    mat=(z.T*w)@z+lam*np.eye(z.shape[1])
    return (np.linalg.solve(mat,(z.T*w)@y)/rms).tolist()


def measure(rows,heads):
    from statistics import mean
    groups=defaultdict(list)
    for r in rows:
        r['predictions']=dict(native=r['native'],cash=r['cash'])
        for label,head in heads.items():
            r['predictions'][label]=r['native']+sum(a*b for a,b in zip(xrow(r,label),head,strict=True))
        groups[r['terrain']].append(r)
    stats={}
    for terrain,group in groups.items():
        stats[terrain]=dict(rows=len(group), metrics={})
        for label in group[0]['predictions']:
            err=lambda rows: mean(abs(r['predictions'][label]-r['gold']) for r in rows)
            x=[(r['predictions'][label],r['gold']) for r in group]
            stats[terrain]['metrics'][label]=dict(mae=err(group),
                bands=[err([r for r in group if a<=r['step']<b]) for a,b in ((0,100),(100,200),(200,300))],
                correct_sign=sum(int((a>0)==(b>0)) for a,b in x), n=len(x))
    return stats


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('phase',choices=('register','fit','held'))
    cli.add_argument('--cpu',type=int,required=True)
    args=cli.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[v]='1'
    os.environ['CARBON_DEBUG']='1'
    tickets=select()
    lock=protocol(tickets)
    dest=HERE/'protocol.json'
    if args.phase=='register':
        assert not dest.exists() or json.loads(dest.read_bytes())==json.loads(json.dumps(lock))
        print(save('protocol.json',lock))
    else:
        assert json.loads(dest.read_bytes())==json.loads(json.dumps(lock))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        m=package()
        torch.set_num_threads(1)
        weights=HERE/'weights.json'
        if args.phase=='held':
            fitted=json.loads(weights.read_bytes())
            assert fitted['protocol']==sha(dest.read_bytes())
            assert fitted['train_rows']==sha((HERE/'fit_rows.json.gz').read_bytes())
            heads=fitted['heads']
        rows=collect(tickets,args.phase,m)
        if args.phase=='fit':
            heads={tag:fit(rows,tag,lock['ridge']) for tag in ('phase','representation')}
        else:
            assert set(fitted['maps']).isdisjoint({r['terrain'] for r in rows})
        metrics=measure(rows,heads)
        data_sha=save(args.phase+'_rows.json.gz',rows)
        if args.phase=='fit':
            save('weights.json',dict(protocol=sha(dest.read_bytes()),train_rows=data_sha,maps=sorted(metrics),heads=heads))
        save(args.phase+'.json',dict(weights=sha(weights.read_bytes()),rows=len(rows),data=data_sha,maps=metrics))
