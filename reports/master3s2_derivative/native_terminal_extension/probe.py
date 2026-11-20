"""Measure repeatable optional completion and returned actions under the real V2 clock."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import types
import zipfile

from build import FAMILY, HERE, PARENT, PARENT_SHA, build, sha

ROOT=FAMILY.parents[1]
OPPONENT='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
STEPS=(287,290)


def encode(obj):
    return (json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()


def registration():
    tickets=[]
    src=FAMILY/'native_speed_dev/hosted'
    for file in sorted(src.glob('*.zip')):
        raw=file.read_bytes()
        with zipfile.ZipFile(file) as z:
            reg=z.read('parent/registration.json')
        info=json.loads(reg)
        seed, seat=info['seed'],info['candidate_seat']
        assert seed in range(621782460,621782464)
        assert [info['files'][k]['sha256'] for k in ('candidate','opponent')]==[PARENT_SHA,OPPONENT]
        for step in STEPS:
            tickets.append(dict(seed=seed,seat=seat,terrain=info['physical_map_sha256'],step=step,
                                archive=str(file.relative_to(FAMILY)),registration_sha256=sha(reg),
                                archive_sha256=sha(raw),arm='parent',rng=int(sha(f'terminal-extension/1/{seed}/{seat}/{step}'.encode())[:8],16)))
    assert len(tickets)==len({(r['seed'],r['seat'],r['step']) for r in tickets})==16
    return dict(schema='native-terminal-extension/1', packages=[PARENT_SHA,build()], original=OPPONENT,
                plan='Four completed development maps both seats and t287/t290. Alternate double public calls with V2. '
                     'Only credit an observed action change if both reproduce, match root menu and root enemy command, '
                     'and the candidate completes optional settlement in both real clocks.', tickets=tickets)


def run(cpu, p, out=None):
    os.sched_setaffinity(0,{cpu})
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0,str(FAMILY/'root_rotation_policy/native'))
    from probe import each_history
    sys.path.insert(0,str(FAMILY/'opponent_policy/search_probe'))
    from full_search_check import root_call
    models={}
    for label,path,digest in zip(('parent','extension'),(PARENT,HERE/'candidate.py'),p['packages'],strict=True):
        raw=path.read_bytes()
        assert sha(raw)==digest
        m=types.ModuleType(label)
        exec(compile(raw,str(path),'exec'),m.__dict__)
        assert m._net() is not None and m._S_ABORT==.58 and m._TEMP==1 and m._SEARCH_OK
        models[label]=m
    torch.set_num_threads(1)
    rules=ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json'
    cfg={k:v['default'] for k,v in json.loads(rules.read_bytes())['configuration'].items()
         if isinstance(v,dict) and 'default' in v}
    cfg['episodeSteps']=300
    parent=models['parent']
    opening=list(parent._S_COST)
    outcomes=[]
    for first,last in zip(p['tickets'][::2],p['tickets'][1::2],strict=True):
        assert first['seed']==last['seed'] and first['seat']==last['seat']
        targets={first['step']: first,last['step']:last}
        tracker,previous=parent.BeliefTracker(parent._Bel,first['seat']),None
        for obs,issued,digest in each_history(first):
            step=obs['step']
            if step+1 in targets:
                seed=targets[step+1]['rng']^0x12F191AC
                incoming=root_call(parent,tracker,previous,obs,cfg,seed,opening)['ending_costs']
            if step in targets:
                ticket=targets[step]
                order=['parent','extension','extension','parent']
                if len(outcomes)%2:
                    order=['extension','parent','parent','extension']
                replies={name:[] for name in models}
                for name in order:
                    m=models[name]
                    m._S_STATS.pop('terminal_extension',None)
                    result=root_call(m,tracker,previous,obs,cfg,ticket['rng'],incoming)
                    result['extension']=m._S_STATS.get('terminal_extension')
                    replies[name].append(result)
                a,b=replies['parent'],replies['extension']
                common=('menu','act0','opponent_samples')
                matched=all(q[k]==a[0][k] for q in a+b for k in common) and a[0]['depth']==a[1]['depth']==8
                complete=all(result['extension'] is not None and result['depth']==299-step for result in b)
                if complete:
                    assert all(r['extension']['length']==b[0]['extension']['length']==291-step for r in b)
                stable=all(v[0]['issued']==v[1]['issued'] for v in replies.values())
                changed=matched and complete and stable and a[0]['issued']!=b[0]['issued']
                outcomes.append(dict(ticket=ticket,cost=incoming,public_sha256=digest,matched=matched,
                                     complete=complete,changed=changed,replies=replies))
                print(ticket['seed'], ticket['seat'],step,matched,complete,changed,flush=True)
            tracker.step(obs['players'],step,obs['carbon'])
            tracker.set_actions(obs['players'],issued)
            previous=obs['players']
            if step==last['step']:
                break
    assert len(outcomes)==16
    (Path(out if out is not None else HERE)/'results.json.gz').write_bytes(gzip.compress(encode(dict(protocol=sha(encode(p)),rows=outcomes)),mtime=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--register',action='store_true')
    ap.add_argument('--cpu',type=int)
    a=ap.parse_args()
    p=registration()
    dest=HERE/'protocol.json'
    raw=encode(p)
    if a.register:
        assert not dest.exists() or dest.read_bytes()==raw
        dest.write_bytes(raw)
        print(sha(raw),flush=True)
    else:
        assert dest.read_bytes()==raw
        run(a.cpu,p)
