"""Compare native V2, opportunistic extension, and up-front budget reservation."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import types

from build import FAMILY, HERE, OPTIONAL, OPTIONAL_SHA, V2, V2_SHA, build, sha

ROOT=FAMILY.parents[1]


def encode(x):
    return (json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()


def register():
    p=FAMILY/'native_terminal_extension/replication/protocol.json'
    raw=p.read_bytes()
    selected=json.loads(raw)
    assert tuple(selected['packages'])==(V2_SHA,OPTIONAL_SHA)
    return dict(schema='terminal-budget-reserve/1', packages=dict(v2=V2_SHA,optional=OPTIONAL_SHA,reserve=build()),
                source=sha(raw), original=selected['original'], tickets=selected['tickets'],
                design='Reuse the same four maps, both seats, t287/t290. Rotate three arms and reverse the order for second calls. '
                       'Track completion and cost; budget reservation is allowed to change width.')


def run(cpu, p):
    os.sched_setaffinity(0,{cpu})
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0,str(FAMILY/'root_rotation_policy/native'))
    from probe import each_history
    sys.path.insert(0,str(FAMILY/'opponent_policy/search_probe'))
    from full_search_check import root_call
    paths=dict(v2=V2,optional=OPTIONAL,reserve=HERE/'candidate.py')
    models={}
    for tag,path in paths.items():
        raw=path.read_bytes()
        assert sha(raw)==p['packages'][tag]
        module=types.ModuleType(tag)
        exec(compile(raw,str(path),'exec'),module.__dict__)
        assert module._net() is not None and module._TEMP==1 and module._S_ABORT==.58
        models[tag]=module
    torch.set_num_threads(1)
    cfgpath=ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json'
    cfg={k:v['default'] for k,v in json.loads(cfgpath.read_bytes())['configuration'].items()
         if isinstance(v,dict) and 'default' in v}
    cfg['episodeSteps']=300
    opening=models['v2']._S_COST[:]
    kept=[]
    for first,last in zip(p['tickets'][::2],p['tickets'][1::2],strict=True):
        assert (first['seed'],first['seat'])==(last['seed'],last['seat'])
        seat=first['seat']
        target={r['step']:r for r in (first,last)}
        tr,prior=models['v2'].BeliefTracker(models['v2']._Bel,seat),None
        for obs,issued,digest in each_history(first):
            step=obs['step']
            if step+1 in target:
                incoming=root_call(models['v2'],tr,prior,obs,cfg,target[step+1]['rng']^0x12F191AC,opening)['ending_costs']
            if step in target:
                keys=list(paths)
                i=len(kept)%3
                order=keys[i:]+keys[:i]
                reads={k:[] for k in keys}
                for tag in order+list(reversed(order)):
                    m=models[tag]
                    m._S_STATS.pop('terminal_extension',None)
                    r=root_call(m,tr,prior,obs,cfg,target[step]['rng'],incoming)
                    r['extension']=m._S_STATS.get('terminal_extension')
                    if r['extension'] is not None:
                        assert r['depth']==299-step
                    reads[tag].append(r)
                v2,reserve=reads['v2'],reads['reserve']
                changed=all(v[0]['issued']==v[1]['issued'] for v in (v2,reserve)) and all(r['extension'] for r in reserve)
                changed=bool(changed and v2[0]['issued']!=reserve[0]['issued'])
                kept.append(dict(ticket=target[step],public_sha256=digest,cost=incoming,order=order,replies=reads,
                                 stable_completed_change=changed))
                print(target[step]['seed'],seat,step,
                      {tag:[sum(i['extension'] is not None for i in arm),[i['candidates'] for i in arm]]
                       for tag,arm in reads.items()},changed,flush=True)
            tr.step(obs['players'],step,obs['carbon'])
            tr.set_actions(obs['players'],issued)
            prior=obs['players']
            if step==last['step']:
                break
    assert len(kept)==16
    (HERE/'results.json.gz').write_bytes(gzip.compress(encode(dict(protocol=sha(encode(p)),rows=kept)),mtime=0))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    a=cli.parse_args()
    p=register()
    raw=encode(p)
    path=HERE/'protocol.json'
    if a.register:
        assert not path.exists() or path.read_bytes()==raw
        path.write_bytes(raw)
        print(sha(raw),flush=True)
    else:
        assert path.read_bytes()==raw
        run(a.cpu,p)
