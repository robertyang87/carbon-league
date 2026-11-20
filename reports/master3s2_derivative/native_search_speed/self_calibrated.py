"""Measure native work using each package's own measured cost after three public requests."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import types

from probe import HERE, FAMILY, baseline, check, load

WARM = 3


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu',type=int,required=True)
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[var]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0,str(FAMILY/'root_rotation_policy/native'))
    each_history=load('root_probe_helpers', FAMILY/'root_rotation_policy/native/probe.py').each_history
    sys.path.insert(0,str(FAMILY/'opponent_policy/search_probe'))
    from full_search_check import root_call
    raw=(HERE/'protocol.json').read_bytes()
    reg=json.loads(raw)
    variants={}
    seeds={}
    for tag,path,expected in zip(('parent','faster'),(baseline.SOURCE,HERE/'candidate.py'),reg['packages'],strict=True):
        packed=path.read_bytes()
        assert check.sha(packed)==expected
        module=types.ModuleType(tag)
        exec(compile(packed,tag,'exec'),module.__dict__)
        assert module._SEARCH_OK and module._net() is not None and module._S_ABORT==.58 and module._TEMP==1
        variants[tag]=module
        seeds[tag]=list(module._S_COST)
    config=json.loads((check.ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())
    cfg={k:v['default'] for k,v in config['configuration'].items() if isinstance(v,dict) and 'default' in v}|{'episodeSteps':300}
    rows=[]
    for first,second in zip(reg['tickets'][::2],reg['tickets'][1::2],strict=True):
        assert first['archive']==second['archive']
        targets={r['step']:r for r in (first,second)}
        costs={n:list(v) for n,v in seeds.items()}
        tracker=variants['parent'].BeliefTracker(variants['parent']._Bel,first['seat'])
        prev=None
        for obs,issued,checksum in each_history(first):
            t=obs['step']
            next_target=next((a for a in targets if a-WARM<=t<a),None)
            order=('parent','faster') if (first['seed']+first['seat']+t)%2 else ('faster','parent')
            if next_target is not None:
                rng=targets[next_target]['rng']^(0x12F191AC+t)
                for tag in order:
                    costs[tag]=root_call(variants[tag],tracker,prev,obs,cfg,rng,costs[tag])['ending_costs']
            if t in targets:
                ticket=targets[t]
                ans={n:[] for n in variants}
                for tag in (*order,*reversed(order)):
                    ans[tag].append(root_call(variants[tag],tracker,prev,obs,cfg,ticket['rng'],costs[tag]))
                stable={n:x[0]['issued']==x[1]['issued'] for n,x in ans.items()}
                row=dict(ticket=ticket,public_sha256=checksum,cost=costs.copy(),answers=ans,stable=stable,
                         stable_change=all(stable.values()) and ans['parent'][0]['issued']!=ans['faster'][0]['issued'])
                rows.append(row)
                print(ticket['seed'],ticket['seat'],t,[ans[n][0]['candidates'] for n in variants],row['stable_change'],flush=True)
                costs={n:list(v) for n,v in seeds.items()}
            tracker.step(obs['players'],t,obs['carbon'])
            tracker.set_actions(obs['players'],issued)
            prev=obs['players']
            if t==max(targets):
                break
    assert len(rows)==len(reg['tickets'])
    out=dict(source_protocol_sha256=check.sha(raw),warm_calls=WARM,packages=reg['packages'],core=args.cpu,rows=rows)
    (HERE/'self_calibrated.json.gz').write_bytes(gzip.compress(check.encode(out),mtime=0))


if __name__=='__main__':
    main()
