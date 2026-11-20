"""Paired full native roots with two independently sourced original enemy commands."""
import argparse
import copy
import gzip
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import zipfile

from build import FAMILY, HERE, PARENT, PARENT_SHA, build, sha

ROOT=FAMILY.parents[1]
ORIGINAL=ROOT/'sources/master3s2/master3s2.py'
ORIGINAL_SHA='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'


def encode(x):
    return (json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()


def registration():
    cases=[]
    for location in ('native_terminal_extension','native_terminal_extension/replication'):
        d=FAMILY/location
        prior=(d/'protocol.json').read_bytes()
        data=json.loads(gzip.decompress((d/'results.json.gz').read_bytes()))
        assert data['protocol']==sha(prior)
        for row in data['rows']:
            if row['ticket']['step']!=290:
                continue
            t=row['ticket']
            ticket=t|dict(step=298,rng=int(sha(f"final-response/1/{t['seed']}/{t['seat']}".encode())[:8],16),
                           history_source_package=PARENT_SHA,initial_cost=row['cost'])
            ticket['rival_rng']=ticket['rng']^0x7D4A1F6B
            with zipfile.ZipFile(FAMILY/ticket['archive']) as z:
                q=json.loads(gzip.decompress(z.read(t['arm']+'/observations.jsonl.gz')).splitlines()[298])
            ticket['own_public_sha256']=q['public_observation_sha256'][t['seat']]
            cases.append(ticket)
    assert len(cases)==16 and len({t['terrain'] for t in cases})==8
    return dict(schema='last-turn-complete-response/1',packages=[PARENT_SHA,build()],examiner=ORIGINAL_SHA,
                design='All last turns of both previously fixed four-map panels. Read actual historical instructions only after '
                       'both public policies have issued. Additionally compare each selected order against one fresh original root.',
                tickets=cases)


def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def historical_opponent(t):
    archive=FAMILY/t['archive']
    assert sha(archive.read_bytes())==t['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        path=t['arm']+'/'
        receipt=json.loads(z.read(path+'receipt.json'))
        tape=z.read(path+'turns.jsonl')
    assert sha(tape)==receipt['turns_sha256']
    r=json.loads(tape.splitlines()[t['step']])
    assert r['step']==t['step'] and r['public_observation_sha256'][t['seat']]==t['own_public_sha256']
    return r['calls'][1-t['seat']]['response']['action'], receipt['official_scores_by_seat']


def run(p,cpu):
    os.sched_setaffinity(0,{cpu})
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path[:0]=[str(FAMILY/'opponent_policy/search_probe'),
                 str(ROOT/'code/env_src/carbon_challenge-master'),str(FAMILY/'symmetry_value/terminal_pairs')]
    import full_search_check
    import branches

    namespace={}
    code=inspect.getsource(full_search_check.root_call)
    assert code.count('len(enemy_samples) <= 1')==1
    exec(code.replace('len(enemy_samples) <= 1','len(enemy_samples) <= 8'),vars(full_search_check),namespace)
    root_call=namespace['root_call']
    wrapped=load('native_final_reply_wrapper',FAMILY/'single_worker_priority/response/run.py')
    runner=wrapped.load_runner()
    runner.PLAN=p
    freight=branches.freight
    live=freight.load('native_terminal_last_turn',freight.STUDY/'cargo_tree_action_learning_dev/ladder_live/live.py')
    paths=(PARENT,HERE/'candidate.py',ORIGINAL)
    assert [sha(f.read_bytes()) for f in paths]==[*p['packages'],p['examiner']]
    parent,candidate,rival=[load(name,path) for name,path in zip(('before','multi','original'),paths,strict=True)]
    torch.set_num_threads(1)
    assert all(m._net() is not None and m._TEMP==1 and m._S_ABORT==.58 and m._SEED is None for m in (parent,candidate,rival))
    out=[]
    for i,t in enumerate(p['tickets']):
        seat=t['seat']
        models=[parent,rival] if seat==0 else [rival,parent]
        _,cfg,ref,sim,obs,prefix=runner.replay_root(t,models,freight,live)
        assert obs[seat]['step']==298
        st=copy.deepcopy(parent._ST)
        readings={'parent':[],'candidate':[]}
        order=('parent','candidate','candidate','parent') if i%2==0 else ('candidate','parent','parent','candidate')
        for tag in order:
            m=parent if tag=='parent' else candidate
            r=root_call(m,st['tr'],st['prev'],obs[seat],cfg,t['rng'],t['initial_cost'])
            readings[tag].append(r)
        historical,scores=historical_opponent(t)
        fresh,clock=freight.measured(rival,obs[1-seat],cfg)
        actual={'historical':historical,'fresh':fresh}
        snapshot=freight.official_state.take(ref,[o['remainingOverageTime'] for o in obs])
        end={}
        for tag,command in actual.items():
            results={}
            for arm in readings:
                freight.official_state.restore(ref,snapshot)
                own=readings[arm][0]['issued']
                joint=[own,command] if seat==0 else [command,own]
                ref.step(joint,[])
                assert ref.done and [s.status for s in ref.states]==['DONE','DONE']
                official=[float(s.reward) for s in ref.states]
                results[arm]=dict(scores=official,margin=official[seat]-official[1-seat],
                                  own_action=own)
            end[tag]=results
        a,b=readings['parent'][0],readings['candidate'][0]
        stable=all(r[0]['issued']==r[1]['issued'] for r in readings.values())
        matched=a['menu']==b['menu'] and a['opponent_samples']==b['opponent_samples'][:1]
        row=dict(ticket=t, root_public=prefix['public_sha256_by_seat'], parent_archived_terminal_scores=scores,
                 enemy=actual, original_fresh_clock=clock, roots=readings, stable=stable,matched=matched,terminal=end)
        out.append(row)
        print(t['seed'],seat,len(set(encode(k) for k in b['opponent_samples'])),a['issued']!=b['issued'],
              [x['candidate']['margin']-x['parent']['margin'] for x in end.values()],flush=True)
    assert len(out)==16
    (HERE/'results.json.gz').write_bytes(gzip.compress(encode(dict(protocol=sha(encode(p)),roots=out)),mtime=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--register',action='store_true')
    ap.add_argument('--cpu',type=int)
    a=ap.parse_args()
    p=registration()
    raw=encode(p)
    dest=HERE/'protocol.json'
    if a.register:
        assert not dest.exists() or dest.read_bytes()==raw
        dest.write_bytes(raw)
        print(sha(raw),flush=True)
    else:
        assert dest.read_bytes()==raw
        run(p,a.cpu)
