"""Managed conditional responses to the two observed late equal-wage recruitment choices."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'native'
T=HERE.parent/'tree_response_training'
HEAD=T/'response_head'
ROOT=HERE.parents[3]
HELPER=HERE.parents[1]/'root_recovery_branches'
PARENT=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
sys.path.insert(0,str(HELPER))
import live
ROOT_SOURCE='70f1f3d7c7651f7b2a67508e14481f2e55a1362ea14ac659b58c1d15f7d91b4a'
ENV={**os.environ,**{s:'1' for s in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def check(ok,message):
    if not ok:raise ValueError(message)


def freeze(p,data):
    if p.exists():check(read(p)==data,f'frozen evidence differs: {p}')
    else:live.save(p,data)


def mine(obs):
    return set(obs['players'][obs['player']][2])


def root_receipt(raw,row):
    from fastsim import FastCarbon
    seat=row['seat'];t=row['t'];base=row['worker'];cfg=raw['game_configuration']
    output={}
    for arm,choice in [('parent','RECPLANTER'),('alternative','RECCOLLECTOR')]:
        off=live.begin(raw);sim=FastCarbon(cfg).reset(seed=row['seed'])
        for joint in raw['actions'][:t]:off.step(joint);sim.step(*joint)
        before=live.public(off,seat);uid_pool=mine(before);basecell=before['players'][seat][1][base]
        check(sim.obs(seat)['players']==before['players'],'two engines disagree before intervention')
        cmd=[dict(a) for a in raw['actions'][t]]
        check(cmd[seat][base]=='RECPLANTER','original action changed')
        cmd[seat][base]=choice
        off.step(cmd);sim.step(*cmd);after=live.public(off,seat)
        check(after['players']==sim.obs(seat)['players'],'two engines disagree after intervention')
        workers=after['players'][seat][2]
        new=[uid for uid in set(workers)-uid_pool if workers[uid][0]==basecell and workers[uid][2]==choice[3:]]
        check(len(new)==1,'role-specific new employee must actually arrive alive at its own base')
        num=sum(1 for side in before['players'] for w in side[2].values() if w[2]==choice[3:])
        cost=cfg['rec'+choice[3:].title()+'Cost']+num*cfg['rec'+choice[3:].title()+'IncreaseCost']
        output[arm]={'worker':new[0],'role':choice[3:],'hire_cost':cost,'base_cell':basecell,'joint_action':cmd}
    check(output['parent']['hire_cost']==output['alternative']['hire_cost']==row['price'],'unequal real payroll')
    return output


def prepare():
    OUT.mkdir(exist_ok=True)
    package=read(HERE/'package_replay.json');manifest=read(HERE/'manifest.json');screen=read(HERE/'screen.json')
    check(package['public_mechanism_both_seats'] and package['candidate_sha256']==sha(HERE/'agent.py')==manifest['policy_sha256'],'no real candidate')
    check(sha(live.SOURCE)==live.MASTER_SHA==hashlib.sha256(subprocess.check_output(['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=ROOT)).hexdigest() and sha(PARENT)==manifest['parent_sha256']
          and sha(Path(live.__file__))==ROOT_SOURCE,'wrong parent, engine or examiner')
    candidates=[g for g in screen['games'] if g['proposal_turns']]
    check(len(candidates)==2 and {g['seat'] for g in candidates}=={0,1} and len({g['terrain'] for g in candidates})==2,
          'require exactly the two genuinely discovered seats')
    cases=[]
    for k,row in enumerate(package['games']):
        known=next(r for r in candidates if r['seed']==row['seed'] and r['seat']==row['seat'])
        e,=row['valid_prefix_edits'];check(e['base'] and known['raw_sha256']==row['raw_sha256']==sha(ROOT/known['path']),'wrong source')
        r={'seed':row['seed'],'seat':row['seat'],'source':known['path'],'source_sha256':row['raw_sha256'],'t':e['step'],
           'worker':e['base'],'old':'RECPLANTER','alternative':'RECCOLLECTOR','price':e['price'],'tree':'','stream_seed':992831+k}
        raw=read(ROOT/known['path']);r['root_check']=root_receipt(raw,r);cases.append(r)
    reg={'schema':'two-late-hire-native-responses-v1','parent_sha256':sha(PARENT),'candidate_sha256':sha(HERE/'agent.py'),
         'examiner_sha256':live.MASTER_SHA,'mechanism_sha256':sha(HERE/'package_replay.json'),'screen_sha256':sha(HERE/'screen.json'),
         'selection_sha256':sha(HERE/'protocol.json'),'shared_runner_sha256':ROOT_SOURCE,'cases':cases,'cpu_allocation':[0,1],
         'num_pairs':2,'num_arms':4,'post_root':'both arms continue with original frozen v1 and reactivated native Master3s2; once-per-game candidate has expended its single modification on the intervention arm; exact common recorded Master root, independent restored streams start from a common saved state per pair. Native wall-clock search can diverge.',
         'gate':'Request full new development maps only when both conditional net terminal own-minus-rival margin differences improve and each focal alternative collector makes deposits exceeding the wage. A single trajectory per branch on each of two selected training maps cannot establish overall strength. Any other outcome retires this candidate from whole-game testing.',
         'priority':"complete intake_08 and the earlier eight endgame last-move calls. A qualified tree-head's registered new-map 8-game study runs before this experiment."}
    freeze(OUT/'protocol.json',reg)
    for r in cases:
        wd=case_dir(r);wd.mkdir(exist_ok=True)
        record={k:r[k] for k in ('seed','seat','source','source_sha256')}
        seed=r['stream_seed'];rng=np.random.RandomState(seed).get_state()
        q=dict(seed=r['seed'],seat=r['seat'],family='late_workforce_hire',root_t=r['t'],source_dir=str(Path(r['source']).parent),
               record=record,root={k:r[k] for k in ('worker','old','alternative','t','tree')},study_protocol_sha256=sha(OUT/'protocol.json'))
        freeze(wd/'live_protocol.json',q)
        shared=wd/'live_shared_init.json'
        if not shared.is_file():
            m=live.module(live.SOURCE,'late_hire_warm_examiner')
            check(m._SEED is None and m._TEMP==1 and m._SEARCH_OK and m._RX,'wrong examiner settings')
            freeze(shared,dict(mt_algorithm=rng[0],mt_keys=rng[1].tolist(),mt_index=rng[2],has_normal=rng[3],
                               stored_normal=rng[4],source_sha256=live.MASTER_SHA,warm_cost=list(m._S_COST)))
        check(read(shared)['mt_keys']==rng[1].tolist(),'native continuation stream changed')
    return reg


def case_dir(r):
    return OUT/f"case_{r['seed']}_{r['seat']}"


def native_module():
    path=Path(live.__file__);check(sha(path)==ROOT_SOURCE,'native driver changed')
    source=path.read_text()
    old='r["worker"] not in own_obs["players"][seat][2]';new='r["worker"] not in own_obs["players"][seat][1]'
    old2='elif r["alternative"] in ("UP","RIGHT","DOWN","LEFT"):z[r["worker"]]=r["alternative"]'
    new2='elif r["old"]=="RECPLANTER" and r["alternative"]=="RECCOLLECTOR":z[r["worker"]]=r["alternative"]'
    check(source.count(old)==source.count(old2)==1,'unknown live runner format')
    mod=types.ModuleType('full_native_hire_recovery');mod.__file__=str(path)
    exec(compile(source.replace(old,new).replace(old2,new2),str(path),'exec'),mod.__dict__)
    mod.PACK=PARENT;mod.PACK_HASH=sha(PARENT)
    return mod


def arm(index,name,cpu):
    check(os.sched_getaffinity(0)=={cpu},'wrong native affinity')
    reg=read(OUT/'protocol.json');r=reg['cases'][index];wd=case_dir(r)
    grant=read(OUT/'granted.json')
    check(sha(OUT/'protocol.json')==read(wd/'live_protocol.json')['study_protocol_sha256']==grant['protocol_sha256']
          and grant['native_runner_sha256']==sha(Path(__file__)) and sha(PARENT)==reg['parent_sha256'] and sha(HERE/'agent.py')==reg['candidate_sha256']
          and grant['cpus']==[0,1] and sha(live.SOURCE)==reg['examiner_sha256'], 'native study, executable, policy or exam court differs')
    with (wd/f'.{name}.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        native_module().run(name,wd,read(wd/'live_protocol.json'),resume=True)


if __name__=='__main__':
    args=sys.argv[1:]
    if args==['prepare']:prepare()
    elif len(args)==4 and args[0]=='worker':arm(int(args[1]),args[2],int(args[3]))
    elif args==['schedule']:
        from schedule import run
        run()
    else:raise SystemExit('prepare | schedule | worker INDEX parent/alternative CPU')
