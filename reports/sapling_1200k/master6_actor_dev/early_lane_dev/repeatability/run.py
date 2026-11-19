"""Bounded, resumable remeasurement of the two explicitly outcome-selected native blockade roots."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

D=Path(__file__).resolve().parent
A=D.parents[1]
BASE=A.parent
ROOT=BASE.parents[1]
T=A/'tree_response_training';OLD=D.parent/'native_response'
sys.path.insert(0,str(BASE/'root_recovery_branches'))
import live
up=live.module(OLD/'run.py','immutable_early_lane_source')
reg='de7e4eac05b364fa5c10b74a0686e178a5495485e37158b992d1993a664a1bbd'
sha=up.sha
require=up.require
seal=up.seal
arms=('parent','alternative')


def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)


def bind():
    plan=read(D/'protocol.json')
    require(sha(D/'protocol.json')==reg and sha(OLD/'run.py')==plan['entry_sha256']
            and sha(OLD/'verify.py')==plan['verifier_sha256'] and sha(OLD/'protocol.json')==plan['old_plan_sha256']
            and sha(OLD/'result.json')==plan['old_result_sha256'],'unrecognized replication or prior study')
    old=up.identity();previous=read(OLD/'result.json')
    ranked=sorted(zip(previous['by_case'],old['roots']),key=lambda item:item[0]['delta_margin'])
    for (m,x),r in zip([ranked[-1],ranked[0]],plan['cases']):
        require([x[k] for k in ('seed','seat','t')]==[r[k] for k in ('seed','seat','t')] and m['delta_margin']==r['old_effect']
                and r['old_stream']==x['new_stream_seed'] and all(sha(OLD/p)==h for p,h in r['old_files'].items()),'published roots differ')
    require(sha(live.SOURCE)==live.MASTER_SHA==plan['examiner_sha256']==hashlib.sha256(subprocess.check_output(
            ['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=ROOT)).hexdigest()
            and sha(up.PARENT)==plan['parent_sha256'] and sha(Path(live.__file__))==plan['responder_sha256'],'invalid actual examiner, v1 or search runner')
    live.PACK=up.PARENT;live.PACK_HASH=plan['parent_sha256']
    return plan


def prepare():
    plan=bind();cases=[]
    for r in plan['cases']:
        prev=OLD/r['old_case'];wd=D/r['old_case'];wd.mkdir(exist_ok=True)
        common=read(prev/'live_shared_init.json');old=read(prev/'live_protocol.json')
        seal(wd/'live_protocol.json',{**old,'family':'early_lane_exposed_extremes_repeat','replication_study_sha256':reg,
             'signed_historical_root_sha256':sha(prev/'live_protocol.json'),'fresh_rng_seed':r['new_stream']})
        state=np.random.RandomState(r['new_stream']).get_state()
        bank={**common,'mt_algorithm':state[0],'mt_keys':state[1].tolist(),'mt_index':state[2],
              'has_normal':state[3],'stored_normal':state[4]}
        require(bank['mt_keys']!=common['mt_keys'] and bank['warm_cost']==common['warm_cost'] and common['warm_cost'],'stream reuse or different warm-state')
        seal(wd/'live_shared_init.json',bank)
        cases.append({'case':wd.name,'case_sha256':sha(wd/'live_protocol.json'),'stream_sha256':sha(wd/'live_shared_init.json')})
    seal(D/'execution.json',{'protocol_sha256':reg,'driver_sha256':sha(Path(__file__)),'official_replay_sha256':sha(OLD/'verify.py'),'cases':cases})
    return plan


def worker(i,name):
    plan=bind();r=plan['cases'][i];wd=D/r['old_case'];job=read(D/'execution.json')['cases'][i]
    require(name in arms and os.sched_getaffinity(0)=={r['new_'+name+'_cpu']} and read(D/'execution.json')['driver_sha256']==sha(Path(__file__))
            and sha(wd/'live_protocol.json')==job['case_sha256'] and sha(wd/'live_shared_init.json')==job['stream_sha256'],'worker or inputs changed')
    with (wd/f'.{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tape=wd/f'live_{name}_actions.json.gz'
        if tape.exists() and not (wd/f'live_{name}.json').exists():raise ValueError('preserve existing full raw tape until it can be reconciled')
        live.run(name,wd,read(wd/'live_protocol.json'),resume=True)


def certify(plan):
    verifier=live.module(OLD/'verify.py','bound_independent_early_lane_verifier')
    items=[]
    for r in plan['cases']:
        old=OLD/r['old_case'];wd=D/r['old_case']
        first=verifier.verify_case(old);second=verifier.verify_case(wd)
        for name in arms:
            a=read(old/f'live_{name}.json');b=read(wd/f'live_{name}.json')
            require(a['root_actions']==b['root_actions'] and a['root_opponent_observation_sha256']==b['root_opponent_observation_sha256']
                    and a['original_trajectory_sha256']==b['original_trajectory_sha256'] and a['shared_init_sha256']!=b['shared_init_sha256'],
                    'root, common simultaneous command or newer random init differs')
        require(first['metrics']['delta_margin']==r['old_effect'] and first['official_map_sha256']==second['official_map_sha256'],'source mismatch')
        arms_new={name:{'physical_scores':second['arms'][name]['score_by_physical_seat'],'wdl':second['arms'][name]['wdl'],
                        'margin':second['arms'][name]['margin'],'after_root_original_search_calls':second['arms'][name]['search_calls'],
                        'action_tape_sha256':second['arms'][name]['action_sha256']} for name in arms}
        items.append(dict(case=r['old_case'],seed=r['seed'],seat=r['seat'],t=r['t'],old_rng=r['old_stream'],new_rng=r['new_stream'],
                          old_metrics=first['metrics'],new_metrics=second['metrics'],old_root_sha256=sha(old/'verified.json.gz'),
                          new_root_sha256=sha(wd/'verified.json.gz'),new_shared_sha256=sha(wd/'live_shared_init.json'),
                          first_different_opponent_order_old=first['opponent_orders_different_at'][0] if first['opponent_orders_different_at'] else None,
                          first_different_opponent_order_new=second['opponent_orders_different_at'][0] if second['opponent_orders_different_at'] else None,
                          new_arms=arms_new,sign_flip=bool(first['metrics']['delta_margin']*second['metrics']['delta_margin']<0),
                          elapsed_pair_seconds=read(wd/'finished.json')['wall_seconds']))
    seal(D/'results.json',dict(protocol_sha256=reg,execution_sha256=sha(D/'execution.json'),cases=items,new_arms=4,new_map_count=0,
                               original_after_root_search_calls=sum(a['after_root_original_search_calls'] for z in items for a in z['new_arms'].values())))
    return items


def schedule():
    with (D/'.schedule.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare()
        if (D/'results.json').exists():certify(plan);return
        with (T/'.continuation.lock').open('a') as court:
            fcntl.flock(court,fcntl.LOCK_EX)
            if not (D/'court.json').exists():seal(D/'court.json',dict(granted=up.utc(),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),cores=[3,4],host_state=up.read_env(),execution_sha256=sha(D/'execution.json')))
            for i,r in enumerate(plan['cases']):
                wd=D/r['old_case']
                if (wd/'finished.json').exists():continue
                if not (wd/'started.json').exists():
                    seal(wd/'started.json',{'utc':up.utc(),'monotonic':time.monotonic(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
                start=read(wd/'started.json')
                require(start['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'machine epoch changed')
                def dispatch(name):
                    subprocess.run(['taskset','-c',str(r['new_'+name+'_cpu']),sys.executable,'-u',str(Path(__file__)),'worker',str(i),name],
                                   env=up.ENV,check=True)
                with ThreadPoolExecutor(2) as pool:list(pool.map(dispatch,arms))
                seal(wd/'finished.json',{'utc':up.utc(),'wall_seconds':time.monotonic()-start['monotonic'],
                                         'arm_receipts':{n:sha(wd/f'live_{n}.json') for n in arms}})
        print([(x['case'],x['new_metrics']['delta_margin']) for x in certify(plan)],flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['schedule']:schedule()
    elif sys.argv[1:]==['verify']:certify(bind())
    elif len(sys.argv)==4 and sys.argv[1]=='worker':worker(int(sys.argv[2]),sys.argv[3])
    else:raise SystemExit('schedule | verify')
