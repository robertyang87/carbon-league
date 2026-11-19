"""Replay the exact two selected tree choices under four additional predeclared native streams."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np

from choose import D,T,REG,choose
from select_roots import load,retain,sha,SOURCE,ROOT,live
from source_games import ENV
import label as authority

ROOTS_SHA='4ad348b5b6c30694c1e4ceb71aa02667b26155e69670d2cf2a599bd78c3836d5'
LAYOUT=[(0,0),(1,0),(1,1),(0,1)]
ARMS=('parent','alternative')


def require(ok,why):
    if not ok:raise ValueError(why)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bind():
    require(sha(D/'protocol.json')==REG and sha(D/'roots.json')==ROOTS_SHA,'study freeze changed')
    conf=load(D/'protocol.json')
    require(sha(Path(live.__file__))==conf['original_live_sha256'] and sha(SOURCE)==conf['required_parent_pack_sha256']
            and sha(live.SOURCE)==live.MASTER_SHA==conf['required_original_source_sha256']
            and hashlib.sha256(subprocess.check_output(['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=ROOT)).hexdigest()==live.MASTER_SHA,
            'original search source, parent or responder changed')
    live.PACK=SOURCE;live.PACK_HASH=sha(SOURCE)
    return conf


def group(i):
    ri,si=LAYOUT[i];pick=load(D/'roots.json')['roots'][ri];r=pick['registered_case']
    old=T/pick['area']/r['case_directory']
    return pick,r,old,D/f"root_{ri}_stream_{si}",pick['new_stream_seeds'][si]


def prepare():
    cfg=bind();check_roots=choose()
    require(sha(D/'roots.json')==ROOTS_SHA,'mechanical roots differ from signed choices')
    info=[]
    for i in range(len(LAYOUT)):
        pick,reg,old,wd,seed=group(i);wd.mkdir(exist_ok=True)
        require(all(sha(old/f)==v for f,v in pick['old_files'].items()),'original case or capture changed')
        src=ROOT/reg['source'];require(sha(src)==reg['source_sha256'],'different declared source episode')
        root=wd/'root.json.gz'
        if root.exists():require(sha(root)==reg['root_sha256'],'wrong cloned public state')
        else:root.write_bytes((old/'root.json.gz').read_bytes())
        orig=load(old/'live_protocol.json')
        config={**orig,'family':'frozen_tree_pay_repeatability','repeat_study_sha256':REG,'repeat_roots_sha256':ROOTS_SHA,
                'original_live_protocol_sha256':sha(old/'live_protocol.json'),'new_seed':seed}
        retain(wd/'live_protocol.json',config)
        prev=load(old/'live_shared_init.json');rng=np.random.RandomState(seed).get_state()
        new={**prev,'mt_algorithm':rng[0],'mt_keys':rng[1].tolist(),'mt_index':rng[2],'has_normal':rng[3],'stored_normal':rng[4]}
        require(new['mt_keys']!=prev['mt_keys'] and new['warm_cost']==prev['warm_cost'] and new['warm_cost'],'ineligible repeated stream or cost reset')
        retain(wd/'live_shared_init.json',new)
        info.append({'case':wd.name,'case_sha256':sha(wd/'live_protocol.json'),'root_sha256':sha(root),'stream_sha256':sha(wd/'live_shared_init.json'),
                     'parent_cpu':i%2,'alternative_cpu':1-i%2,'old_shared_sha256':sha(old/'live_shared_init.json'),'new_seed':seed})
    require(len({x['stream_sha256'] for x in info})==4,'repeated continuation stream')
    release={'protocol_sha256':REG,'roots_sha256':ROOTS_SHA,'run_sha256':sha(Path(__file__)),
             'authority_sha256':sha(Path(authority.__file__)),'native_source_sha256':cfg['original_live_sha256'],'jobs':info}
    retain(D/'execution.json',release)
    return release


def worker(i,name,cpu):
    bind();ex=load(D/'execution.json');chosen=ex['jobs'][i]
    require(os.sched_getaffinity(0)=={cpu} and cpu==chosen[name+'_cpu'] and ex['run_sha256']==sha(Path(__file__))
            and ex['roots_sha256']==ROOTS_SHA and (D/'court.json').exists(),'wrong engine affinity or program')
    _,r,_,wd,_=group(i)
    require(sha(wd/'live_protocol.json')==chosen['case_sha256'] and sha(wd/'live_shared_init.json')==chosen['stream_sha256']
            and sha(wd/'root.json.gz')==chosen['root_sha256'],'different randomized branch inputs')
    with (wd/f'.{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        live.run(name,wd,load(wd/'live_protocol.json'),resume=True)


def audited(i):
    picked,reg,old,wd,seed=group(i);original=load(old/'training_label.json')
    authority.HERE=D;cfg=load(D/'protocol.json')
    authority.PLAN={'parent_pack_sha256':cfg['required_parent_pack_sha256']}
    z=authority.label({**reg,'case_directory':wd.name})
    require(z['pay_intent_filled'] and z['root_sha256']==original['root_sha256'] and z['pay_is_v1']==original['pay_is_v1']
            and z['pay']['root_debit']==original['pay']['root_debit'] and z['defer']['root_debit']==0,'payment or branch orientation changed')
    arms={}
    for name in ARMS:
        r=load(wd/f'live_{name}.json');o=load(old/f'live_{name}.json')
        require(r['root_actions']==o['root_actions'] and r['root_opponent_observation_sha256']==o['root_opponent_observation_sha256']
                and r['original_trajectory_sha256']==reg['source_sha256'] and r['source_sha256']==cfg['required_original_source_sha256']
                and r['statuses']==['DONE','DONE'] and r['terminal_t']>reg['step'],'changed root/visible history, examiner or terminal')
        arms[name]={'scores':r['score'],'margin':r['margin'],'u':r['u'],'elapsed_seconds':r['elapsed'],'real_search_calls':r['search_calls'],
                    'search_completed':r['search_completed'],'search_changed':r['search_changed'],'search_aborted':r['search_aborted'],
                    'action_tape_sha256':r['action_sha256'],'result_sha256':sha(wd/f'live_{name}.json'),'checkpoint_resumes':r['resumed_checkpoints']}
    out={'root':reg['root_sha256'],'root_source':reg['source_sha256'],'rng_seed':seed,'shared_state_sha256':sha(wd/'live_shared_init.json'),
         'margin_pay_minus_defer':z['target_delta_official_margin'],'actual_debit':z['pay']['root_debit'],'arms':arms,
         'official_verified_label':z,'elapsed_pair_seconds':load(wd/'completed_timing.json')['pair_wall_seconds']}
    retain(wd/'certified.json.gz',out)
    return out


def signed_report(pairs):
    output=[]
    for ri in [0,1]:
        indices=[i for i,(k,s) in enumerate(LAYOUT) if k==ri]
        pick,r,old,_,_=group(indices[0])
        authority.HERE=old.parent;authority.PLAN=load(old.parent/'protocol.json')
        baseline=authority.label(r)
        require(baseline==load(old/'training_label.json'),'independent historical payment/score differs from original')
        rep=[pairs[i] for i in indices];ys=[baseline['target_delta_official_margin']]+[q['margin_pay_minus_defer'] for q in rep]
        vals=[{'kind':'historical_training','seed':r['rng_seed'],'effect':ys[0],
               'source_sha256':sha(old/'training_label.json'),'parallel_wall_from_native_timers':max(load(old/f'live_{j}.json')['elapsed'] for j in ARMS)}]
        for index,item in zip(indices,rep):
            vals.append({'kind':'new_rng_replicate','seed':item['rng_seed'],'effect':item['margin_pay_minus_defer'],'source_sha256':sha(group(index)[3]/'certified.json.gz'),
                         'pair_wall_seconds':item['elapsed_pair_seconds'],'parallel_wall_from_native_timers':max(v['elapsed_seconds'] for v in item['arms'].values())})
        output.append({'map':r['seed'],'seat':r['seat'],'turn':r['step'],'uid':r['uid'],'map_sha256':pick['map_sha256'],
                       'n':len(ys),'replicates':vals,'min':min(ys),'max':max(ys),'range':max(ys)-min(ys),
                       'mean':statistics.mean(ys),'sample_sd':statistics.stdev(ys),'both_signs':min(ys)<0<max(ys)})
    walls=[q['elapsed_pair_seconds'] for q in pairs]
    result={'schema':'published-repeated-terminal-tree-roots-v1','protocol_sha256':REG,'roots_sha256':ROOTS_SHA,'execution_sha256':sha(D/'execution.json'),
            'registered_cases':output,'new_native_arms':8,'new_genuine_pairs':4,'new_unseen_maps':0,
            'native_pair_wall_seconds':walls,'mean_native_pair_wall_seconds':statistics.mean(walls),
            'observed_sign_flip_roots':sum(q['both_signs'] for q in output),'formal_games':0}
    retain(D/'results.json',result);print(json.dumps(result),flush=True)


def busy():
    active=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (OSError,FileNotFoundError):continue
        if not args or b'python' not in args[0]:continue
        if any(x.endswith(b'tree_pack_arena.py') or x.endswith(b'native_last.py') and b'run' in args for x in args):active.append(int(p.name))
    return active


def schedule():
    with (D/'.owner.lock').open('a') as single:
        fcntl.flock(single,fcntl.LOCK_EX|fcntl.LOCK_NB);release=prepare()
        with (T/'.continuation.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            while busy():time.sleep(4)
            require(load(T.parent/'late_workforce_dev/native/results.json')['all_four_responses']
                    and load(T.parent/'endgame_queue_dev/native_result.json')['native_calls']==8,'known court predecessors open')
            if not (D/'court.json').exists():
                retain(D/'court.json',{'time_utc':utc(),'cpus':[0,1],'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                     'loadavg_at_lease':Path('/proc/loadavg').read_text().strip(),'registered_jobs_sha256':sha(D/'execution.json'),
                     'active_other_native':busy()})
            pairs=[]
            for i,job in enumerate(release['jobs']):
                pick,r,old,wd,seed=group(i)
                if not (wd/'completed_timing.json').is_file():
                    if not (wd/'timing_start.json').exists():
                        retain(wd/'timing_start.json',{'utc':utc(),'monotonic':time.monotonic(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                            'arm_cores':{name:job[name+'_cpu'] for name in ARMS},'cgroup_cpu_stat':Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines()})
                    t0=load(wd/'timing_start.json')
                    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==t0['boot_id'],'monotonic clock epoch changed')
                    def launch(name):
                        cpu=job[name+'_cpu']
                        subprocess.run(['taskset','-c',str(cpu),sys.executable,'-u',str(Path(__file__)),'worker',str(i),name,str(cpu)],env=ENV,check=True)
                    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(launch,ARMS))
                    retain(wd/'completed_timing.json',{'ending_utc':utc(),'pair_wall_seconds':time.monotonic()-t0['monotonic'],
                                                       'arm_seals':{n:sha(wd/f'live_{n}.json') for n in ARMS}})
                pairs.append(audited(i))
                print('signed native repeated pair',wd.name,pairs[-1]['margin_pay_minus_defer'],flush=True)
            signed_report(pairs)


if __name__=='__main__':
    args=sys.argv[1:]
    if args==['prepare']:prepare()
    elif args==['schedule']:schedule()
    elif len(args)==4 and args[0]=='worker':worker(int(args[1]),args[2],int(args[3]))
    else:raise SystemExit('prepare | schedule | worker INDEX ARM CPU')
