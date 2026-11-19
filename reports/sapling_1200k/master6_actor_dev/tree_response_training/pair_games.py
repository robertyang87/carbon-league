"""Two original-opponent workers, resuming each independently registered paired intervention."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import datetime
import time
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from select_roots import HERE,PLAN,ROOT,SOURCE,clean,live,load,retain,sha
from source_games import ENV

def records():
    selected=load(HERE/'selection.json')
    if (selected['protocol_sha256']!=sha(HERE/'protocol.json') or selected['source_receipt_sha256']!=sha(HERE/'source_receipt.json')
            or selected['all_opportunities_sha256']!=sha(HERE/'opportunities.jsonl.gz')):
        raise ValueError('training protocol, upstream source or observation index changed')
    items=selected['cases']
    if len(items)>8 or len({(r['seed'],r['seat'],r['window']) for r in items})!=len(items):
        raise ValueError('duplicate training pair or excess intervention quota')
    if sha(SOURCE)!=PLAN['parent_pack_sha256'] or sha(live.SOURCE)!=live.MASTER_SHA:
        raise ValueError('wrong student or original examiner')
    live.PACK=SOURCE;live.PACK_HASH=PLAN['parent_pack_sha256']
    return items

def play(case_no,arm):
    reg=records()[case_no];wd=HERE/reg['case_directory'];src=ROOT/reg['source']
    r=load(wd/'root.json.gz');tape=load(src);s,t=reg['seat'],reg['step']
    if (sha(wd/'root.json.gz')!=reg['root_sha256'] or sha(src)!=reg['source_sha256']
            or r['v1']!=clean(tape['actions'][t][s]) or r['old']!=reg['old'] or r['new']!=reg['new']
            or {k for k in set(r['v1'])|set(r['alternative']) if r['v1'].get(k)!=r['alternative'].get(k)}!={reg['uid']}):
        raise ValueError('wrong input state or more than one worker intervention')
    q={'seed':reg['seed'],'seat':s,'root_t':t,'family':'tree_terminal_training',
       'source_dir':str(Path(reg['source']).parent),
       'record':{'seed':reg['seed'],'seat':s,'source':reg['source'],'source_sha256':reg['source_sha256']},
       'root':{'t':t,'worker':reg['uid'],'old':reg['old'],'alternative':reg['new'],'tree':''},
       'training_root_sha256':reg['root_sha256'],'training_protocol_sha256':sha(HERE/'protocol.json'),
       'paired_responder_source_sha256':sha(Path(live.__file__))}
    with (wd/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        retain(wd/'live_protocol.json',q)
        rng=np.random.RandomState(reg['rng_seed']).get_state()
        scalars={'source_sha256':live.MASTER_SHA,'mt_algorithm':rng[0],'mt_keys':rng[1].tolist(),'mt_index':rng[2],
                 'has_normal':rng[3],'stored_normal':rng[4]}
        shared=wd/'live_shared_init.json'
        if not shared.exists():
            original=live.module(live.SOURCE,'tree_training_source_master')
            retain(shared,{**scalars,'warm_cost':list(original._S_COST)})
        if any(load(shared)[k]!=v for k,v in scalars.items()):
            raise ValueError('responder random or original source changed')
    live.run(arm,wd,q,resume=True)

def schedule():
    items=records()
    allocation=load(HERE/'response_resources.json')
    if allocation['parent_protocol_sha256']!=sha(HERE/'protocol.json') or allocation['paired_case_lanes']!=[[0,1],[2,3]]:
        raise ValueError('unmatched resource amendment')
    def other_native():
        jobs=[]
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                args=(proc/'cmdline').read_bytes().split(b'\0')
            except (OSError,PermissionError):
                continue
            if len(args)<2 or b'python' not in Path(os.fsdecode(args[0])).name.encode():
                continue
            values=[os.fsdecode(q) for q in args if q.endswith(b'.py')]
            if any('tree_response_training' in v for v in values):
                continue
            if any(('tree_pack_arena.py' in v or '/workforce_policy/branch_' in v or
                    '/opponent_transfer/run.py' in v or '/terminal_return_fit/multimap/run.py' in v) for v in values):
                jobs.append(int(proc.name))
        return jobs
    def lane(job):
        lane_index,cpus=job
        for i in range(lane_index,len(items),2):
            while other_native():
                time.sleep(8)
            directory=HERE/items[i]['case_directory']
            stamp=directory/'resource_at_start.json'
            if not stamp.exists():
                retain(stamp,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpus':cpus,
                            'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                            'cgroup_stat_at_launch':Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines(),
                            'loadavg_at_launch':Path('/proc/loadavg').read_text().strip(),'other_native_pids_at_launch':other_native(),
                            'allocation_sha256':sha(HERE/'response_resources.json')})
            def command(worker):
                k,cpu=worker
                arm='parent' if (k+i//2)%2==0 else 'alternative'
                subprocess.run(['taskset','-c',str(cpu),sys.executable,'-u',str(Path(__file__)), 'play',
                                '--index',str(i),'--arm',arm],env=ENV,check=True)
            with ThreadPoolExecutor(max_workers=2) as pair:
                list(pair.map(command,enumerate(cpus)))
    with (HERE/'.response_schedule.lock').open('a') as lock,ThreadPoolExecutor(max_workers=2) as pool:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        list(pool.map(lane,enumerate(allocation['paired_case_lanes'])))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('play','schedule'));ap.add_argument('--index',type=int);ap.add_argument('--arm',choices=('parent','alternative'));args=ap.parse_args()
    if args.mode=='schedule':schedule()
    else:play(args.index,args.arm)
