"""Run disjoint frozen tickets while preserving the already-started single-process episode."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import full as F
H=F.H
def tickets(stage):
    if stage=='axes':
        return [('train',f'axis{i}_{k}',s,side) for i in range(3) for k in ('plus','minus') for s in F.P['train'] for side in (0,1)]
    return [(fold,a,s,side) for fold in ('train','holdout') for s in F.P[fold] for side in (0,1) for a in ('parent','selected')]
def dest(ticket):
    f,a,s,side=ticket;return H/'games'/f/f'{a}_{s}_{side}.json.gz'
def worker(stage,cpu):
    assert os.sched_getaffinity(0)=={cpu}
    lock=F.A/'tree_response_training/.continuation.lock'
    plan=F.read(H/f'worker_{stage}.json');cache={}
    for f,a,s,side in plan['assignment'][str(cpu)]:
        if a not in cache:
            if a=='parent':p=F.assets['parent']
            elif a=='selected':p=H/'selected.py'
            else:
                axis=int(a[4]);sign=1 if a.endswith('plus') else -1
                v=[0.,0.,0.];v[axis]=sign*.5
                p,_=F.packaged(a+'_cpu'+str(cpu),v)
            cache[a]=p
        with lock.open('a+') as native:
            if cpu in (0,1):fcntl.flock(native,fcntl.LOCK_SH)
            F.play(a,cache[a],s,side,f)
            if cpu in (0,1):fcntl.flock(native,fcntl.LOCK_UN)
def pause(pid):
    raw=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'full.py' in raw and Path(f'/proc/{pid}/cwd').resolve()==H
    os.kill(pid,signal.SIGSTOP)
    for i in range(50):
        if 'State:\tT' in Path(f'/proc/{pid}/status').read_text():return
        time.sleep(.02)
    raise RuntimeError('own sequential worker was not paused')
def parallel(pid,stage,cpus):
    pause(pid);procs=[]
    try:
        scheduled=tickets(stage)
        missing=[t for t in scheduled if not dest(t).exists()]
        if len(missing)<=1:return
        # The stopped sequential process may only have entered the FIRST missing ticket.
        # Its current in-memory game and RNG state are preserved. No helper receives that ticket.
        reserved=missing[0];jobs=missing[1:]
        plan={'registration':F.reg['protocol'],'main_pid':pid,'paused_ticket':list(reserved),'assignment':{str(cpu):jobs[i::len(cpus)] for i,cpu in enumerate(cpus)}}
        F.put(H/f'worker_{stage}.json',plan)
        for cpu in cpus:
            log=(H/f'worker_{stage}_cpu{cpu}.log').open('a')
            cmd=['nice','-n','19','taskset','-c',str(cpu),sys.executable,'-u',str(Path(__file__)),'worker',stage,str(cpu)]
            procs.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
        for p,_ in procs:
            if p.wait()!=0:raise RuntimeError('registered ticket worker failed; retain completed receipts')
        assert all(dest(t).is_file() for t in jobs)
    finally:
        for p,f in procs:
            if p.poll() is None:p.wait()
            f.close()
        os.kill(pid,signal.SIGCONT)
def run(pid,cpus):
    parallel(pid,'axes',cpus)
    while not (H/'selection.json').is_file():
        if not Path(f'/proc/{pid}').is_dir():raise RuntimeError('sequential trainer exited before sealing final selection')
        time.sleep(1)
    parallel(pid,'comparison',cpus)
if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
    else:run(int(sys.argv[2]),list(map(int,sys.argv[3:])))
