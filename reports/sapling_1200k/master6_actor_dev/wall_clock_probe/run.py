"""Call the byte-identical public examiner on identical historical states with wall-clock scheduling as the intervention."""
from collections import Counter
from pathlib import Path
import copy
import ctypes
import fcntl
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/"root_recovery_branches"))
import live
from branches import begin, clean, public
from tree_win_gate import validate_examiner_configuration
P=HERE/"protocol.json"
def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)
def plain(o):
    if isinstance(o,np.ndarray):return o.tolist()
    if isinstance(o,np.generic):return o.item()
    raise TypeError(type(o))
def encoded(o):return json.dumps(o,default=plain,sort_keys=True,allow_nan=False,separators=(",",":")).encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def save(p,o):
    data=encoded(o)+b"\n"
    if p.exists():
        if data!=p.read_bytes():raise ValueError(f"frozen diagnostic differs {p}")
    else:
        q=p.with_suffix(".partial");q.write_bytes(data);q.replace(p)
def process_sample():
    # Count sibling work in our cgroup using accounted task CPU. Do not use host-wide averages.
    txt=Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines()
    cg={k:int(v) for k,v in (x.split() for x in txt)}
    out={}
    for ent in Path('/proc').iterdir():
        if not ent.name.isdigit() or int(ent.name)==os.getpid():continue
        try:
            if (ent/'cgroup').read_text()!=Path('/proc/self/cgroup').read_text():continue
            for thread in (ent/'task').iterdir():
                z=(thread/'stat').read_text().rsplit(')',1)[1].split()
                if z[0]!='Z':out[(ent.name,thread.name)]=(int(z[11])+int(z[12]),int(z[36]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return time.monotonic(),cg,out
def clearance(cpu):
    while True:
        t,a,src=process_sample();time.sleep(.7);end,b,tar=process_sample();dt=end-t
        cores=(b['usage_usec']-a['usage_usec'])/1e6/dt
        near=sum(max(0,v[0]-src[k][0]) for k,v in tar.items() if k in src and v[1]==cpu and src[k][1]==cpu)/os.sysconf('SC_CLK_TCK')/dt
        if cores<3.35 and near<.13:return {'sample_seconds':dt,'cgroup_cores':cores,'other_on_target_core':near,'pre_nr_throttled':b['nr_throttled']}
        print('waiting for idle CPU',round(cores,2),round(near,2),flush=True)
        time.sleep(3)
def burn(ready):
    ctypes.CDLL(None).prctl(1,signal.SIGKILL)
    if os.getppid()==1:os._exit(2)
    os.write(ready,b'R');os.close(ready);i=9
    while True:i=(i*1103515245+12345)&2147483647
def state_snapshot(path):
    return {k:int(v) for k,v in (line.split() for line in path.read_text().splitlines())}
def measured(master,obs,cfg,tracker,previous,seed,cost,stats,mode):
    st=master._ST;st.clear();st.update(tr=copy.deepcopy(tracker),prev=copy.deepcopy(previous),rng=np.random.RandomState(seed))
    master._S_COST[:]=cost;master._S_STATS=copy.deepcopy(stats);master._S_REC=[]
    rng_before=digest(encoded(st['rng'].get_state()));before=state_snapshot(Path('/sys/fs/cgroup/cpu.stat'))
    burner=None
    if mode=='contended':
        rd,wr=os.pipe();burner=os.fork()
        if burner==0:
            os.close(rd);burn(wr);os._exit(3)
        os.close(wr)
        if os.read(rd,1)!=b'R':raise RuntimeError('burner not ready')
        os.close(rd)
    try:
        t=time.perf_counter();c=time.process_time();decision=master.agent(obs,cfg);cpu=time.process_time()-c;wall=time.perf_counter()-t
    finally:
        if burner:
            os.kill(burner,signal.SIGTERM);_,_,ru=os.wait4(burner,0)
    after=state_snapshot(Path('/sys/fs/cgroup/cpu.stat'))
    assert np.isfinite(wall) and master._TEMP==1 and master._SEED is None
    s=master._S_STATS;rec=master._S_REC
    assert len(rec)<=1 and len(s['time'])<=1
    if rec:
        step,pos,candidates,depths,values,chosen=rec[0]
        assert step==obs['step'] and pos==obs['player'] and len(depths)==len(candidates) and clean(decision)==clean(candidates[chosen])
    else:
        candidates=[];depths=[];chosen=None
    return dict(mode=mode,action=decision,action_sha256=digest(encoded(decision)),rng_before_sha256=rng_before,
        rng_after_sha256=digest(encoded(st['rng'].get_state())),cost_before=cost,cost_after=list(master._S_COST),
        wall=wall,cpu=cpu,competing_child_cpu=ru.ru_utime+ru.ru_stime if burner else 0,
        cgroup_throttled_us=after['throttled_usec']-before['throttled_usec'],search_completed=s['searched'],
        aborted=s['aborted'],single=s['single'],returned_candidate_count=len(candidates),per_candidate_depth=depths,
        completed_depth_hist=s['depth'],reported_cands_hist=s['cands'],selected_candidate=chosen,selected_other=s['changed'],
        native_last=s.get('last'),returned_candidate_sequence_sha256=digest(encoded(candidates)))
def run(plan):
    cpu=plan['cpu'];os.sched_setaffinity(0,{cpu});os.environ['CARBON_DEBUG']='1'
    assert all(os.environ[k]=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    assert digest(live.SOURCE.read_bytes())==live.MASTER_SHA==digest(subprocess.check_output(['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=ROOT))
    warm=ROOT/plan['warm_cost_source'];assert digest(warm.read_bytes())==plan['warm_cost_sha256']
    assert Path('/sys/fs/cgroup/cpu.max').read_text().split()==['400000','100000']
    directory=HERE/'roots';directory.mkdir(exist_ok=True)
    code_sha=digest(Path(__file__).read_bytes());protocol_sha=digest(P.read_bytes())
    receipts=[]
    for i,row in enumerate(plan['roots']):
        target=directory/f'{i:02d}.json'
        if target.exists():
            old=read(target);assert (old['protocol_sha256'],old['runner_sha256'],old['root'])==(protocol_sha,code_sha,row)
            receipts.append(old);continue
        rawfile=ROOT/row['source'];assert digest(rawfile.read_bytes())==row['sha256']
        raw=read(rawfile);validate_examiner_configuration(raw['configuration']);side=row['actor_seat'];t=row['t']
        assert side==1-raw['candidate_seat'] and row['seed']==raw['seed']
        env=begin(raw);master=live.module(live.SOURCE,'wall_clock_original')
        assert all(getattr(master,k)==v for k,v in plan['fixed_search'].items())
        stats=copy.deepcopy(master._S_STATS)
        tracker=master.BeliefTracker(master._Bel,side);prev=None;history=hashlib.sha256()
        for j,orders in enumerate(raw['actions'][:t]):
            obs=public(env,side);assert obs['step']==j
            history.update(encoded([obs,orders[side]]))
            if j==t-1:
                machine=clearance(cpu)
                warmup=measured(master,obs,raw['game_configuration'],tracker,prev,row['paired_rng_seed']-773,
                                read(warm)['warm_cost'],stats,'preceding-state-cost')
                costs=warmup['cost_after']
            tracker.step(obs['players'],j,obs['carbon']);tracker.set_actions(obs['players'],orders[side]);prev=obs['players']
            env.step(orders)
        obs=public(env,side);out=[]
        for mode in row['order']:
            clear=clearance(cpu)
            result=measured(master,obs,raw['game_configuration'],tracker,prev,row['paired_rng_seed'],costs,stats,mode)
            result['host_precheck']=clear;out.append(result)
        assert len({v['rng_before_sha256'] for v in out})==1 and all(v['cost_before']==costs for v in out)
        r=dict(protocol_sha256=protocol_sha,runner_sha256=code_sha,root=row,public_root_sha256=digest(encoded(obs)),
               actual_history_sha256=history.hexdigest(),nice=os.getpriority(os.PRIO_PROCESS,0),terrain=raw['initial_terrain_u8_sha256'],
               historical_cost_calibration=warmup,machine_at_calibration=machine,measurements=out)
        save(target,r);receipts.append(r);print(i,t,[(x['mode'],x['wall'],x['aborted'],x['completed_depth_hist']) for x in out],flush=True)
    aggregate=[]
    for r in receipts:
        q={v['mode']:v for v in r['measurements']};idle1,idle2,cont=[q[k] for k in ('idle_1','idle_2','contended')]
        aggregate.append(dict(t=r['root']['t'],idle_repeat_action_difference=idle1['action']!=idle2['action'],
            contention_vs_idle_1_action_difference=cont['action']!=idle1['action'],
            contention_vs_idle_2_action_difference=cont['action']!=idle2['action'],
            cont_differs_both=all(cont['action']!=v['action'] for v in (idle1,idle2))))
    save(HERE/'results.json',dict(protocol_sha256=protocol_sha,runner_sha256=code_sha,examiner_sha256=live.MASTER_SHA,
         cpu=cpu,cpu_model=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
         measured_calls=24,roots=aggregate,receipt_sha256=[digest((directory/f'{i:02d}.json').read_bytes()) for i in range(8)]))
if __name__=='__main__':
    with (HERE/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(read(P))
