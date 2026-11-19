"""One natural Master3s2 common prefix, then one changed Sapling root and identical policy continuations."""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CARBON_DEBUG']='1'
import numpy as np

HERE=Path(__file__).resolve().parent;LEARN=HERE.parent;EXP=LEARN.parent;ACTOR=EXP.parent;BASE=ACTOR.parent
sys.path.insert(0,str(ACTOR/'elite_role_mix_dev/conditional'))
import responses as R
P=HERE/'protocol.json';REGISTERED='907cca96ae0c86d57195ceb32fef718575084d12cf77d1f2430fb9afff117b78'
PARENT=EXP/'candidate.py';STUDENT=LEARN/'candidate.py';ORIGINAL=R.live.SOURCE
STUDENT_SHA='bc69b55e9d61ee06b5c6f3ca038602d2283be4c8b61b0c1fd8f96e92a86ea597'
def digest(b):return hashlib.sha256(b).hexdigest()
def compact(data):return json.dumps(data,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()
def blob_write(path,raw):
    if path.exists():
        assert path.read_bytes()==raw
        return
    tmp=path.with_suffix('.incomplete');tmp.write_bytes(raw);tmp.replace(path)
def game_raw(case):
    f=R.ROOT/case['archived_map_fixture'];assert R.sha(f)==case['fixture_sha256']
    raw=R.load(f);assert raw['seed']==case['seed'] and raw['initial_terrain_u8_sha256']==case['terrain']
    return raw
def provenance():
    plan=R.load(P);assert R.sha(P)==REGISTERED and R.sha(PARENT)==plan['parent_sha256'] and R.sha(STUDENT)==STUDENT_SHA
    assert R.sha(LEARN/'protocol.json')==plan['fit_prereg_sha256']
    assert R.sha(ORIGINAL)==plan['source_original_sha256']==digest(subprocess.check_output(['git','-C',str(R.ROOT),'show','HEAD:sources/master3s2/master3s2.py']))
    gate=R.load(HERE/'qualified.json')
    assert gate['passed'] and gate['protocol']==REGISTERED and gate['candidate']==STUDENT_SHA
    for name,h in gate['evidence'].items():assert R.sha(LEARN/name)==h
    return dict(protocol=REGISTERED,parent=R.sha(PARENT),student=R.sha(STUDENT),original=R.sha(ORIGINAL),runner=R.sha(Path(__file__)),gate=R.sha(HERE/'qualified.json'))

def load_master(cpu):
    # Register the named module so a private local standard-library pickle can reconstruct its Belief classes.
    name='sapling_observation_native_master'
    spec=importlib.util.spec_from_file_location(name,ORIGINAL)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    assert m._SEED is None and m._TEMP==1 and m._RX and m._SEARCH_OK and m._S_ABORT==.58 and m._S_REC is None
    host=R.native_clock.host_snapshot(m);R.native_clock.validate_host(host,[cpu])
    assert host['torch_threads']==2 and all(x['cpus']==[cpu] for x in host['thread_affinity'])
    assert all(x['num_threads']==1 for x in host['pools'] if x['user_api']=='blas')
    return m,host
def capture(m):
    # bytes for CPU torch state avoid storage IDs in torch's tensor pickler.
    return pickle.dumps((m._ST,m._S_COST,m._S_STATS,random.getstate(),np.random.get_state(),
                         m._torch.get_rng_state().numpy().tobytes()),protocol=5)
def sig(m):
    # A value digest, independent of pickle's memo IDs and unordered-container insertion details.
    def canonical(x):
        if isinstance(x,np.random.RandomState):return ('rng',canonical(x.get_state()))
        if isinstance(x,np.ndarray):return ('array',str(x.dtype),x.shape,digest(x.tobytes()))
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,dict):return ('dict',[(canonical(k),canonical(v)) for k,v in sorted(x.items(),key=lambda kv:repr(kv[0]))])
        if isinstance(x,(set,frozenset)):return ('set',sorted([canonical(v) for v in x],key=repr))
        if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
        if isinstance(x,type):return ('type',x.__module__,x.__qualname__)
        if isinstance(x,(float,str,int,bool)) or x is None:return x
        if hasattr(x,'__dict__'):return ('object',x.__class__.__name__,canonical(vars(x)))
        raise TypeError(type(x))
    return digest(compact([canonical(m._ST),canonical(m._S_COST),canonical(m._S_STATS),canonical(random.getstate()),canonical(np.random.get_state()),digest(m._torch.get_rng_state().numpy().tobytes())]))
def restore(m,snapshot):
    st,cost,stats,py,npstate,torchstate=pickle.loads(snapshot)
    m._ST.clear();m._ST.update(st);m._S_COST[:]=cost;m._S_STATS.clear();m._S_STATS.update(stats)
    random.setstate(py);np.random.set_state(npstate)
    m._torch.set_rng_state(m._torch.tensor(list(torchstate),dtype=m._torch.uint8))
def checked_env(raw):return R.live.begin(raw)
def public(env,s):
    obs=R.live.public(env,s);assert 'full_carbon' not in obs and obs['player']==s
    return obs
def score(env):
    assert env.done
    st=[z.status for z in env.states];reward=[float(z.reward) for z in env.states]
    assert st==['DONE','DONE'];return reward,[float(p[0]) for p in env.states[0].observation.players],st
def physical(a,b,s):
    return [a,b] if s==0 else [b,a]
def own_call(model,obs,cfg):
    start=time.perf_counter();action=R.live.clean(model.agent(obs,cfg));return action,time.perf_counter()-start
def agreements(obs,a,b):
    workers=obs['players'][obs['player']][2]
    return {w:dict(role=v[2],at=v[0],cargo=v[1],parent=a.get(w,'NONE'),student=b.get(w,'NONE')) for w,v in workers.items() if a.get(w)!=b.get(w)}
def clock_result(rec,m):
    if rec.errors:raise AssertionError(rec.errors)
    return dict(calls=rec.calls,summary=R.native_clock.summarize(rec.calls,m._S_ABORT,m._S_D2) if rec.calls else None)

def shared(case,cpu,raw,m,host,folder,identity):
    env=checked_env(raw);s=case['seat'];op=1-s;cfg=raw['game_configuration'];rec=R.native_clock.Recorder(m)
    opponent=rec.wrap(m.agent); models={k:R.live.module(p,'observation_shared_'+k) for k,p in [('parent',PARENT),('student',STUDENT)]}
    trace=[];prefix=[];changes=None;t0=time.perf_counter()
    while not env.done:
        t=env.states[0].observation.step;a=public(env,s);e=public(env,op)
        # Both own actors see only their seat and no just-produced enemy command.
        enemy=R.live.clean(opponent(e,cfg))
        parent,pt=own_call(models['parent'],a,cfg);student,nt=own_call(models['student'],a,cfg)
        changed=agreements(a,parent,student)
        trace.append(dict(t=t,parent=parent,student=student,enemy=enemy,own_sha=digest(compact(a)),enemy_sha=digest(compact(e)),
                          changed=changed,parent_s=pt,student_s=nt))
        if 24<=t<=128 and changed:
            changes=dict(t=t,parent=parent,student=student,enemy=enemy,changed=changed,own_sha=digest(compact(a)),enemy_sha=digest(compact(e)))
            break
        prefix.append(physical(parent,enemy,s));env.step(prefix[-1])
        if t==128:break
    snapshot_sha=None;state_sha=None
    if changes is not None:
        state_sha=sig(m);data=capture(m);snapshot_sha=digest(data)
        restore(m,data);assert sig(m)==state_sha
        blob_write(folder/'original_root.pickle.gz',gzip.compress(data,mtime=0))
    result=dict(identity=identity,case=case,prefix=prefix,trace=trace,root=changes,host=host,cpu=cpu,clock=clock_result(rec,m),
                wall_s=time.perf_counter()-t0,snapshot_sha=snapshot_sha,state_sha=state_sha,prefix_sha=digest(compact(prefix)))
    assert len(rec.calls)==len(trace)
    R.write(folder/'shared.json.gz',result)
    return result

def branch(case,cpu,raw,m,folder,common,arm):
    path=folder/(arm+'.json.gz')
    if path.exists():
        data=R.load(path);assert data['identity']==common['identity'] and data['state_sha']==common['state_sha']
        return
    s=case['seat'];op=1-s;cfg=raw['game_configuration'];t0=time.perf_counter();env=checked_env(raw)
    owner=R.live.module(PARENT,'observation_continue_parent_'+arm)
    # Rebuild every observation-dependent feature of the shared continuation actor, including the root.
    for t,joint in enumerate(common['prefix']):
        obs=public(env,s);ref=common['trace'][t]
        assert digest(compact(obs))==ref['own_sha'] and R.live.clean(owner.agent(obs,cfg))==joint[s]
        env.step(joint)
    root=common['root'];obs=public(env,s)
    assert root['t']==obs['step'] and root['own_sha']==digest(compact(obs)) and root['enemy_sha']==digest(compact(public(env,op)))
    assert R.live.clean(owner.agent(obs,cfg))==root['parent']
    snap=gzip.decompress((folder/'original_root.pickle.gz').read_bytes())
    assert digest(snap)==common['snapshot_sha'];restore(m,snap);assert sig(m)==common['state_sha']
    rec=R.native_clock.Recorder(m); original=rec.wrap(m.agent); initial_host=rec.start
    root_joint=physical(root[arm],root['enemy'],s);tape=[root_joint];env.step(root_joint);own_wall=[];observed=[]
    while not env.done:
        t=env.states[0].observation.step;opp_obs=public(env,op);a=public(env,s);observed.append([digest(compact(a)),digest(compact(opp_obs))])
        en=R.live.clean(original(opp_obs,cfg));ours,w=own_call(owner,a,cfg);own_wall.append(w)
        joint=physical(ours,en,s);tape.append(joint);env.step(joint)
    rewards,cash,status=score(env);endhost=R.native_clock.host_snapshot(m);R.native_clock.validate_host(initial_host,[cpu]);R.native_clock.validate_host(endhost,[cpu])
    record=dict(identity=common['identity'],arm=arm,root_t=root['t'],root_joint=root_joint,seat=s,seed=case['seed'],case_terrain=case['terrain'],
                tape=tape,tape_sha=digest(compact(tape)),clock=clock_result(rec,m),own_public_seconds=own_wall,suffix_observations=observed,
                status=status,physical_cash=cash,official_scores=rewards,official_margin=rewards[s]-rewards[op],host_at_start=initial_host,host_at_end=endhost,
                state_sha=common['state_sha'],snapshot_sha=common['snapshot_sha'],shared_sha=R.sha(folder/'shared.json.gz'),wall_s=time.perf_counter()-t0)
    assert len(rec.calls)==len(tape)-1 and [r['step'] for r in rec.calls]==list(range(root['t']+1,root['t']+len(tape)))
    R.write(path,record)

def run(cpu,idx):
    os.sched_setaffinity(0,{cpu});identity=provenance();case=R.load(P)['cases'][idx];raw=game_raw(case)
    folder=HERE/f"{case['seed']}_{case['seat']}";folder.mkdir(exist_ok=True)
    with (folder/'execution.lock').open('a') as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Existing case process')
        m,host=load_master(cpu);common_path=folder/'shared.json.gz'
        if common_path.exists():
            c=R.load(common_path);assert c['identity']==identity and c['case']==case and c['cpu']==cpu
        else:
            # This lease prevents recreating a natural-random prefix after process loss.
            lease=folder/'prefix_started.json'
            if lease.exists():raise RuntimeError('An incomplete natural prefix is already registered for this case')
            R.write(lease,dict(identity=identity,cpu=cpu,seed=case['seed'],utc=time.time()))
            c=shared(case,cpu,raw,m,host,folder,identity)
        if c['root']:
            for arm in case['branch_order']:branch(case,cpu,raw,m,folder,c,arm)
        print(case['seed'],bool(c['root']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int,choices=(0,1),required=True);parser.add_argument('--cpu',type=int,required=True)
    args=parser.parse_args();run(args.cpu,args.case)
