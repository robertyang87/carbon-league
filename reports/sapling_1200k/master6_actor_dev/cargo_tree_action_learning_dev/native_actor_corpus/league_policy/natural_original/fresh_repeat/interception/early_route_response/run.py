"""Reconstruct a signed public prefix, then compare two frozen commands until official terminal."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'.git').exists())
CARGO=next(p for p in HERE.parents if (p/'ladder_live/live.py').is_file())
PROTOCOL='a0ddd82661631f8607767fba13ce4ad394115e262eb8e1d76e947cc6def56547'
RELEASE='d8f1e2770d2ca3fd123860d70ff97f72999be959'
ARM='sampled_child'
ACTOR_STATE=('_FEAT','_COUNTS','_chosen_route','_obs_key','_behavior_rng')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def sha(b):
    return hashlib.sha256(b).hexdigest()


def require(condition,message):
    if not condition: raise RuntimeError(message)


def save(name,doc):
    data=json.dumps(doc,allow_nan=False,separators=(',',':')).encode()
    if name.suffix=='.gz':data=gzip.compress(data,mtime=0)
    latest=name.with_suffix('.tmp')
    latest.write_bytes(data)
    latest.replace(name)


def setup(stage):
    for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[var]='1'
    os.environ['CARBON_DEBUG']='1'
    p=HERE/'protocol.json'
    signed=subprocess.check_output(['git','-C',str(ROOT),'show',RELEASE+':'+str(p.relative_to(ROOT))])
    require(signed==p.read_bytes() and sha(signed)==PROTOCOL,'unreleased protocol')
    plan=json.loads(signed)
    cpu=plan['cpu'][stage]
    os.sched_setaffinity(0,{cpu})
    k=load('original_kernel',CARGO/'learned_carbon_observation/native_response/run.py')
    live=load('original_ladder_public',CARGO/'ladder_live/live.py')
    sys.path.insert(0,str(CARGO/'native_actor_corpus/league_policy/natural_original'))
    import economy as e
    for key in ('candidate','previous_study','source_archive'):
        require(sha((ROOT/plan[key]['path']).read_bytes())==plan[key]['sha256'],'changed '+key)
    require(sha(k.ORIGINAL.read_bytes())==plan['original']['sha256']==
            sha(subprocess.check_output(['git','-C',str(ROOT),'show',RELEASE+':'+plan['original']['git_path']])),'unbound source')
    with zipfile.ZipFile(ROOT/plan['source_archive']['path']) as z:
        tape=z.read(ARM+'/turns.jsonl');receipt=z.read(ARM+'/receipt.json')
        require(sha(tape)==plan['tape_sha256'] and sha(receipt)==plan['receipt_sha256'],'changed game')
    actor=load('natural_early_child',ROOT/plan['candidate']['path'])
    original,host=k.load_master(cpu)
    cfg=e.config();seed=plan['seed']
    e.np.random.seed(seed);env=e.make('carbon',configuration={**cfg,'randomSeed':seed})
    return plan,k,live,e,env,{k:v for k,v in cfg.items() if k!='randomSeed'},[json.loads(x) for x in tape.splitlines()],actor,original,host


def play(stage,out):
    plan,k,live,e,env,cfg,tape,actor,original,host=setup(stage)
    prefix=stage=='prefix'
    rec=k.R.native_clock.Recorder(original); native=rec.wrap(original.agent)
    bank=[5.,5.]; trace=[]
    start=time.monotonic()
    for t,row in enumerate(tape[:plan['t']+1]):
        obs=[live.observe(env,s,bank[s]) for s in (0,1)]
        fps=[sha(live.encoded(o)) for o in obs]
        require(fps==row['public_observation_sha256'],'different signed public input')
        actual=[c['response']['action'] for c in row['calls']]
        bank=row['overage_remaining_seconds'].copy()
        if prefix:
            a,at=k.own_call(actor,obs[0],cfg); b=k.R.live.clean(native(obs[1],cfg))
            # The replay executed the archived actions even where fresh natural
            # inference disagrees. BeliefTracker explicitly remembers execution.
            original._ST['tr'].set_actions(obs[1]['players'],actual[1])
            trace.append(dict(t=t,published=fps,child_proposed=a,original_proposed=b,
                              child_wall=at,executed=actual))
            if t%40==0: print('reconstructed',t,flush=True)
        elif t==0:
            # Load immutable tensors once. Overwrite dynamic state from the
            # common snapshot after the last archived decision.
            actor.agent(obs[0],cfg);original._net()
        if t<plan['t']:env.step(actual)
    require(actual==plan['initial_joint_commands'] and fps==plan['root_public_hashes'],'different root')
    root_physical=sha(k.compact(dict(env.states[0].observation)))
    import zerosum_env.envs.carbon.idgen as ids
    id_sha=sha(k.compact(ids._id_dict))
    if prefix:
        require(a==actual[0] and a.get(plan['defender'])==plan['arm_actions']['control'],
                'candidate root action differs on faithful reconstruction')
        require(actor._capture is None and actor._forced is None,'unfinished decoder')
        original_state=k.capture(original)
        child_state=pickle.dumps({key:getattr(actor,key) for key in ACTOR_STATE},protocol=5)
        for name,value in [('original',original_state),('child',child_state)]:
            (out/(name+'.pickle.gz')).write_bytes(gzip.compress(value,mtime=0))
        bankdoc=dict(physical=root_physical,id_state=id_sha,root_public=fps,trace=trace,
            original_state_sha=sha(original_state),child_state_sha=sha(child_state),original_sig=k.sig(original),
            original_net=[type(original._NET).__name__,type(original._RNET).__name__],
            historical_public_checked=2*len(trace),original_prefix_disagreements=sum(r['original_proposed']!=r['executed'][1] for r in trace),
            child_prefix_disagreements=sum(r['child_proposed']!=r['executed'][0] for r in trace),
            clock=k.clock_result(rec,original),host=host,registered_commit=RELEASE,runner_sha=sha(Path(__file__).read_bytes()),
            wall=time.monotonic()-start)
        save(out/'shared.json.gz',bankdoc)
        return
    common=json.loads(gzip.decompress((out/'shared.json.gz').read_bytes()))
    require(common['physical']==root_physical and common['id_state']==id_sha and common['root_public']==fps,'fork mismatch')
    require(common['original_net']==[type(original._NET).__name__,type(original._RNET).__name__],'different warm backend')
    for name in ('child','original'):
        data=gzip.decompress((out/(name+'.pickle.gz')).read_bytes())
        require(sha(data)==common[name+'_state_sha'],'bad copied private state')
        if name=='child':
            for key,val in pickle.loads(data).items():setattr(actor,key,val)
        else:k.restore(original,data)
    require(k.sig(original)==common['original_sig'],'nonidentical opponent restore')
    rec=k.R.native_clock.Recorder(original);native=rec.wrap(original.agent)
    actions=deepcopy(actual);actions[0][plan['defender']]=plan['arm_actions'][stage]
    require(actions[1]==actual[1] and sum(actions[0].get(key)!=actual[0].get(key) for key in actual[0])==(stage!='control'),'fork action')
    turns=[];sim=e.FastCarbon(cfg).reset(seed=plan['seed'])
    for row in tape[:plan['t']]:sim.step(*[c['response']['action'] for c in row['calls']])
    sim.cash=e.CashEvents(sim.cash)
    max_t=cfg['episodeSteps']
    while not env.done:
        t=env.states[0].observation.step
        require(t<=max_t and sim.step_count==t,'turn mismatch')
        if t!=plan['t']:
            obs=[live.observe(env,s,bank[s]) for s in (0,1)]
            fps=[sha(live.encoded(o)) for o in obs]
            own,at=k.own_call(actor,obs[0],cfg)
            enemy=k.R.live.clean(native(obs[1],cfg))
            times=[at,rec.calls[-1]['public_wall_seconds']]
            actions=[own,enemy]
            bank=[max(0.,v-max(0.,cost-1.)) for v,cost in zip(bank,times)]
        else:times=None
        sim.cash.events=[];sim.step(*actions);env.step(actions)
        o=env.states[0].observation
        require(sim.full_observation()['players']==o.players and sim.full_observation()['trees']==o.trees and sim.carbon==o.full_carbon,'physical twin mismatch')
        turns.append(dict(t=t,pre_public_sha=fps,joint=actions,public_wall=times,remaining_overage=bank.copy(),
            post_cash=list(sim.cash),target_after=o.players[1][2].get(plan['target']),
            defender_after=o.players[0][2].get(plan['defender']),cash_events=sim.cash.events))
    scores,cash,status=k.score(env)
    save(out/(stage+'.json.gz'),dict(arm=stage,registered_commit=RELEASE,runner_sha=sha(Path(__file__).read_bytes()),
        shared_sha=sha((out/'shared.json.gz').read_bytes()),root_physical=root_physical,root_id_state=id_sha,
        clocks=k.clock_result(rec,original),host=host,host_end=k.R.native_clock.host_snapshot(original),
        source_original_sig=common['original_sig'],source_child_sha=common['child_state_sha'],
        turns=turns,official_scores=scores,cash=cash,status=status,margin=round(scores[0]-scores[1],2),
        wall=time.monotonic()-start))
    print(stage,scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prefix','control','intervention'])
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/(args.stage+'.lock')).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (args.out/(('shared' if args.stage=='prefix' else args.stage)+'.json.gz')).exists(),'attempt already completed')
        play(args.stage,args.out)
