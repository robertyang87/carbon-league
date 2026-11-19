"""One reproducible sampled joint-action update followed by fixed greedy deployment checks against two cheap opponents."""
import argparse
import copy
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from world import World, make_opponent, replay_official, CFG
import policy
from trust_update import train

HERE=policy.HERE; BASE=HERE.parents[1]; ROOT=BASE.parents[1]
TRACE=HERE.parent/'tree_response_training'
POINTS=(260,271,282)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def write(p,v):
    if p.exists():
        if read(p)!=v:raise ValueError('different frozen result '+str(p))
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(v,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
    if p.suffix=='.gz':b=gzip.compress(b,mtime=0)
    temp=p.with_suffix('.partial');temp.write_bytes(b);temp.replace(p)


def bind():
    files=['protocol.json','map_declaration.json','policy.py','world.py','trust_update.py','run.py']
    return {k:sha(HERE/k) for k in files}


def background():
    tables=[TRACE/'source_receipt.json',TRACE/'batches/intake_02/source_receipt.json']
    return {(r['seed'],r['seat']):(ROOT/r['raw_path'],r['raw_sha256']) for q in tables for r in read(q)['games']}


def snapshots(module,seed,seat,opponent):
    path,hash_=background()[seed,seat];assert sha(path)==hash_
    raw=read(path);assert raw['seed']==seed and raw['candidate_seat']==seat
    w=World(seed);assert w.terrain_sha256==raw['initial_terrain_u8_sha256']
    f=sys.modules['cnet2_feat'].Feat(seat);enemy=make_opponent(opponent);poss=[]
    policy.set_delta(module,policy.ZERO)
    for t in range(1+max(POINTS)):
        assert w.step==t
        x=f.observe(w.obs(seat));e=enemy(w.obs(1-seat),CFG)
        if t in POINTS:
            local=copy.deepcopy(f)
            root=policy.Root(module,local,x);pick=root.primary()
            if pick is not None:
                slot,entropy,command=pick
                poss.append((entropy,t,slot,command,root,copy.deepcopy(w),local,copy.deepcopy(enemy),e))
        if t!=max(POINTS):
            if w.advance(raw['actions'][t]) is not None:break
    if not poss: raise ValueError('registered training source had no selectable worker '+str((seed,seat)))
    chosen=max(poss,key=lambda v:(v[0],-v[1]))
    profile=[dict(t=t,slot=s,greedy_prefix_entropy=en) for en,t,s,*_ in poss]
    return chosen,profile,raw['actions'][:chosen[1]],path,hash_


def verify(seed,tape,terminal):
    official=replay_official(seed,tape)
    for k in ('terrain_sha256','steps','termination','result_by_seat','extinct_by_seat'):
        assert official[k]==terminal[k],(k,official[k],terminal[k])
    # The platform rounds each reported physical-seat reward to cents.
    assert max(abs(a-b) for a,b in zip(official['scores'],terminal['scores']))<=.005001,(official['scores'],terminal['scores'])
    return official


def suffix(module,seat,seed,root_world,feature,enemy,enemy_order,cmd,opening):
    w=copy.deepcopy(root_world);f=copy.deepcopy(feature);opp=copy.deepcopy(enemy)
    tape=[];ours=cmd;theirs=enemy_order
    while w.terminal is None:
        joint=[ours,theirs] if seat==0 else [theirs,ours];tape.append(joint);w.advance(joint)
        if w.terminal is None:
            x=f.observe(w.obs(seat));ours=f.to_actions(module._mixed_actions(x,f.aux))
            theirs=opp(w.obs(1-seat),CFG)
    return dict(actions=tape,official=verify(seed,opening+tape,w.terminal))


def collect(module,seed,seat,name,binding):
    chosen,profile,opening,path,h=snapshots(module,seed,seat,name)
    entropy,t,slot,greedy,root,w,f,opp,enemy=chosen
    model=root.distribution(slot,policy.ZERO);keys=sorted(model)
    p=np.array([model[k][0] for k in keys])
    rng=np.random.default_rng(np.random.SeedSequence([994371,seed,seat,t]))
    uniforms=rng.random(8);cdf=np.cumsum(p);cdf[-1]=1
    draws=[keys[int(np.searchsorted(cdf,v,side='right'))] for v in uniforms]
    signed=HERE/'training'/f'{seed}_{seat}.json.gz'
    zero=policy.key(greedy);rootsig=dict(binding=binding,source_sha256=h,t=t,seat=seat,seed=seed,
                  terrain=w.terrain_sha256,source=str(path.relative_to(ROOT)),slot=slot,opponent=name,
                  screened=profile,greedy=zero,root_public_sha256=hashlib.sha256(policy.key(w.obs(seat)).encode()).hexdigest(),
                  mass={k:float(v[0]) for k,v in model.items()},uniforms=uniforms.tolist(),draws=draws)
    if signed.exists():
        audit=read(signed);assert audit['root']==rootsig
    else:
        outcomes={k:suffix(module,seat,seed,w,f,opp,enemy,json.loads(k),opening) for k in sorted(set([zero,*draws]))}
        audit=dict(root=rootsig,outcomes=outcomes);write(signed,audit)
    trainrow=dict(seed=seed,seat=seat,t=t,terrain=w.terrain_sha256,samples=[
        dict(key=k,u=audit['outcomes'][k]['official']['u_by_seat'][seat],old_p=model[k][0]) for k in draws])
    return trainrow,(root,slot)


def build(delta):
    path=HERE.parent/'blended_actor_dev/serving/build.py'
    spec=importlib.util.spec_from_file_location('verified_packer',path);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    src,initial,expert=b.validate_baseline()
    dest=HERE/'candidate.npz'
    with np.load(b.SOURCE,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
        old=z['whead.bias'].copy();data['whead.bias']=old+np.asarray(delta,np.float32)
        if dest.exists():
            with np.load(dest,allow_pickle=False) as prev: assert all(np.array_equal(data[k],prev[k]) for k in data)
        else: np.savez_compressed(dest,**data)
        values=b.quantize(dest,z)
    generated=b.substitute(src,values)
    assert b.substitute(generated,initial)==b.substitute(src,initial)
    cp=HERE/'candidate.py';b.save(cp,generated.encode())
    deployed=policy.load(cp); assert np.array_equal(deployed._initial_bias,data['whead.bias'])
    return dict(parent_sha256=policy.PARENT_SHA,parent_weights_sha256=b.SOURCE_SHA,expert_sha256=expert,
                candidate_sha256=sha(cp),weights_sha256=sha(dest),parameter='whead.bias',old=old.tolist(),new=data['whead.bias'].tolist(),
                weights_changed=int(np.sum(data['whead.bias']!=old)),size_bytes=cp.stat().st_size)


def held_game(seed,seat,opponent,arm,binding,package):
    p=HERE/'development'/f'{arm}_{seed}_{seat}.json.gz'
    if p.exists():
        row=read(p);assert row['binding']==binding and row['package']==package;return row
    w=World(seed);opp=make_opponent(opponent);models={a:policy.load(q) for a,q in [('parent',policy.PACK),('candidate',HERE/'candidate.py')]}
    tape=[];changed=[]
    while w.terminal is None:
        obs=w.obs(seat);orders={a:m.agent(obs,CFG) for a,m in models.items()}
        if orders['parent']!=orders['candidate']:changed.append(w.step)
        en=opp(w.obs(1-seat),CFG);cmd=orders[arm]
        joint=[cmd,en] if seat==0 else [en,cmd];tape.append(joint);w.advance(joint)
    row=dict(seed=seed,seat=seat,arm=arm,binding=binding,package=package,opponent=opponent,
             common_history_changed=changed,actions=tape,official=verify(seed,tape,w.terminal))
    write(p,row);return row


def main():
    assert os.sched_getaffinity(0)=={3} and all(os.environ[k]=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    binding=bind();write(HERE/'binding.json',binding)
    maps=read(HERE/'map_declaration.json');module=policy.load();training=[];contexts={}
    for i,seed in enumerate(maps['training']):
        for seat in (0,1):
            r,ctx=collect(module,seed,seat,('economy','raid')[i%2],binding);training.append(r);contexts[seed,seat]=ctx
            print('received',seed,seat,r['t'],len({s['u'] for s in r['samples']}),flush=True)
    def distribution(row,delta):
        ctx,s=contexts[row['seed'],row['seat']];return ctx.distribution(s,delta)
    result=train(training,distribution);write(HERE/'update.json',result)
    package=build(result['delta']);write(HERE/'package.json',package)
    held=[]
    for i,seed in enumerate(maps['new_proxy_development']):
        for seat in (0,1):
            for arm in ('parent','candidate'):
                d=held_game(seed,seat,('economy','raid')[i],arm,binding,package)
                held.append(dict(seed=seed,seat=seat,arm=arm,opponent=d['opponent'],official=d['official'],
                                 changed_on_this_arm=len(d['common_history_changed'])))
                print('dev',arm,seed,seat,d['official']['margin_by_seat'][seat],flush=True)
    write(HERE/'result.json',dict(decision=result['decision'],held=held,package=package,training_sha256=sha(HERE/'update.json'),formal_games=0))


if __name__=='__main__':
    with (HERE/'.process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
