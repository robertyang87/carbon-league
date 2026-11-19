"""Fit the state-conditioned Sapling worker head on archived decision windows with frozen native-strength self-owned opponent."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from common import HERE, OLD, ROOT, World, assert_cpu, encode, finish, net, open_state, policy, read, sha, tapes, write, decision
from conditioned import Conditioning
from decoder import Root,apply,instrument

ZERO=np.zeros(15)
POINTS=(205,271)
HIDDEN=(60,110,155,205,238,271)


def context(m,seed,original_seat,t):
    state,ff,opening,source=open_state(seed,original_seat,t)
    seat=original_seat if t<=205 else 1-original_seat
    x=ff[seat].observe(state.obs(seat))
    return Root(m,ff[seat],x),state,ff,opening,source,seat


def projection(m,plan,binding):
    archive=HERE/'projection_inputs.npz';receipt=HERE/'projection_receipt.json'
    if receipt.exists():
        saved=read(receipt);assert sha(archive)==saved['npz_sha256'] and saved['binding']==binding
        with np.load(archive) as data:
            h,y,ids=data['h'],data['y'],data['terrain'].tolist()
    else:
        h=[];y=[];ids=[];rows=[]
        for seed in plan['training']:
            for srcseat in (0,1):
                for t in HIDDEN:
                    root,w,ff,opening,source,seat=context(m,seed,srcseat,t)
                    _,details=root.inspect(True)
                    for slot,(_,hidden) in details.items():
                        if slot>=10:continue
                        h.append(hidden.copy());y.append(root.physical(slot));ids.append(source['terrain'])
                        rows.append(dict(seed=seed,source_seat=srcseat,seat=seat,t=t,slot=slot,uid=root.f.own_uid[slot],
                                         role=int(root.x['tok_kind'][slot]),source_sha256=source['sha256']))
        tmp=archive.with_name('inputs.partial.npz')
        np.savez_compressed(tmp,h=np.stack(h),y=np.asarray(y),terrain=np.asarray(ids));tmp.replace(archive)
        write(receipt,dict(binding=binding,rows=rows,npz_sha256=sha(archive)))
    model=Conditioning.fit(h,y,ids);model.write(HERE/'projection.json');return model


def root_work(m,foe,project,binding,seed,ix,original_seat,k):
    t=POINTS[k];apply(m,None,ZERO)
    root,w,ff,opening,source,seat=context(m,seed,original_seat,t)
    choice=root.choose((ix+original_seat+k)%2==0)
    header=dict(binding=binding,terrain=w.terrain_sha256,seed=seed,seat=seat,original_seat=original_seat,t=t,source=source,choice=choice)
    p=HERE/'train'/f'{seed}_{original_seat}_{t}.json.gz'
    if choice is None:
        write(p,dict(header=header,skipped=True));return None
    slot=choice['slot'];law,segments=root.mass(slot,project,ZERO)
    uniforms=np.random.default_rng(np.random.SeedSequence([8721103,seed,seat,t])).random(6)
    sampled=[]
    for u in uniforms:
        matches=[(key,rollback) for low,high,key,rollback in segments if low<=u<high]
        assert len(matches)==1;kkey,rollback=matches[0]
        sampled.append(dict(key=kkey,common_uniform=float(u),rollback=bool(rollback),old_p=float(law[kkey][0])))
    expected=dict(header=header,samples=sampled,executed_mass={key:float(q) for key,(q,_) in law.items()},intervals=[list(z) for z in segments])
    if p.exists():
        record=read(p);assert record['state']==expected
    else:
        nets=[m,foe] if seat==0 else [foe,m]
        enemy=decision(foe,ff[1-seat],w.obs(1-seat));out={}
        for key in sorted({r['key'] for r in sampled}|{choice['greedy']}):
            own=json.loads(key);first=[own,enemy] if seat==0 else [enemy,own]
            out[key]=finish(seed,opening,w,ff,nets,first=first)
        record=dict(state=expected,outcomes=out);write(p,record)
    trainrow=dict(seed=seed,seat=seat,t=t,terrain=w.terrain_sha256,samples=[])
    for r in sampled:
        o=record['outcomes'][r['key']]['official']
        trainrow['samples'].append(dict(key=r['key'],old_p=r['old_p'],u=o['u_by_seat'][seat],margin=o['margin_by_seat'][seat]))
    return trainrow,root,slot


def make_pack(project,delta,training):
    path=OLD.parent/'blended_actor_dev/serving/build.py'
    spec=importlib.util.spec_from_file_location('return_packer',path)
    server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
    src,orig,expert=server.validate_baseline();dst=HERE/'candidate.npz';pack=HERE/'candidate.py'
    with np.load(server.SOURCE) as ref:
        state=project.inject(ref,delta)
        if dst.exists():
            with np.load(dst) as old:assert all(np.array_equal(old[k],v) for k,v in state.items())
        else: np.savez_compressed(dst,**state)
        values=server.quantize(dst,ref)
        changed={k:int(np.count_nonzero(ref[k]!=v)) for k,v in state.items() if not np.array_equal(ref[k],v)}
    assert set(changed)<={'whead.weight','whead.bias'}
    out=server.substitute(src,values)
    assert server.substitute(out,orig)==server.substitute(src,orig)
    server.save(pack,out.encode());real=policy.load(pack)
    assert all(np.array_equal(real._NETS[1].w[k],state[k]) for k in ('whead.weight','whead.bias'))
    return dict(parent_sha256=policy.PARENT_SHA,source_weight_sha256=server.SOURCE_SHA,expert_sha256=expert,
                weights_sha256=sha(dst),candidate_sha256=sha(pack),changed=changed,train_sha256=sha(training),
                projection_sha256=sha(HERE/'projection.json'),free_parameters=15)


def main():
    from optimizer import optimize
    assert_cpu()
    f=read(HERE/'protocol.json');s=read(HERE/'screen.json');op=s['selected']
    assert op and f['screen_sha256']==sha(HERE/'screen.json') and f['maps_sha256']==sha(HERE/'maps.json') and f['opponent']==op
    current={k:sha(HERE/k) for k in ('maps.json','protocol.json','screen.json','worlds.py','common.py','conditioned.py','decoder.py','optimizer.py','run.py')}
    write(HERE/'binding.json',current)
    maps=read(HERE/'maps.json');m=instrument(net('it20_self'));foe=net(op)
    projection_model=projection(m,maps,current);rows=[];ctx={}
    for i,seed in enumerate(maps['training']):
        for orig in (0,1):
            for window in (0,1):
                got=root_work(m,foe,projection_model,current,seed,i,orig,window)
                if got:
                    row,r,s=got;rows.append(row);ctx[seed,row['seat'],row['t']]=r,s
                print('registered training root',seed,orig,POINTS[window],bool(got),flush=True)
    def law(row,delta):
        r,slot=ctx[row['seed'],row['seat'],row['t']]
        return r.mass(slot,projection_model,delta)[0]
    report=optimize(rows,law);training=HERE/'update.json';write(training,report)
    delta=report['delta'];package=make_pack(projection_model,delta,training)
    write(HERE/'package.json',package)
    # Read the literal deployed tensors from the generated standalone package and verify old/new
    # execution mass at every training root, including all original supports.
    packed=policy.load(HERE/'candidate.py')
    for k in ('whead.bias','whead.weight'):
        expected=projection_model.inject(m._orig_head,delta)[k]
        assert np.array_equal(expected,packed._NETS[1].w[k])
    print('signed',report['decision'],package['candidate_sha256'],flush=True)
    import development
    development.run(current)


if __name__=='__main__':
    with (HERE/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
