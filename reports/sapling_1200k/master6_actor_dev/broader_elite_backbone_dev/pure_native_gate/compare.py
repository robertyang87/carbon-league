"""Compare the two unwrapped Sapling actors on exact public states and short original-response pairs."""
import base64
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import sys
import zlib

import numpy as np

HERE=Path(__file__).resolve().parent;T=HERE.parent;A=T.parent;ROOT=A.parents[2]
sys.path.insert(0,str(T/'native_gate'))
import gate as h
r=h.r
p=r.load(HERE/'protocol.json')
packs={'parent':ROOT/p['parent_path'],'candidate':T/'standalone.py'}


def bind():
    hp=h.binding(); assert r.sha(T/'recipe.json')==p['recipe_sha256'] and r.sha(T/'panel.json')==p['panel_sha256']
    new=h.artifacts(packs['candidate']);old=h.artifacts(packs['parent'])
    assert new[0]==old[0] and new[1]['_W64']!=old[1]['_W64'] and r.sha(packs['parent'])==p['parent']
    assert h.artifacts(T/'candidate.py')[1]==new[1], "new hybrid and pure models have different student assets"
    with np.load(T/'candidate.npz',allow_pickle=False) as z:
        n=sum(z[k].size for k in z.files if k!='meta'); assert n==1206476
    for name,f in packs.items():
        m=r.live.module(f,'verify_only_'+name)
        assert m._OPTS==dict(teacher=2,shield=False,nocollide=False,endgame=False,pshield=False,rguard=False)
        assert not hasattr(m,'_B1') and not hasattr(m,'_HYBRID') and len(m._SRC)==3
    panel=r.load(T/'panel.json')['games']
    for c in p['held_roots']+p['exposures']:
        assert all(z['seed']!=c['seed'] and z['terrain_sha256']!=c['terrain'] for z in panel)
    v=dict(protocol=r.sha(HERE/'protocol.json'),hybrid_shared_student_binding=r.sha(T/'native_gate/binding.json'),old_pack=p['parent'],new_pack=r.sha(packs['candidate']),
           parameters=n,old_checkpoint=p['old_checkpoint'],new_checkpoint=hp['files']['candidate.npz'],common_code_after_removing_weight_literals_sha256=hashlib.sha256(new[0].encode()).hexdigest(),
           same_runtime=True,teacher=2,guards=False,echo_weights=False,search=False)
    r.write(HERE/'binding.json',v);return v


def raw(c):
    f=ROOT/c['source'];g=r.load(f);assert r.sha(f)==c['source_sha256'] and g['initial_terrain_u8_sha256']==c['terrain'] and g['pack_sha256']==c['logging_policy_sha256'] and g['candidate_seat']==c['seat']
    assert g['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and g['steps']==299 and not g['errors']
    return g


def idle(m,v):
    # With all optional guards off the only inter-turn state is this observation history.
    st=inspect.getclosurevars(m._AG).nonlocals['st']
    assert set(st)=={'f'}
    st['f'].observe(v)


def observe(c,held):
    g=raw(c);s=c['seat'];sim=r.FastCarbon(g['game_configuration']).reset(seed=c['seed'])
    mods={n:r.live.module(f,'pure_compare_'+n) for n,f in packs.items()}
    keep=set(range(0,271,30))|{297} if held else {0}|set(range(270,286));records=[]
    for t,joint in enumerate(g['actions']):
        v=sim.obs(s)
        if t in keep:
            ds={n:r.live.clean(m.agent(v,g['game_configuration'])) for n,m in mods.items()}
            if c['prefix_type']=='pure':assert ds['parent']==r.live.clean(joint[s])
            diff={k:dict(type='base' if k in v['players'][s][1] else v['players'][s][2][k][2],parent=ds['parent'].get(k,'NONE'),candidate=ds['candidate'].get(k,'NONE')) for k in set(ds['parent'])|set(ds['candidate']) if ds['parent'].get(k)!=ds['candidate'].get(k)}
            rec=dict(t=t,**ds,changes=diff,opponent=joint[1-s])
            if held:records.append(rec)
            elif t>=270 and (diff or t==285):
                return dict(**c,**rec)
        else:
            for m in mods.values():idle(m,v)
        sim.step(*joint)
        if t==max(keep):break
    if not held:raise ValueError('precommitted diagnostic root unavailable')
    return dict(**c,checked=records)


def play(job):
    c,name,cpu,warm=job;os.sched_setaffinity(0,{cpu});s=c['seat'];op=1-s;t0=c['t'];g=raw(c)
    out=HERE/f"response_{c['seed']}_{s}_{name}.json.gz"
    b=dict(protocol=r.sha(HERE/'protocol.json'),binding=r.sha(HERE/'binding.json'),cpu=cpu,warm=warm,arm=name,source=c['source_sha256'])
    if out.exists():
        doc=r.load(out);assert doc['binding']==b and doc['statuses']==['DONE','DONE'];return doc
    env=r.live.begin(g);own=r.live.module(packs[name],'native_pure_'+name)
    master=r.live.module(r.live.SOURCE,'original_pure_'+name)
    assert r.sha(r.live.SOURCE)==p['original'] and master._SEED is None and master._TEMP==1 and master._S_ABORT==.58 and master._RX and master._SEARCH_OK
    tr=master.BeliefTracker(master._Bel,op)
    for t,joint in enumerate(g['actions'][:t0+1]):
        v=r.live.public(env,s);other=r.live.public(env,op);tr.step(other['players'],t,other['carbon']);tr.set_actions(other['players'],joint[op])
        if t in (0,t0):a=r.live.clean(own.agent(v,g['game_configuration']))
        else:idle(own,v)
        if t<t0:env.step(joint)
    assert a==c[name] and joint[op]==c['opponent']
    rng=int(hashlib.sha256(f"pure-final|{c['seed']}|{s}|{t0}".encode()).hexdigest()[:8],16)
    master._S_COST[:]=warm;master._ST.clear();master._ST.update(tr=tr,prev=other['players'],rng=np.random.RandomState(rng))
    recorder=r.native_clock.Recorder(master);agent=recorder.wrap(master.agent)
    opening=list(joint);opening[s]=a;tape=[opening];tree=[h.trees(env)];env.step(opening);tree.append(h.trees(env))
    while not env.done:
        q=[None,None];q[s]=own.agent(r.live.public(env,s),g['game_configuration']);q[op]=agent(r.live.public(env,op),g['game_configuration']);env.step(q);tape.append(q);tree.append(h.trees(env))
    score=[z.reward for z in env.states];statuses=[z.status for z in env.states]
    assert statuses==['DONE','DONE'] and len(recorder.calls)==len(tape)-1>0 and not recorder.errors
    doc=dict(binding=b,t=t0,rng=rng,statuses=statuses,score_by_seat=score,margin=score[s]-score[op],tree_control=tree,
        cash=r.cash_summary(g,s,t0,tape,[v[0] for v in env.states[0].observation.players]),tape=tape,calls=recorder.calls,
        clock=r.native_clock.summarize(recorder.calls,master._S_ABORT,master._S_D2))
    r.write(out,doc);return doc


def run():
    b=bind();panel=[observe(c,True) for c in p['held_roots']]
    num=sum(bool(z['changes']) for g in panel for z in g['checked']);assert num>0
    r.write(HERE/'held_actions.json.gz',dict(arms_binding=r.sha(HERE/'binding.json'),changed=num,roots=sum(len(g['checked']) for g in panel),cases=panel))
    cases=[observe(c,False) for c in p['exposures']]
    r.write(HERE/'response_roots.json',dict(binding=r.sha(HERE/'binding.json'),cases=cases))
    rows=[]
    with r.LOCK.open('a') as slot:
        fcntl.flock(slot,fcntl.LOCK_EX)
        for c in cases:
            s=c['seat'];wf=HERE/f"warm_{c['seed']}_{s}.json"
            if not wf.exists():
                with ProcessPoolExecutor(1) as pool:warm=pool.submit(r.native_warm).result()
                r.write(wf,dict(warm=warm,original=p['original'],binding=r.sha(HERE/'binding.json')))
            w=r.load(wf);assert w['binding']==r.sha(HERE/'binding.json')
            with ProcessPoolExecutor(2) as pool:a,b=pool.map(play,[(c,'parent',s,w['warm']),(c,'candidate',1-s,w['warm'])])
            assert a['rng']==b['rng'] and a['binding']['warm']==b['binding']['warm']
            rows.append(dict(seed=c['seed'],seat=s,t=c['t'],prefix_type=c['prefix_type'],changed=c['changes'],margins=[a['margin'],b['margin']],delta=b['margin']-a['margin'],
              original_calls=[len(z['calls']) for z in (a,b)],enemy_order_differences=sum(x[1-s]!=y[1-s] for x,y in zip(a['tape'],b['tape'])),
              delta_tree=b['cash']['tree_income_margin']-a['cash']['tree_income_margin'],delta_delivery=b['cash']['delivery_margin']-a['cash']['delivery_margin']))
            print('PURE ROOT',rows[-1]['seed'],rows[-1]['margins'],rows[-1]['delta'],flush=True)
    r.write(HERE/'responses.json',dict(held_actual_changed=num,registrations=r.sha(HERE/'protocol.json'),rows=rows,
        score_direction_filters_fresh_match=False))


if __name__=='__main__':run()
