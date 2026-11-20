"""Inspect the deployed V4 root menu on six fixed archived public positions."""
import collections
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
GAME = '6e994d80674dc2430a4925b71bb6a947'
STEPS = (267,269,272,274,277,279)
PARENT = D/'advantage_policy_update/v2_bc.py'
IDENTITY = 'cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a'
scanpath = HERE.parent/'v4_alphacarbon_decisions/scan.py'
spec = importlib.util.spec_from_file_location('alphacarbon_public_rules', scanpath)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def sha(x):
    return hashlib.sha256(x).hexdigest()


def put(p, obj):
    raw = (json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    p.write_bytes(gzip.compress(raw,mtime=0) if p.suffix=='.gz' else raw)


def inspect_geometry(ps, order):
    own, enemy = ps
    banks = set(own[1].values())
    enemybanks = set(enemy[1].values())
    dest = {uid:s.move(p,order.get(uid,'NONE')) for uid,(p,_,_) in own[2].items()}
    counts = collections.Counter(dest.values())
    for uid,p in own[1].items():
        if order.get(uid,'NONE')!='NONE':
            counts[p]+=1
    routes, deposits = [], []
    for uid,(p,load,role) in own[2].items():
        if role!='COLLECTOR':continue
        q=dest[uid]
        if counts[q]>1:continue
        if load>0 and q in banks:
            deposits.append([uid,q,load])
        if q in enemybanks:continue
        for enemy_uid,(z,cargo,kind) in enemy[2].items():
            if kind!='COLLECTOR' or not load<cargo:continue
            # Route to a nearest bank only. The intercept cell must lie BEFORE that bank.
            d=min(s.dist(z,b) for b in enemybanks)
            valid=lambda v: any(s.dist(z,v)+s.dist(v,b)==d for b in enemybanks)
            if valid(q) and 1<=s.dist(z,q)<=3:
                routes.append([uid,enemy_uid,q,s.dist(z,q),load,cargo])
    return dict(corridors=routes,deposits=deposits)


def process(core):
    os.sched_setaffinity(0,{core})
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0,str(D/'opponent_policy/search_probe'))
    from full_search_check import root_call
    raw=PARENT.read_bytes()
    assert sha(raw)==IDENTITY
    m=types.ModuleType('original_v4');exec(compile(raw,'original_v4','exec'),m.__dict__)
    assert m._net() is not None and m._SEARCH_OK and m._S_ABORT==.58 and m._TEMP==1 and m._SEED is None
    card=s.read();frames=card['steps'];cfg={k:v for k,v in card['configuration'].items() if k!='randomSeed'}
    tracker=m.BeliefTracker(m._Bel,0);prev=None;cost=list(m._S_COST);protocol_seed=lambda t:int.from_bytes(hashlib.sha256(f'alphacarbon-menu/{GAME}/0/{t}'.encode()).digest()[:4],'big')
    measured={}
    for t,frame in enumerate(frames[:-1]):
        ps=frame[0]['observation']['players'];o=frame[0]['observation'];obs={k:o[k] for k in ('step','player','players','trees','carbon','remainingOverageTime')}
        if t in STEPS or t in [u-1 for u in STEPS]:
            paid=root_call(m,tracker,prev,obs,cfg,protocol_seed(t),cost)
            if t in STEPS:
                measured[t]=dict(public_input_sha=sha(json.dumps(obs,sort_keys=True).encode()),seed=protocol_seed(t),measured_initial_cost=cost,measurement=paid,
                                  before=[inspect_geometry(ps,a) for a in paid['menu']],parent_geometry=inspect_geometry(ps,paid['act0']))
            cost=paid['ending_costs']
        tracker.step(ps,t,obs['carbon']);tracker.set_actions(ps,frames[t+1][0]['action'] or {});prev=ps
    # Each native call has completed. Restore the referee's full state solely to label
    # those frozen menus against the simultaneous historical opponent command.
    sim=s.rule.econ.FastCarbon(card['configuration']).reset(full_carbon=frames[0][0]['observation']['full_carbon'])
    sim.cash=s.rule.econ.CashEvents(sim.cash)
    for t in range(299):
        true=[p['action'] or {} for p in frames[t+1]]
        if t in measured:
            r=measured[t];j=r['measurement'];r['actual_own']=true[0];r['actual_enemy']=true[1];r['actual_physics']=s.after(sim,true)[1]
            r['menu_physics']=[s.after(sim,[a,true[1]])[1] for a in j['menu']]
            r['root_physics']=s.after(sim,[j['act0'],true[1]])[1]
            r['team_equals_recorded']=[i for i,a in enumerate(j['menu']) if a==true[0]]
        sim.cash.events.clear();sim.step(*true)
        assert sim.obs(0)['players']==frames[t+1][0]['observation']['players'] and list(sim.carbon)==frames[t+1][0]['observation']['full_carbon']
    put(HERE/'evidence.json.gz',dict(identity=IDENTITY,match=GAME,replay_sha256=sha(s.SRC.read_bytes()),step_ids=STEPS,cpu=core,rows=measured))


if __name__=='__main__':
    process(int(sys.argv[1]))
