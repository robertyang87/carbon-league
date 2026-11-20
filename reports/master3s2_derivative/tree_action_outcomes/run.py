"""Run an archived, frozen one-order tree intervention with natural native responses."""
import copy
import gzip
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
import zipfile
from register import HERE,D as FAMILY,sha
from order import Anchor

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

rr=load('response_tools',FAMILY/'return_rank/development/root_responses.py')
base=rr.base
PROTOCOL='1cd2c0e45c5838386a2f7c833bc98a7ed1e4eeca47ee7e1bc904dfbc9ab0eee9'

def run(ticket,plan):
    t,s,seed=ticket['step'],ticket['seat'],ticket['seed']
    file=FAMILY/ticket["archive_relpath"]
    assert sha(file.read_bytes())==ticket['archive_sha256']
    with zipfile.ZipFile(file) as z:
        src=z.read('parent/observations.jsonl.gz');turns=z.read('parent/turns.jsonl')
        reg=json.loads(z.read('parent/registration.json'));rec=json.loads(z.read('parent/receipt.json'))
        assert sha(src)==ticket['public_stream_sha256'] and sha(turns)==rec['turns_sha256']
        assert reg['files']['opponent']['sha256']==plan['original'] and reg['files']['candidate']['sha256']==plan['candidate']
        frames=gzip.decompress(src).splitlines()[:t+1]
        cmds=turns.splitlines()[:t+1]
    cfg=base.econ.config();cfg.pop('randomSeed',None)
    base.np.random.seed(seed)
    env=base.make('carbon',configuration={**cfg,'randomSeed':seed})
    sim=base.econ.FastCarbon(cfg).reset(seed=seed)
    assert sha(bytes(env.states[0].observation.full_carbon))==ticket['terrain']==sha(bytes(map(int,sim.carbon)))
    original=FAMILY.parents[1]/'data/master3s2_source.py'
    models=[None,None];paths=[None,None];paths[s]=FAMILY/'cold_start/candidate.py';paths[1-s]=original
    for i,p in enumerate(paths):
        assert sha(p.read_bytes())==plan['candidate' if i==s else 'original']
        models[i]=base.load(f'paid_tree_{seed}_seat{i}',p)
        assert models[i]._TEMP==1 and models[i]._SEED is None and models[i]._S_ABORT==.58 and models[i]._RX and models[i]._SEARCH_OK
    bank=[5.,5.];calibration=[];history=[]
    for turn,frame in enumerate(frames):
        frame=json.loads(frame);views=[base.public(env,i,bank) for i in (0,1)]
        hs=[base.digest(o) for o in views]
        assert frame['observations']==views and frame['public_observation_sha256']==hs
        assert list(sim.cash)==[p[0] for p in views[0]['players']]
        history.append(hs)
        if turn==t:break
        record=json.loads(cmds[turn]);assert record['public_observation_sha256']==hs
        action=[c['response']['action'] for c in record['calls']]
        for i,m in enumerate(models):
            players=views[i]['players']
            if turn<3:
                _,clock=base.measured(m,views[i],cfg);calibration.append(dict(t=turn,seat=i,clock=clock))
            else:m._ST['tr'].step(players,turn,views[i]['carbon'])
            m._ST['tr'].set_actions(players,action[i]);m._ST['prev']=players
        sim.step(*action);env.step(action,[]);bank=record['overage_remaining_seconds']
    assert hs==ticket['public_root_hashes'] and t==env.states[0].observation.step
    issued=json.loads(cmds[t])['calls'][s]['response']['action']
    actual=ticket['stay'] if ticket['historical']=='NONE' else ticket['leave']
    alternative=ticket['leave'] if ticket['historical']=='NONE' else ticket['stay']
    assert actual==issued
    single=Anchor(views[s],issued,ticket['uid'],cfg)
    assert single.audit(alternative)['accepted'] and single.audit(alternative)['changed']==ticket['focal_delta']
    for i,m in enumerate(models):m._ST['rng']=base.np.random.RandomState(ticket['rng'][i])
    # The signed packets for both modes already exist. Compute an independent new native
    # reply before applying either intervention, then keep each actor's stream unchanged across arms.
    common=[base.measured(m,views[i],cfg) for i,m in enumerate(models)]
    stamp=[base.original_state.h(m._ST) for m in models]
    snaps=[rr.snapshot(m) for m in models]
    official=base.official_state.take(env,bank);shadow=copy.deepcopy(sim)
    root=dict(protocol=PROTOCOL,ticket=ticket,public_root_by_seat=views,historical_prefix_sha256=base.original_state.h(history),
              private_postcall_state_by_seat=stamp,local_root_calls=common,calibration=calibration)
    rr.save(HERE/f'{seed}_root.json.gz',root)
    result={}
    for arm in ticket['execution_order']:
        bank=base.official_state.restore(env,official);sim=copy.deepcopy(shadow);sim.cash=base.econ.CashEvents(sim.cash)
        for i,m in enumerate(models):rr.restore(m,snaps[i]);assert base.original_state.h(m._ST)==stamp[i]
        models[s]._ST['tr'].set_actions(views[s]['players'],ticket[arm])
        rows=[];wallet=Counter()
        while not env.done:
            public=[base.public(env,i,bank) for i in (0,1)];step=public[0]['step']
            p=[(ticket[arm],common[s][1]) if i==s else common[i] for i in (0,1)] if step==t else [base.measured(m,public[i],cfg) for i,m in enumerate(models)]
            actions=[x[0] for x in p];clocks=[x[1] for x in p]
            if step==t:assert actions[1-s]==common[1-s][0]
            bank=[v-max(0,c['wall']-1.) for v,c in zip(bank,clocks)]
            sim.cash.events.clear();sim.step(*actions);env.step(actions,[])
            obs=env.states[0].observation;authority=sim.obs(0)
            assert obs.players==authority['players'] and obs.full_carbon==list(sim.carbon) and obs.trees==authority['trees']
            focal=[dict(seat=i,tree=k,payload=v) for i,pl in enumerate(obs.players) for k,v in pl[3].items() if v[0]==ticket['cell']]
            for cate,who,amount,*_ in sim.cash.events:wallet[(who,cate)]+=amount
            rows.append(dict(step=step,public_sha256=[base.digest(x) for x in public],actions=actions,clocks=clocks,
                             wallets=list(sim.cash),flows=list(sim.cash.events),tree_on_root_cell=focal,
                             focal_planter=obs.players[s][2].get(ticket['uid'])))
        sc=[v.reward for v in env.states];wall=[c['wall'] for r in rows for c in r['clocks']]
        r=dict(protocol=PROTOCOL,seed=seed,seat=s,arm=arm,root=t,status=[v.status for v in env.states],
               scores=sc,official_margin=round(sc[s]-sc[1-s],2),root_enemy=common[1-s][0],bank_end=bank,
               steps=len(rows),final_t=env.states[0].observation.step,max_clock=max(wall),slow_calls=sum(w>1 for w in wall),
               hard_calls=sum(w>3 for w in wall),flows_by_seat=[{cate:amount for (who,cate),amount in wallet.items() if who==i} for i in (0,1)],rows=rows)
        dest=HERE/f'{seed}_{arm}.json.gz';assert not dest.exists()
        rr.save(dest,r);result[arm]={k:v for k,v in r.items() if k not in ('rows','root_enemy','flows_by_seat')}
        result[arm].update(sha256=sha(dest.read_bytes()),focal_tree_after_action=rows[0]['tree_on_root_cell'],focal_after_action=rows[0]['focal_planter'])
        print(seed,arm,sc,r['official_margin'],r['max_clock'],flush=True)
    rr.save(HERE/f'{seed}_result.json',dict(protocol=PROTOCOL,root_sha256=sha((HERE/f'{seed}_root.json.gz').read_bytes()),arms=result,
           delta_stay_minus_leave=round(result['stay']['official_margin']-result['leave']['official_margin'],2)))
if __name__=='__main__':
    os.environ['CARBON_DEBUG']='1'
    assert os.sched_getaffinity(0)=={4}
    import torch
    torch.set_num_threads(1)
    raw=(HERE/'protocol.json').read_bytes();assert sha(raw)==PROTOCOL
    plan=json.loads(raw)
    assert len(plan['tickets'])==4
    for ticket in plan['tickets']:
        if any(HERE.glob(f"{ticket['seed']}_*")):raise ValueError('Registered root already started')
    for ticket in plan['tickets']:
        run(ticket,plan)
