"""Intervene at one frozen team decision; both sides make every subsequent real move."""
import copy,gzip,json,os,sys,zipfile
from pathlib import Path
from register import HERE,FAMILY,fp
sys.path.insert(0,str(HERE.parent))
from probe import rr,base,np

P='767b40d67ebb077789859a40939cae310e5efbc4e82083c25063306c982e73c8'
def acquire(ticket,plan):
    seed,s,t=ticket['seed'],ticket['seat'],ticket['step'];prefix=f'{seed}_{t}';foe=1-s
    source=FAMILY/'cold_start/development/hosted'/(ticket['archive']+'.zip');assert fp(source.read_bytes())==ticket['archive_sha256']
    with zipfile.ZipFile(source) as z:
        frames=gzip.decompress(z.read('cold_repair/observations.jsonl.gz')).splitlines()
        tape=z.read('cold_repair/turns.jsonl');assert fp(tape)==json.loads(z.read('cold_repair/receipt.json'))['turns_sha256']
    recorded=tape.splitlines()
    cfg=base.econ.config();cfg.pop('randomSeed',None);np.random.seed(seed)
    env=base.make('carbon',configuration={**cfg,'randomSeed':seed});sim=base.econ.FastCarbon(cfg).reset(seed=seed)
    assert fp(bytes(env.states[0].observation.full_carbon))==ticket['terrain']==fp(bytes(map(int,sim.carbon)))
    packs=[FAMILY.parents[1]/'data/master3s2_source.py',FAMILY/'cold_start/candidate.py']
    assert s==1 and [fp(f.read_bytes()) for f in packs]==[plan['original'],plan['parent']]
    models=[base.load(f'{prefix}_seat{i}',f) for i,f in enumerate(packs)]
    assert all(m._S_ABORT==.58 and m._TEMP==1. and m._SEED is None and m._RX and m._SEARCH_OK for m in models)
    trackers=[m.BeliefTracker(m._Bel,i) for i,m in enumerate(models)]
    prev=None;bank=[5.,5.];costs=[[0.]*4 for m in models];warm=[]
    for i in range(t+1):
        pub=[base.public(env,k,bank) for k in (0,1)]
        record=json.loads(recorded[i]);f=json.loads(frames[i]);hh=[base.digest(x) for x in pub]
        assert hh==record['public_observation_sha256']==f['public_observation_sha256'] and pub==f['observations']
        if i==t:break
        acts=[q['response']['action'] for q in record['calls']];players=pub[0]['players']
        for k,m in enumerate(models):
            if i>=t-4:
                receipt=rr.call(m,(trackers[k],prev),pub[k],cfg,(seed+i*193+k)%2**32,costs[k],'P')
                costs[k]=list(receipt['clocks']['cost']);warm.append(dict(t=i,seat=k,clock=receipt['clocks']))
            trackers[k].step(players,i,pub[k]['carbon']);trackers[k].set_actions(players,acts[k])
        prev=players;bank=record['overage_remaining_seconds'];env.step(acts,[]);sim.step(*acts)
        assert list(sim.cash)==[v[0] for v in env.states[0].observation.players]
    snap=base.official_state.take(env,bank);shadow=copy.deepcopy(sim);source_root=json.loads(recorded[t])
    costs[s]=ticket['empirical_costs']
    for si,stream in enumerate(ticket['streams']):
        bank=base.official_state.restore(env,snap)
        common=[rr.call(m,(trackers[k],prev),pub[k],cfg,stream['prng'][k],costs[k],'P') for k,m in enumerate(models)]
        enemy=source_root['calls'][foe]['response']['action'] if si==0 else common[foe]['action']
        snapshots=[rr.snapshot(m) for m in models]
        hashes=[base.original_state.h(m._ST) for m in models]
        result={}
        for arm in stream['order']:
            bank=base.official_state.restore(env,snap);sim=copy.deepcopy(shadow);sim.cash=base.econ.CashEvents(sim.cash)
            for k,m in enumerate(models):rr.restore(m,snapshots[k]);assert base.original_state.h(m._ST)==hashes[k]
            our=ticket['orders'][arm]
            models[s]._ST['tr'].set_actions(pub[s]['players'],our);models[foe]._ST['tr'].set_actions(pub[foe]['players'],enemy)
            rows=[]
            while not env.done:
                ob=[base.public(env,k,bank) for k in (0,1)];step=ob[0]['step']
                if step==t:pair=[(enemy,common[foe]['clocks']),(our,common[s]['clocks'])]
                else:pair=[base.measured(m,ob[k],cfg) for k,m in enumerate(models)]
                acts=[x[0] for x in pair];clocks=[x[1] for x in pair]
                bank=[b-max(0,c['wall']-1.) for b,c in zip(bank,clocks)]
                sim.cash.events.clear();sim.step(*acts);env.step(acts,[])
                authoritative=env.states[0].observation
                assert authoritative.players==sim.obs(0)['players'] and authoritative.trees==sim.obs(0)['trees'] and authoritative.full_carbon==list(sim.carbon)
                rows.append(dict(step=step,public=[base.digest(x) for x in ob],actions=acts,clocks=clocks,cash=list(sim.cash),events=list(sim.cash.events)))
            scores=[v.reward for v in env.states]
            d=dict(plan=P,seed=seed,seat=s,t=t,stream=stream,arm=arm,maps=ticket['terrain'],opponent_root=enemy,root_private=hashes,
                   source_reference_root=[x['response']['action'] for x in source_root['calls']],native_root=[x['action'] for x in common],source_root_hashes=hh,
                   calibration=warm,final_step=env.states[0].observation.step,official_scores=scores,
                   margin=round(scores[s]-scores[foe],2),status=[v.status for v in env.states],rows=rows)
            file=HERE/f'{prefix}_{si}_{arm}.json.gz';assert not file.exists()
            rr.save(file,d);result[arm]=dict(file=file.name,sha256=fp(file.read_bytes()),official_scores=scores,margin=d['margin'])
        print(prefix,stream['kind'],result['stock']['margin'],result['home']['margin'],round(result['home']['margin']-result['stock']['margin'],2),flush=True)
if __name__=='__main__':
    assert os.sched_getaffinity(0)=={3}
    import torch
    torch.set_num_threads(1);os.environ['CARBON_DEBUG']='1'
    raw=(HERE/'protocol.json').read_bytes();assert fp(raw)==P
    job=json.loads(raw)
    for ticket in job['tickets']:acquire(ticket,job)
