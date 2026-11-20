"""One exposed official root, two new native response streams, four complete suffixes."""
import collections
import copy
import gzip
import importlib.util
import json
import os
import random
from pathlib import Path

from register import HERE, FAMILY, ROOT, RAW, FREIGHT, V4, ORIGINAL, IDS, encode, register, require, save, sha


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def physical(env,sim,reference=None):
    shared=env.states[0].observation
    for side in (0,1):
        observed=sim.obs(side)
        require(observed['players']==shared.players and observed['trees']==shared.trees and
                observed['carbon']==env.states[side].observation.carbon and
                [p[0] for p in shared.players]==list(sim.cash) and list(sim.carbon)==shared.full_carbon,
                'Official and fast map or wallets differ')
        if reference:
            own=reference[side]['observation']; common=reference[0]['observation']
            require(common['step']==shared.step and common['players']==shared.players
                    and common['trees']==shared.trees and common['full_carbon']==shared.full_carbon
                    and own['carbon']==observed['carbon'] and own['player']==side,'Different archived public prefix')


def advance(env,sim,orders):
    sim.cash.events.clear();sim.step(*orders);env.step(orders,[]);physical(env,sim)


def prefix(family,game,streams,start):
    cfg=game['configuration'];frames=game['steps']
    random.seed(cfg['randomSeed']);family.np.random.seed(cfg['randomSeed'])
    env=family.make('carbon',configuration=cfg)
    terrain=frames[0][0]['observation']['full_carbon']
    env.states[0].observation.full_carbon=copy.deepcopy(terrain)
    sim=family.econ.FastCarbon(cfg).reset(full_carbon=terrain);sim.cash=family.econ.CashEvents(sim.cash)
    mods=[load('paired_v4',V4),load('paired_native_original',ORIGINAL)]
    for m in mods:
        require(m._SEARCH_OK and m._S_ABORT==.58 and m._TEMP==1 and m._SEED is None and m._net() is not None,
                'Unexpected original policy or timer')
    trackers=[m.BeliefTracker(m._Bel,i) for i,m in enumerate(mods)];prev=None;order_hashes=[];obs_hashes=[]
    for t in range(start+1):
        physical(env,sim,frames[t])
        bank=[r['observation']['remainingOverageTime'] for r in frames[t]]
        obs=[family.public(env,s,bank) for s in (0,1)]
        obs_hashes.append([family.digest(o) for o in obs])
        if t==start: break
        orders=[r['action'] or {} for r in frames[t+1]];order_hashes.append(orders)
        if t==start-1:
            for s,m in enumerate(mods):
                m._ST=dict(tr=copy.deepcopy(trackers[s]),prev=prev,rng=family.np.random.RandomState(streams[s]^0x20a9e8))
                family.measured(m,obs[s],cfg)
        for s,tracker in enumerate(trackers):
            tracker.step(obs[s]['players'],t,obs[s]['carbon']);tracker.set_actions(obs[s]['players'],orders[s])
        prev=obs[0]['players'];advance(env,sim,orders)
    require(sha(encode(order_hashes))==PLAN['prefix_orders'],'Historical joint orders differ')
    env.steps[:-1]=[None]*start;env.logs=[]
    for s,m in enumerate(mods):
        m._ST=dict(tr=trackers[s],prev=prev,rng=family.np.random.RandomState(streams[s]))
    require(list(sim.cash)==PLAN['root_cash'] and all(obs[0]['players'][s][2].get(u)==PLAN['root_units'][u] for s,u in enumerate(IDS)),'Root workers differ')
    root_info=dict(checked_history_states=len(obs_hashes),history_public_sha256=sha(encode(obs_hashes)),prefix_orders_sha256=PLAN['prefix_orders'],
                   public_sha256=obs_hashes[-1],cash=list(sim.cash),time_bank=bank,
                   dynamic_hash=[family.original_state.h(family.original_state.dynamic(m)) for m in mods],
                   sampled_states=[family.original_state.h(m._ST['rng']) for m in mods],initial_cost=[list(m._S_COST) for m in mods])
    # Share the simultaneous BEFORE-intervention pair, including the native decisions and timesteps spent in search.
    root_pair=[family.measured(m,obs[s],cfg) for s,m in enumerate(mods)]
    env_snapshot=family.official_state.take(env,bank)
    # Each snapshot implementation enforces the exact file for its own player.
    snapshots=[]
    for i,(m,source) in enumerate(zip(mods,(V4,ORIGINAL))):
        adapter=load('snapshot_native_role_'+str(i),Path(family.original_state.__file__))
        adapter.SOURCE=PLAN['executables'][('v4','official_original_copy')[i]]['sha256']
        require(sha(source.read_bytes())==adapter.SOURCE,'Incorrect snapshot policy')
        snapshots.append((adapter,adapter.take(m)))
    return cfg,env,sim,mods,root_info,env_snapshot,snapshots,root_pair


def branch(family,state,ticket,arm):
    cfg,env,original_sim,mods,history,env_snap,snaps,root_pair=state
    bank=family.official_state.restore(env,env_snap);sim=copy.deepcopy(original_sim)
    for m,(adapter,snap) in zip(mods,snaps):adapter.restore(m,snap)
    root=PLAN['step'];rows=[];cash_totals=[collections.Counter(),collections.Counter()]
    stop=299
    while not env.done:
        obs=[family.public(env,s,bank) for s in (0,1)];t=obs[0]['step']
        require(t<stop,'Exceeded official step cap')
        p=obs[0]['players']
        before=[p[s][2].get(uid) for s,uid in enumerate(IDS)]
        calls=root_pair if t==root else [family.measured(m,obs[s],cfg) for s,m in enumerate(mods)]
        natural=[dict(a) for a,_ in calls];orders=copy.deepcopy(natural);clocks=[clock for _,clock in calls]
        forced=None
        if arm=='three_moves' and t-root<3 and before[0]:
            forced=PLAN['v4_action_override'][t-root]
            if forced=='NONE':orders[0].pop(IDS[0],None)
            else:orders[0][IDS[0]]=forced
            mods[0]._ST['tr'].set_actions(p,orders[0])
        bank=[b-max(0,k['wall']-1) for b,k in zip(bank,clocks)]
        advance(env,sim,orders)
        flow=family.econ.cash_flow(sim.cash.events)
        for s in (0,1):cash_totals[s].update(flow[s])
        following=sim.obs(0)['players']
        rows.append(dict(step=t,public_sha256=[family.digest(o) for o in obs],native_orders=natural,orders=orders,forced=forced,
                         clocks=clocks,cash=list(sim.cash),flow_by_seat=[dict(v) for v in flow],pre_workers=before,
                         post_workers=[following[s][2].get(uid) for s,uid in enumerate(IDS)],collisions=list(sim.last_collisions),
                         own_base_deposits=[e for e in sim.cash.events if e[0]=='own_delivery']))
    score=[float(row.reward) for row in env.states];require(all(s.status=='DONE' for s in env.states),'Missing official terminal')
    for s,x in enumerate(cash_totals): family.econ.check_equal(sum(x.values()),sim.cash[s]-history['cash'][s],'suffix cash origins')
    focuses=[]
    for seat,uid in enumerate(IDS):
        deliveries=[dict(step=r['step'],credited=e[2]) for r in rows for e in r['own_base_deposits'] if e[1]==seat and e[3]==uid and e[2]>0]
        losses=[dict(step=r['step'],collision=event) for r in rows for event in r['collisions'] if event[0]==uid]
        focuses.append(dict(worker=uid,positive_deposits=deliveries,cash_to_own_base=sum(d['credited'] for d in deliveries),
                            collisions=losses,terminal_worker=rows[-1]['post_workers'][seat]))
    clocks=[]
    for s in (0,1):
        cs=[r['clocks'][s] for r in rows]
        clocks.append(dict(calls=len(cs),cpu=sum(c['cpu'] for c in cs),wall=sum(c['wall'] for c in cs),max_wall=max(c['wall'] for c in cs),
                           slow=sum(c['wall']>1 for c in cs),K=sum(c['K'] for c in cs)/len(cs),H=sum(c['H'] for c in cs)/len(cs),
                           rolls=sum(c['rolls'] for c in cs),aborts=sum(c['aborted'] for c in cs)))
    result=dict(stream=ticket['stream'],arm=arm,score=score,margin=score[0]-score[1],cash=list(sim.cash),
                outcomes=[s.status for s in env.states],moves=len(rows),focus=focuses,clocks=clocks,cash_flows=[dict(x) for x in cash_totals],
                rival_first_three=[r['orders'][1].get(IDS[1],'NONE') if r['pre_workers'][1] else None for r in rows[:3]],
                actual_pivot_overrides=sum(r['orders'][0]!=r['native_orders'][0] for r in rows))
    name=f"stream{ticket['stream']}_{arm}.json.gz"
    digest=save(HERE/name,dict(protocol_sha256=LOCK,history=history,ticket=ticket,arm=arm,shared_root_orders=root_pair,result=result,rows=rows))
    return dict(**result,artifact=name,sha256=digest)


if __name__=='__main__':
    os.sched_setaffinity(0,{4})
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import sys
    sys.path.insert(0,str(ROOT/'code/env_src/carbon_challenge-master'))
    family=load('natural_response',FREIGHT)
    LOCK=sha((HERE/'protocol.json').read_bytes())
    PLAN=json.loads((HERE/'protocol.json').read_bytes());require(encode(PLAN)==encode(register()),'Protocol or bytes changed')
    game=json.loads(gzip.decompress(RAW.read_bytes()));results=[]
    for ticket in PLAN['streams']:
        state=prefix(family,game,ticket['seed_by_seat'],PLAN['step'])
        for side in ticket['order']:
            r=branch(family,state,ticket,side);results.append(r);print(ticket['stream'],side,r['score'],flush=True)
    save(HERE/'results.json',dict(protocol_sha256=LOCK,cpu_affinity=sorted(os.sched_getaffinity(0)),
         host_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),threads=torch.get_num_threads(),results=results))
