"""Re-evaluate all frozen public roots on native clocks; continue qualified changed roots to real terminal."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parents[1]
REPO = FAMILY.parents[1]
PROTOCOL = '3b5bba544d13cb26ae9692e2414a1d14a49277b0232864a31230f8ea3a040f4c'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load('actual_original_response', FAMILY/'freight_response/run.py')
np = base.np
sha = base.sha


def save(path, data):
    raw=(json.dumps(data, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n').encode()
    temp=path.with_suffix('.partial')
    temp.write_bytes(gzip.compress(raw, mtime=0) if path.suffix=='.gz' else raw)
    temp.replace(path)


def tape(ticket):
    zpath=FAMILY/'ladder'/ticket['lane']/'development/hosted'/(str(ticket['archive_id'])+'.zip')
    assert sha(zpath.read_bytes()) == ticket['archive_sha256']
    with zipfile.ZipFile(zpath) as z:
        raw=z.read('parent/observations.jsonl.gz')
        commands=z.read('parent/turns.jsonl')
        assert sha(raw)==ticket['public_stream_sha256']
        assert sha(commands)==json.loads(z.read('parent/receipt.json'))['turns_sha256']
    return gzip.decompress(raw).splitlines(),commands.splitlines()


def historical_root(ticket, side, module, cfg, frames, commands):
    tracker=module.BeliefTracker(module._Bel,side)
    previous=None
    for t in range(ticket['step']):
        obs=json.loads(frames[t])['observations'][side]
        row=json.loads(commands[t])
        assert base.digest(obs)==row['public_observation_sha256'][side]
        tracker.step(obs['players'], t, obs['carbon'])
        tracker.set_actions(obs['players'],row['calls'][side]['response']['action'])
        previous=obs['players']
    return tracker, previous


def call(m, history, obs, cfg, seed, costs, mode):
    tracker,prev=history
    m._ST=dict(tr=copy.deepcopy(tracker), prev=copy.deepcopy(prev), rng=np.random.RandomState(seed))
    m._S_COST[:]=costs
    m._S_STATS.pop('last', None)
    m._S_REC=[]
    if mode!='P':
        m._RETURN_ENABLED=(mode=='E')
        m._RETURN_TRACE={}
    action, clocks=base.measured(m,obs,cfg)
    records=list(m._S_REC);assert len(records)<=1
    m._S_REC=None
    trace={} if mode=='P' else m._RETURN_TRACE
    if mode!='P':m._RETURN_TRACE=None
    menu=depths=values=chosen=None
    if records:
        _,_,menu,depths,values,chosen=records[0]
        assert menu[chosen]==action
    return dict(mode=mode, action=action, clocks=clocks, trace=trace, menu=menu, depths=depths,
                values=values, selected=chosen)


def snapshot(module):
    return copy.deepcopy(base.original_state.dynamic(module)),(module._RX_CFG[0],copy.deepcopy(module._RX_CFG[1]))


def restore(module, snap):
    state,cfg=snap
    for k,v in copy.deepcopy(state).items():setattr(module,k,v)
    module._RX_CFG[:]=[cfg[0],copy.deepcopy(cfg[1])]
    assert base.original_state.h(base.original_state.dynamic(module))==base.original_state.h(state)


def continue_pair(ticket, probes, root_parent, parent, examiner, cfg, frames, commands, child_identity, ordinal):
    seed,seat,t=ticket['seed'],ticket['seat'],ticket['step']
    np.random.seed(seed)
    env=base.make('carbon',configuration={**cfg,'randomSeed':seed})
    sim=base.econ.FastCarbon(cfg).reset(seed=seed)
    assert sha(bytes(env.states[0].observation.full_carbon))==ticket['terrain']==sha(bytes(map(int,sim.carbon)))
    bank=[5.,5.]
    for i in range(t):
        hist=json.loads(commands[i]);public=json.loads(frames[i]);views=[base.public(env,s,bank) for s in (0,1)]
        assert [base.digest(v) for v in views]==hist['public_observation_sha256']==public['public_observation_sha256']
        joint=[v['response']['action'] for v in hist['calls']]
        env.step(joint,[]);sim.step(*joint);bank=hist['overage_remaining_seconds']
    friend, foe=seat,1-seat
    views=[base.public(env,s,bank) for s in (0,1)]
    assert base.digest(views[friend])==ticket['public_observation_sha256']
    # The untouched original sees only its own root, independent of both focal commands.
    opp_history=historical_root(ticket,foe,examiner,cfg,frames,commands)
    opp=call(examiner,opp_history,views[foe],cfg,(19073+7*t+seed+foe)%2**32,[.001,.002,.001,.002],'P')
    post_opponent=snapshot(examiner)
    post_ref=base.official_state.take(env,bank);shadow=copy.deepcopy(sim)
    seed_root=probes['P']['action'];new_root=probes['E']['action']
    focal=probes['E']['trace']['pivot'];initial=views[friend]['players'][friend][2][focal][1]
    order=('parent','new') if ordinal==0 else ('new','parent')
    summary={}
    for arm in order:
        bank=base.official_state.restore(env,post_ref);sim=copy.deepcopy(shadow)
        sim.cash=base.econ.CashEvents(sim.cash)
        restore(parent,root_parent);restore(examiner,post_opponent)
        rootcmd=new_root if arm=='new' else seed_root
        parent._ST['tr'].set_actions(views[friend]['players'],rootcmd)
        rows=[];first_bank=None
        while not env.done:
            public=[base.public(env,s,bank) for s in (0,1)];turn=public[0]['step']
            player=(rootcmd,probes['P' if arm=='parent' else 'E']['clocks']) if turn==t else base.measured(parent,public[friend],cfg)
            other=(opp['action'],opp['clocks']) if turn==t else base.measured(examiner,public[foe],cfg)
            commands_pair=[None,None]; commands_pair[friend]=player;commands_pair[foe]=other
            actions=[x[0] for x in commands_pair]; clocks=[x[1] for x in commands_pair]
            bank=[b-max(0,c['wall']-1.) for b,c in zip(bank,clocks)]
            sim.cash.events.clear();sim.step(*actions);env.step(actions,[])
            assert sim.cash==[p[0] for p in env.states[0].observation.players]
            deposits=[(e[1],e[3],e[2]) for e in sim.cash.events if e[0]=='own_delivery' and e[2]>0]
            if first_bank is None:
                credit=next((v for s,u,v in deposits if s==seat and u==focal),None)
                if credit is not None:first_bank=dict(step=turn,actual_amount=credit,initial_cargo_upper_bound=min(initial,credit))
            rows.append(dict(step=turn, orders=actions, clocks=clocks, public_sha256=[base.digest(v) for v in public],
                             score_wallet=list(sim.cash), own_deposits=deposits,
                             focal_state=env.states[0].observation.players[seat][2].get(focal)))
        scores=[s.reward for s in env.states]
        name=f"root_{ticket['ordinal']}_{arm}.json.gz"
        target=HERE/name
        result=dict(protocol_sha256=PROTOCOL,candidate_sha256=child_identity,arm=arm,seed=seed,seat=seat,starting_step=t,
                    starting_focal=focal,initial_cargo=initial,first_actual_own_bank=first_bank,official_scores=scores,
                    official_margin=scores[seat]-scores[1-seat],winner_seat=None if scores[0]==scores[1] else int(scores[1]>scores[0]),
                    root_foe_order=opp['action'],bank_at_end=bank,rows=rows,
                    timing='Local in-process elapsed and CPU. No platform cgroup adjudication. Any soft or hard timing breach is reported.')
        save(target,result)
        summary[arm]={k:v for k,v in result.items() if k not in ('rows','timing','root_foe_order')}
        summary[arm].update(tape=name,sha256=sha(target.read_bytes()),hard_breaches=sum(x['wall']>3 for row in rows for x in row['clocks']))
    return summary


def run():
    os.environ['CARBON_DEBUG']='1'
    cfg=base.econ.config();cfg.pop('randomSeed',None)
    pbytes=(HERE/'root_protocol.json').read_bytes();assert sha(pbytes)==PROTOCOL
    plan=json.loads(pbytes);release=json.loads((HERE.parent/'package/manifest.json').read_bytes())
    paths=(FAMILY/'cold_start/candidate.py',HERE.parent/'package/candidate.py',REPO/'sources/master3s2/master3s2.py')
    assert [sha(p.read_bytes()) for p in paths]==[plan['parent_sha256'],release['candidate_sha256'],plan['original_sha256']]
    result_path=HERE/'root_results.json.gz'
    if result_path.exists():raise ValueError('Completed frozen study already exists')
    if release['candidate_sha256'] == '6df55e93a5a51f6f740948f6e03494aac695c36937e67b1235a7e9503f83a9d3':
        raise ValueError('Obsolete cargo-tax labels. Wait for the repaired and re-frozen release.')
    if os.sched_getaffinity(0) != {2}:
        raise ValueError('Run local native clocks alone on registered CPU 2')
    parent,child,original=[base.load(f'real_root_s{i}',p) for i,p in enumerate(paths)]
    import torch
    torch.set_num_threads(1)
    for model in (parent,child,original):assert model._TEMP==1 and model._SEED is None and model._S_ABORT==.58 and model._RX and model._SEARCH_OK
    all_rows=[]; used_maps=[]; evaluated=[]
    for ticket in plan['roots']:
        n=ticket['ordinal'];row=dict(ticket=ticket,status='no_eligible_preaction_root')
        if 'step' in ticket:
            seed,seat,t=ticket['seed'],ticket['seat'],ticket['step'];frames,cmds=tape(ticket)
            obs=json.loads(frames[t])['observations'][seat];assert base.digest(obs)==ticket['public_observation_sha256']
            history=historical_root(ticket,seat,parent,cfg,frames,cmds)
            order=('P','E','E','D') if n%2==0 else ('D','E','E','P'); calls=[];pstate=None
            key=(seed+19073+7*t+seat)%2**32
            for tag in order:
                m=parent if tag=='P' else child
                sample=call(m,history,obs,cfg,key,[.001,.002,.001,.002],tag);calls.append(sample)
                if tag=='P':pstate=snapshot(m)
            p,ds=next(x for x in calls if x['mode']=='P'),next(x for x in calls if x['mode']=='D')
            a,b=[x for x in calls if x['mode']=='E']; work=lambda q:(q['clocks']['K'],q['clocks']['H'],q['clocks']['rolls'])
            choice=lambda q:(q['action'],q['menu'],q['depths'],q['selected'])
            changed=a['action']!=p['action']; stable=(choice(p)==choice(ds) and choice(a)==choice(b))
            matched=len({work(x) for x in calls})==1
            learned=all(x['trace'].get('existing') is False and x['trace'].get('offer_index') is not None and x['trace']['offer_index']==x['selected'] for x in (a,b))
            valid=changed and stable and matched and learned
            row.update(status='verified',calls=calls,physical_change=changed,stable=stable,work_matched=matched,new_offer_won=learned,qualified=valid)
            if valid and seed not in used_maps and len(used_maps)<2:
                result=continue_pair(ticket,{'P':p,'E':a},pstate,parent,original,cfg,frames,cmds,release['candidate_sha256'],len(used_maps))
                row['responses']=result;used_maps.append(seed);evaluated.append(n)
            print('root',n,'valid',valid,'physical',changed,'added',a['trace'].get('existing') is False,'won',learned,flush=True)
        all_rows.append(row)
        save(HERE/'root_progress.json.gz',dict(protocol_sha256=PROTOCOL,candidate_sha256=release['candidate_sha256'],rows=all_rows,complete=False))
    assert len(all_rows)==24
    counts=dict(registered=len(all_rows),screened=sum(r['status']=='verified' for r in all_rows),
                added=sum(any(x['trace'].get('existing') is False for x in r.get('calls',[])) for r in all_rows),
                physical=sum(bool(r.get('physical_change')) for r in all_rows),qualified=sum(bool(r.get('qualified')) for r in all_rows),
                continued=len(evaluated),map_clusters_continued=used_maps)
    save(result_path,dict(protocol_sha256=PROTOCOL,candidate_sha256=release['candidate_sha256'],counts=counts,
                          continuation_ordinals=evaluated,rows=all_rows,cpu_affinity=sorted(os.sched_getaffinity(0)),
                          cgroup={k:Path('/sys/fs/cgroup',k).read_text().strip() for k in ['cpu.max','memory.max']},
                          complete=True))
    (HERE/'root_progress.json.gz').unlink();print(counts)


if __name__=='__main__':
    run()
