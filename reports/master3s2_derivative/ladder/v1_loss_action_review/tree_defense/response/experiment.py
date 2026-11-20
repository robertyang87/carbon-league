"""Prospective root selection and a single collector order intervention against responsive native policies."""
import argparse
import copy
from collections import Counter
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

HERE=Path(__file__).resolve().parent
D=HERE.parents[3]
ROOT=D.parents[1]
REG=json.loads((HERE/'protocol.json').read_bytes())
LOCK='612331c1a02b689b021b36ee0c2e939cb3143828b3d3907eaacd017ac6fd4309'
def sha(data):
    return hashlib.sha256(data).hexdigest()
assert sha((HERE/'protocol.json').read_bytes())==LOCK
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def put(name,value):
    path=HERE/name
    b=(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':b=gzip.compress(b,mtime=0)
    if path.exists():
        if path.read_bytes()!=b:raise ValueError('Different previously archived run '+name)
        return sha(b)
    tmp=path.with_suffix('.tmp');tmp.write_bytes(b);tmp.replace(path)
    return sha(b)
def tape(ticket):
    directory=D/'ladder'/ticket['dataset']/'development'
    raw=(directory/'hosted'/ticket['archive']).read_bytes()
    assert sha(raw)==ticket['archive_sha256'] and sha((directory/'protocol.json').read_bytes())==ticket['protocol_sha256']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        reg=z.read('parent/registration.json');rec=z.read('parent/receipt.json')
        bound=next(a for a in map(json.loads,z.read('execution.jsonl').splitlines()) if a['arm']=='parent')
        turns=z.read('parent/turns.jsonl');obs=z.read('parent/observations.jsonl.gz')
    meta=json.loads(reg);receipt=json.loads(rec)
    assert bound['receipt_sha256']==sha(rec) and receipt['protocol_sha256']==sha(reg) and receipt['turns_sha256']==sha(turns)
    assert [meta['files'][k]['sha256'] for k in ('candidate','opponent')]==[REG['candidate'],REG['original']]
    assert (meta['seed'],meta['candidate_seat'],meta['matrix_sha256'])==(ticket['seed'],ticket['seat'],ticket['protocol_sha256'])
    return reg,turns.splitlines(),gzip.decompress(obs).splitlines()
def moves(p):
    y,x=divmod(p,15)
    return {'NONE':p,'UP':((y-1)%15)*15+x,'DOWN':((y+1)%15)*15+x,'LEFT':y*15+(x-1)%15,'RIGHT':y*15+(x+1)%15}
def select(every=False):
    import numpy as np
    from scipy.special import expit
    hazard=load('defense_frozen_hazard',HERE.parent/'fit_hazard.py')
    wfile=HERE.parent/'weights.json'
    assert sha(wfile.read_bytes())==REG['weights'] and sha((HERE.parent/'fit_hazard.py').read_bytes())==REG['feature_extractor']
    w=json.loads(wfile.read_bytes())
    dist=np.array([[hazard.distance(a,b,15) for b in range(225)] for a in range(225)])
    cfg_path=ROOT/'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/direct_original/games/candidate/raw_621758346_1.json.gz'
    cfg=json.loads(gzip.decompress(cfg_path.read_bytes()))['game_configuration']
    counts=Counter();roots=[];tree_cases=[]
    if every:
        assert sha((HERE/"every_step_protocol.json").read_bytes())=="65fff6e22d8d592a88c9da31a13515403f1f9bb83f7801963b27990b3f7dc692"
    for src in REG['inputs']:
        seat=src['seat'];_,turns,frames=tape(src)
        for t in (range(270,295) if every else REG['times']):
            if t>=len(turns) or t>=len(frames):
                counts['root_unavailable']+=1;continue
            counts['source_roots']+=1
            frame=json.loads(frames[t]);obs=frame['observations'][seat]
            history={k:json.loads(frames[t-k])['observations'][seat] for k in (1,3,5)}
            assert obs['step']==t and obs['player']==seat
            players=obs['players'];us,them=players[seat],players[1-seat]
            # In selection consult only parent orders, never the other seat's root order.
            own=json.loads(turns[t])['calls'][seat]['response']['action']
            occupied_bases=set(them[1].values())|{cell for uid,cell in us[1].items() if own.get(uid) or any(v[0]==cell for v in us[2].values())}
            for tree,(spot,age) in us[3].items():
                counts['observed_owned_trees']+=1
                x=hazard.features(obs,history,cfg,seat,tree,dist)
                if x is None:continue
                counts['reachable_by_planter']+=1
                z=(np.array(x[2])-w['mean'])/w['scale']
                risk=float(expit(w['coef'][0]+z@w['coef'][1:]))
                if risk<.2 and not every:continue
                if risk>=.2:counts['hazard_at_least_point20']+=1
                _,eid=min((dist[spot,v[0]],uid) for uid,v in them[2].items() if v[2]=='PLANTER')
                ep=them[2][eid][0];prior=history[1]['players'][1-seat][2].get(eid)
                prev=next((a for a,v in moves(prior[0]).items() if v==ep),None) if prior else None
                near=moves(ep)
                proposal=min(near,key=lambda a:(dist[near[a],spot],a!=prev,near[a]))
                target=near[proposal]
                possible=[(uid,loc,cargo) for uid,(loc,cargo,role) in us[2].items() if role=='COLLECTOR' and cargo<=20 and target in moves(loc).values()]
                case=dict(seed=src['seed'],seat=seat,t=t,tree=tree,risk=risk,target=target,planter=eid,defenders=len(possible))
                if every:tree_cases.append(case)
                if target in occupied_bases:
                    case['reason']='base';continue
                if not possible:
                    case['reason']='no_light_collector_in_range';continue
                case['reason']='already_covered'
                
                for uid,(loc,cargo,role) in us[2].items():
                    if role!='COLLECTOR' or cargo>20 or target not in moves(loc).values():continue
                    action=next(a for a,v in moves(loc).items() if v==target)
                    if action==own.get(uid,'NONE'):continue
                    if any(moves(v[0])[own.get(j,'NONE')]==target for j,v in us[2].items() if j!=uid):
                        if case.get('reason')!='accepted':case['reason']='teammate_conflict'
                        continue
                    case['reason']='accepted'
                    roots.append(dict(seed=src['seed'],seat=seat,step=t,tree=tree,tree_cell=spot,age=age,posted_rent=obs['trees'][tree][1],risk=risk,
                        actor=uid,actor_cargo=cargo,actor_cell=loc,prior_order=own.get(uid,'NONE'),new_order=action,predicted_cell=target,
                        nearest_planter=eid,enemy_cell=ep,predicted_planter_order=proposal,public_sha=frame['public_observation_sha256'],
                        full_parent_order=own))
    roots.sort(key=lambda r:(-r['risk'],r['actor_cargo'],r['seed'],r['seat'],r['step'],r['tree'],r['actor']))
    selected=[];seeds=set()
    for r in roots:
        if len(selected)==2:break
        if r['risk']>=.2 and r['seed'] not in seeds:
            selected.append(r);seeds.add(r['seed'])
    payload=dict(protocol=LOCK,counts=dict(counts),all_admissible=roots,selected=selected)
    if every:
        payload.update(secondary_protocol=sha((HERE/'every_step_protocol.json').read_bytes()),tree_cases=tree_cases)
    print('selection',put('every_step_selected.json' if every else 'selected.json',payload),[(r['seed'],r['seat'],r['step'],r['risk']) for r in selected])

def continue_one(ordinal):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert len(os.sched_getaffinity(0))==1
    os.environ['CARBON_DEBUG']='1'
    frozen=(HERE/'relative_risk_tickets.json').read_bytes()
    assert sha(frozen)=='b6629c874d4e069003f6cccf42e6c612227835192d983402ea733920eb78e5a4'
    registry=json.loads(frozen);r=registry['inputs'][ordinal]
    seat,seed,t=r['seat'],r['seed'],r['step']
    ticket=next(s for s in REG['inputs'] if (s['seed'],s['seat'])==(seed,seat))
    prefix_tools=load('defender_restore',D/'endgame_cash_projection/response_check/recovery.py')
    run=load('defender_views',D/'endgame_cash_projection/response_check/run.py')
    F=prefix_tools.freight
    reg_bytes,turns,observations=tape(ticket)
    meta=json.loads(reg_bytes)
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);reg=tmp/'contract.json';reg.write_bytes(reg_bytes)
        prefix_tools.live.configure(reg,tmp/'unused',tmp/'lock')
        env,cfg=prefix_tools.live.official()
    sim=F.econ.FastCarbon(cfg).reset(seed=seed)
    assert sha(bytes(map(int,sim.carbon)))==meta['physical_map_sha256']
    sim.cash=F.econ.CashEvents(sim.cash)
    bank=[float(meta['platform']['overage_seconds_per_player'])]*2
    prefix=[]
    for i in range(t+1):
        frame=json.loads(observations[i]);line=json.loads(turns[i])
        public=[F.public(env,s,bank) for s in (0,1)]
        assert [F.digest(v) for v in public]==frame['public_observation_sha256']==line['public_observation_sha256']
        assert frame['observations']==public and list(sim.cash)==[p[0] for p in public[0]['players']]
        if i==t:break
        assert all(c['status']=='ok' for c in line['calls'])
        orders=[c['response']['action'] for c in line['calls']]
        prefix.append((frame['observations'],orders));bank=line['overage_remaining_seconds']
        sim.cash.events.clear();sim.step(*orders);env.step(orders,[])
    assert r['public_sha']==frame['public_observation_sha256']
    own=line['calls'][seat]['response']['action']
    assert own==r['full_parent_order']
    changes=dict(own)
    if r['new_order']=='NONE':changes.pop(r['actor'],None)
    else:changes[r['actor']]=r['new_order']
    assert set(k for k in (set(own)|set(changes)) if own.get(k)!=changes.get(k))=={r['actor']}
    assert own.get(r['actor'],'NONE')==r['prior_order']
    expected=dict(archive_sha256=ticket['archive_sha256'],protocol_sha256=ticket['protocol_sha256'],
                  physical_map=meta['physical_map_sha256'],historical_orders_prefix=F.digest([pair for _,pair in prefix]),
                  root_public_sha=r['public_sha'],common_clock_banks=bank)
    history=dict(prefix=prefix,public=public,cfg=cfg)
    parent_path=D/'cold_start/candidate.py'
    native_path=ROOT/'data/master3s2_source.py'
    assert sha(parent_path.read_bytes())==REG['candidate'] and sha(native_path.read_bytes())==REG['original']
    mods=[None,None]
    mods[seat]=F.load('defend_native_subject',parent_path)
    mods[1-seat]=F.load('defend_native_original',native_path)
    run.fresh_stats=copy.deepcopy(mods[seat]._S_STATS)
    rng=int.from_bytes(hashlib.sha256(f'{seed}/{seat}/{t}/tree-defense/v1'.encode()).digest()[:4],'big')
    for i,m in enumerate(mods):
        assert m._SEED is None and m._TEMP==1. and m._S_ABORT==.58 and m._SEARCH_OK
        run.history_state(m,history,i,rng if i==seat else rng^0x7d4a1f6b)
    state_hash=F.original_state.h([F.original_state.dynamic(m) for m in mods])
    rng_hash=F.original_state.h(mods[seat]._ST['rng'])
    self_reply=F.measured(mods[seat],public[seat],cfg)
    snapshot=F.official_state.take(env,bank)
    arm_order=registry['arm_orders'][ordinal]
    ticket_min=dict(seed=seed,seat=seat,step=t,policy_rng=rng,ordinal=ordinal)
    output=run.continue_pair(ticket_min,dict(parent=own,defend=changes,order_source='historical_parent_vs_relative_hazard_intervention'),
                  arm_order,(mods,cfg,env,sim,public,expected),self_reply,state_hash,rng_hash,put)
    for arm in arm_order:
        bank=F.official_state.restore(env,snapshot)
        branch=json.loads(gzip.decompress((HERE/f'{seed}_{seat}_{t}_{arm}.json.gz').read_bytes()))
        trace=[]
        for row in branch['rows']:
            curr=env.states[0].observation.players
            assert [F.digest(F.public(env,i,bank)) for i in (0,1)]==row['input_sha256']
            env.step(row['orders'],[])
            ps=env.states[0].observation.players
            focus=r['nearest_planter']
            tree_owner=next((s for s in (0,1) if r['tree'] in ps[s][3]),None)
            trace.append(dict(after=row['t']+1,tree_owner=tree_owner,enemy_planter=ps[1-seat][2].get(focus),our_collector=ps[seat][2].get(r['actor']),
                      enemy_order=row['orders'][1-seat].get(focus,'NONE') if focus in curr[1-seat][2] else None,
                      our_order=row['orders'][seat].get(r['actor'],'NONE') if r['actor'] in curr[seat][2] else None,
                      cash=row['end_cash'],flow=row['flow']))
            bank=[v-max(0.,clock['wall']-1.) for v,clock in zip(bank,row['clocks'])]
        assert [float(st.reward) for st in env.states]==output['arms'][arm]['scores']
        arms=output['arms'][arm]
        arms.update(target_history=trace,enemy_owned_after_five=any(q['tree_owner']==1-seat for q in trace[:5]),
            root_planter_landed_prediction=bool(trace[0]['enemy_planter'] and trace[0]['enemy_planter'][0]==r['predicted_cell']),
            clock_by_seat=[dict(max_wall=max(c['clocks'][s]['wall'] for c in branch['rows']),
                 cpu=sum(c['clocks'][s]['cpu'] for c in branch['rows']),depth=dict(Counter(c['clocks'][s]['H'] for c in branch['rows'])),
                 slow=sum(c['clocks'][s]['wall']>1 for c in branch['rows'])) for s in (0,1)])
    output.update(protocol=LOCK,tickets_sha256=sha(frozen),focus=r,oracle_free_selection=True,
        margin_difference=output['arms']['defend']['margin']-output['arms']['parent']['margin'])
    print('settled',seed,put(f'{seed}_{seat}_{t}_pair.json',output),output['margin_difference'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('grid','every','continue'));parser.add_argument('ordinal',type=int,nargs='?')
    opt=parser.parse_args()
    if opt.command=='continue':continue_one(opt.ordinal)
    else:select(opt.command=='every')
