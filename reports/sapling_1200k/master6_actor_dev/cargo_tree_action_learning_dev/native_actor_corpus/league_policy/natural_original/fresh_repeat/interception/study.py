"""Enumerate one-step simultaneous responses on the six observed cash-swing deliveries."""
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
HOSTED = HERE.parent / 'hosted'
sys.path.insert(0, str(HERE.parents[1]))
import economy as e
sys.path.insert(0, str(next(p for p in HERE.parents if (p/'ladder_live/live.py').is_file())/'ladder_live'))
import live
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.helpers import Board

MOVES = ('STAY', 'UP', 'RIGHT', 'DOWN', 'LEFT')
ARM = 'sampled_child'


def same_physics(a, b):
    for k in ('step', 'players', 'carbon', 'trees'):
        e.require(a[k] == b[k], f'different physical {k}')


def alter(cmd, who, move):
    result = dict(cmd)
    if move == 'STAY':
        result.pop(who, None)
    else:
        result[who] = move
    return result


def endpoint(sim, pos, move):
    return pos if move == 'STAY' else sim._move_nbr[move][pos]


def simulate(full, cfg, sim, commands):
    # The official interpreter allocates tree/worker identities through a
    # process-global dictionary. Each sibling sees exactly the real prefix.
    with idgen._rlock:
        outer = idgen._id_dict
        idgen._id_dict = outer.copy()
        try:
            truth = Board(full, cfg, commands).next().observation
        finally:
            idgen._id_dict = outer
    clone = deepcopy(sim)
    clone.cash = e.CashEvents(clone.cash)
    clone.step(*commands)
    same_physics(truth, clone.full_observation())
    for seat, flow in enumerate(e.cash_flow(clone.cash.events)):
        e.check_equal(sum(flow.values()), clone.cash[seat]-sim.cash[seat], 'branch cash decomposition')
    return truth, clone.cash.events


def differential(control, intervention, me):
    a, ae = control
    b, be = intervention
    da = [round(y[0]-x[0], 2) for x, y in zip(a['players'], b['players'])]
    net = []
    for x, y in zip(e.cash_flow(ae), e.cash_flow(be)):
        net.append({k:round(y.get(k, 0)-x.get(k, 0), 6) for k in e.KIND
                    if abs(y.get(k, 0)-x.get(k, 0))>1e-6})
    workers = []
    for x, y in zip(a['players'], b['players']):
        old, new = x[2], y[2]
        workers.append(dict(extra_dead={w:old[w] for w in sorted(old.keys()-new.keys())},
                            saved={w:new[w] for w in sorted(new.keys()-old.keys())},
                            cargo_delta=round(sum(z[1] for z in new.values())-sum(z[1] for z in old.values()),4)))
    return dict(cash_delta_by_seat=da, cash_margin_delta=round(da[me]-da[1-me], 2),
                cash_components_delta_by_seat=net, workers_by_seat=workers)


def study_root(full, cfg, sim, commands, target, me, source):
    opp = 1-me
    cargo = sim.workers.get(target)
    if cargo is None:
        return dict(**source, target_present=False, defenders=[], alternatives=[])
    actual = commands[opp].get(target, 'STAY')
    e.require(actual in MOVES, 'invalid observed worker action')
    target_cells = {endpoint(sim,cargo.pos,m) for m in MOVES}
    nearby = [w for w in sim.workers.values() if w.player == me and
              target_cells.intersection(endpoint(sim,w.pos,m) for m in MOVES)]
    root = dict(**source, target_present=True, target_start=[cargo.pos,cargo.carbon],
                actual_target_action=actual, actual_target_cell=endpoint(sim,cargo.pos,actual),
                defenders=[dict(id=w.id, pos=w.pos, cargo=w.carbon,
                                type='COLLECTOR' if w.is_collector else 'PLANTER',
                                actual=commands[me].get(w.id,'STAY')) for w in nearby],
                alternatives=[], enemy_controls={})
    if not nearby:
        return root
    for enemy in MOVES:
        pair = list(commands)
        pair[opp] = alter(commands[opp], target, enemy)
        ctrl = simulate(full,cfg,sim,pair)
        root['enemy_controls'][enemy]=dict(target_after=ctrl[0]['players'][opp][2].get(target),
            cash_after=[p[0] for p in ctrl[0]['players']],
            post_own={w.id:ctrl[0]['players'][me][2].get(w.id) for w in nearby})
        for w in nearby:
            current=commands[me].get(w.id,'STAY')
            e.require(current in MOVES,'invalid observed defending worker action')
            for defense in MOVES:
                if defense==current:
                    # The unmodified action is exactly the signed-control column.
                    continue
                pair[me]=alter(commands[me],w.id,defense)
                trial=simulate(full,cfg,sim,pair)
                child=trial[0]['players']
                target_after=child[opp][2].get(target)
                data=differential(ctrl,trial,me)
                root['alternatives'].append(dict(defender=w.id, defense=defense, enemy=enemy,
                    intersect=endpoint(sim,w.pos,defense)==endpoint(sim,cargo.pos,enemy),
                    target_after=target_after, defender_after=child[me][2].get(w.id),
                    additional_target_kill=target in data['workers_by_seat'][opp]['extra_dead'],
                    **data))
            pair[me]=commands[me]
    return root


def approach_bounds(frames, pivot, target, me, size):
    # One move of delay can be absorbed by legal STAY. Each possible first
    # move is held fixed, then later collisions, threats, expenses and taxes
    # are relaxed to give a geometric upper bound for the archived route.
    dist=e.distances(size)
    def dest(cell, move):
        x,y=cell%size,cell//size
        dx,dy={'STAY':(0,0),'UP':(0,-1),'RIGHT':(1,0),'DOWN':(0,1),'LEFT':(-1,0)}[move]
        return ((y+dy)%size)*size+((x+dx)%size)
    bounds=[]
    for lookback in (10,5,3):
        t=pivot-lookback
        start, cmds=frames[t]
        foe=start[1-me][2].get(target)
        row=dict(pivot=pivot, lookback=lookback, t=t, target=target, target_start=foe, starts=[], routes=[])
        if foe is None:
            bounds.append(row); continue
        bases=set(start[1-me][1].values())
        allies={w:a for w,a in start[me][2].items() if a[2]=='COLLECTOR' and a[1]<foe[1]}
        for w,a in sorted(allies.items()):
            row['starts'].append(dict(uid=w, cell=a[0], carried=a[1], actual_first=cmds[me].get(w,'STAY')))
        for h in range(2,lookback+1):
            state,_=frames[t+h]
            at=state[1-me][2].get(target)
            if at is None or at[0] in bases: continue
            cell=at[0]
            for who,a in allies.items():
                distance=dist[a[0]][cell]
                if distance>h: continue
                first=[m for m in MOVES if dist[dest(a[0],m)][cell]<=h-1]
                e.require(first,'empty legal first move')
                row['routes'].append(dict(uid=who, actual_first=cmds[me].get(who,'STAY'),
                    reachable_first_moves=first,
                    changed_first_moves=[m for m in first if m!=cmds[me].get(who,'STAY')],
                    target_state_turn=t+h, before_deposit_transitions=pivot+1-(t+h),
                    actual_target_cell=cell,start_distance=distance,elapsed_moves=h))
        bounds.append(row)
    return bounds


def run_game(game, ar, meta, cfg):
    seed, seat = game['seed'], game['seat']
    e.require(game['status']=='complete' and game['winner_seat']==1-seat,'expected natural loss')
    pivot = meta['pivotal_command']
    t0=pivot['step']
    target={uid:dict(official_net_delivery=amount,bank_cell=pos) for owner,uid,pos,amount in pivot['deposits']
            if owner==1-seat}
    receipt=ar.read(ARM+'/receipt.json')
    raw=ar.read(ARM+'/turns.jsonl')
    e.require(e.sha(receipt)==game['receipt_sha256'] and e.sha(raw)==json.loads(receipt)['turns_sha256'],
              'unbound receipt or action tape')
    e.np.random.seed(seed)
    env=e.make('carbon',configuration={**cfg,'randomSeed':seed})
    sim=e.FastCarbon(cfg).reset(seed=seed)
    e.require(e.sha(bytes(env.states[0].observation.full_carbon))==game['terrain'],'opening differs')
    records=[]; bank=[5.,5.]; frames={}
    for t, line in enumerate(raw.splitlines()):
        row=json.loads(line); ob=env.states[0].observation
        e.require(t==row['step']==ob.step==sim.step_count and not env.done,'wrong source turn')
        fingerprints=[e.sha(live.encoded(live.observe(env,k,bank[k]))) for k in (0,1)]
        e.require(fingerprints==row['public_observation_sha256'],'unsigned action inputs')
        e.require(all(c['status']=='ok' for c in row['calls']),'source call failed')
        bank=row['overage_remaining_seconds']
        commands=[c['response']['action'] for c in row['calls']]
        full=dict(ob,carbon=ob.full_carbon)
        same_physics(full,sim.full_observation())
        if t0-10<=t<=t0:
            frames[t]=(deepcopy(ob.players),commands)
            originals,events=simulate(full,env.configuration,sim,commands)
            if t==t0:
                measured={ev[3]:(ev[4],ev[2]) for ev in events if ev[:2]==['own_delivery',1-seat]}
                e.require(all(measured[k]==(v['bank_cell'],v['official_net_delivery']) for k,v in target.items())
                          and set(measured)==set(target),'delivery identities or amounts differ')
            for uid, amount in target.items():
                records.append(study_root(full,env.configuration,sim,commands,uid,seat,dict(seed=seed,seat=seat,
                    t=t,target=uid,pivot=t0,**amount,pre_public_sha256=fingerprints)))
        sim.step(*commands);env.step(commands)
        if t0-10<=t<=t0:
            nxt=env.states[0].observation
            same_physics(originals,dict(nxt,carbon=nxt.full_carbon))
    e.require(env.done and sim.step_count==game['official_transitions'] and
              [s.reward for s in env.states]==game['official_scores_by_seat'],'historical terminal differs')
    return records,[dict(seed=seed,seat=seat,**x) for uid in target for x in approach_bounds(frames,t0,uid,seat,cfg['size'])],dict(seed=seed,seat=seat,source_turns=sim.step_count,input_hashes_checked=2*sim.step_count,
                        exact_real_action_branch_controls=11,terrain=game['terrain'],original_scores=game['official_scores_by_seat'],
                        receipt_sha256=e.sha(receipt),tape_sha256=e.sha(raw),targeted_carriers=target)


def summarize(records):
    cases=[]
    for key in sorted({(q['seed'],q['seat'],q['target']) for q in records}):
        roots=sorted((q for q in records if (q['seed'],q['seat'],q['target'])==key),key=lambda r:r['t'])
        contacts=[]; kills=[]; branch_count=0
        for r in roots:
            branch_count+=len(r['alternatives'])
            if not r['target_present']:
                continue
            for x in r['alternatives']:
                if x['enemy']!=r['actual_target_action'] or not x['intersect']:
                    continue
                contacts.append(r['t'])
                if not x['additional_target_kill']:
                    continue
                alt=[v for v in r['alternatives'] if v['defender']==x['defender'] and v['defense']==x['defense']]
                escapes=[v['enemy'] for v in alt if v['target_after'] is not None]
                me=r['seat']
                kills.append(dict(t=r['t'],before_deposit=r['pivot']-r['t'],defender=x['defender'],
                    defense=x['defense'],actual_enemy=x['enemy'],target_cargo=r['target_start'][1],
                    own_after=x['defender_after'], cash_margin_delta=x['cash_margin_delta'],
                    cash_delta_by_seat=x['cash_delta_by_seat'], cash_components_delta_by_seat=x['cash_components_delta_by_seat'],
                    own_deaths=x['workers_by_seat'][me]['extra_dead'], own_carried_delta=x['workers_by_seat'][me]['cargo_delta'],
                    escapes=escapes))
        cases.append(dict(seed=key[0],seat=key[1],target=key[2],pivot=roots[0]['pivot'],
            delivery=roots[0]['official_net_delivery'],bank_cell=roots[0]['bank_cell'],
            root_slots=len(roots),present=sum(q['target_present'] for q in roots),
            any_reachable_roots=sum(bool(q['defenders']) for q in roots),actual_contact_roots=sorted(set(contacts)),
            matched_interventions=branch_count,kill_roots=sorted({x['t'] for x in kills}),kill_alternatives=kills))
    return cases


def main():
    settlement=HOSTED/'settlement.json'
    economy=HOSTED/'economy.json.gz'
    doc=json.loads(settlement.read_bytes()); ledger=json.loads(gzip.decompress(economy.read_bytes()))
    e.require(ledger['settlement_sha256']==e.sha(settlement.read_bytes()),'different ledger settlement')
    records=[]; checks=[]; geometry=[]
    for a in doc['archives']:
        key=(a['seed'],a['seat'])
        g=next(x for x in doc['games'] if (x['seed'],x['seat'],x['arm'])==(*key,ARM))
        meta=next(x for x in ledger['physical_games'] if (x['seed'],x['seat'],x['arm'])==(*key,ARM))
        path=HOSTED/f"{a['id']}.zip"
        e.require(e.sha(path.read_bytes())==a['sha256'],'incorrect source game archive')
        with zipfile.ZipFile(path) as ar:
            result,geo,check=run_game(g,ar,meta,e.config())
        records.extend(result); checks.append(check); geometry.extend(geo)
    cases=summarize(records)
    e.require(len(records)==66 and len(cases)==6,'wrong event sample')
    evidence=e.write(HERE/'branches.json.gz',dict(move_order=MOVES,records=records,geometry=geometry),True)
    e.write(HERE/'summary.json',dict(workflow_run=doc['workflow_run'],arm=ARM,
        input_settlement_sha256=e.sha(settlement.read_bytes()), input_ledger_sha256=e.sha(economy.read_bytes()),
        source_sha256=e.sha(Path(__file__).read_bytes()),branches_sha256=evidence,
        geometry=[dict(seed=g['seed'],seat=g['seat'],target=g['target'],start=g['t'],lookback=g['lookback'],
                       lighter_collectors=len(g['starts']), reachable_changed_before_entry=len({r['uid'] for r in g['routes'] if r['changed_first_moves']}),
                       reachable_changed_2plus_before_entry=len({r['uid'] for r in g['routes'] if r['changed_first_moves'] and r['before_deposit_transitions']>=2}),
                       first_example_2plus=next((r for r in g['routes'] if r['changed_first_moves'] and r['before_deposit_transitions']>=2),None))
                  for g in geometry],source_checks=checks,cases=cases))
    print('cases',[(c['seed'],c['seat'],c['target'],c['kill_roots'],c['matched_interventions']) for c in cases],flush=True)

if __name__=='__main__':
    main()
