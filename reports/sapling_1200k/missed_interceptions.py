"""One-step hindsight upper bound on missed captures in the two actual interception games."""
from collections import Counter
import gzip
import json
from pathlib import Path

import tree_audited_game as arena
from interception_cash import collision_events
from tree_win_gate import validate_game, verify_raw

HERE = Path(__file__).resolve().parent
MOVES = ('NONE', 'UP', 'RIGHT', 'DOWN', 'LEFT')


def destination(sim, pos, action):
    return pos if action in (None, 'NONE') else sim._move_nbr[action][pos]


def received(events, seat):
    return sum(mass for owner, credit, mass, (kind, target) in events.values()
               if owner != seat and credit == seat and (kind == 'worker' or (kind == 'cash' and target == seat)))


def simulate(scratch, snap, joint):
    scratch.set_state(snap)
    scratch.step(*joint)
    return collision_events(snap, scratch, joint)


def census(run, row, parent_events, pack_hash):
    seat = row['candidate_seat']
    verify_raw(run, row, pack_hash)
    validate_game(row)
    raw = json.loads(gzip.decompress((run/row['raw_file']).read_bytes()))
    sim = arena.exam.FastCarbon(raw['game_configuration']).reset(seed=row['seed'])
    scratch = arena.exam.FastCarbon(raw['game_configuration'])
    if sim.config['moveCost'] != 0 or sim.config['smeltCost'] != 0:
        raise ValueError('cargo comparisons assume actual zero movement and smelting costs')
    roots, fixed_contacts, negative = [], [], None
    totals = Counter()
    for tick, joint in enumerate(raw['actions']):
        snap = sim.get_state()
        own = [w for w in sim.workers.values() if w.player == seat]
        foes = [w for w in sim.workers.values() if w.player != seat and w.is_collector]
        end = {w.id:destination(sim,w.pos,joint[w.player].get(w.id)) for w in own+foes}
        recruits = {pos for bid,pos in sim.bases[seat] if joint[seat].get(bid) in ('RECCOLLECTOR','RECPLANTER')}
        enemy_bases = {pos for _,pos in sim.bases[1-seat]}
        foe_reach = {w.id:{destination(sim,w.pos,m) for m in MOVES} for w in foes}
        old_events = simulate(scratch,snap,joint)
        actual = received(old_events,seat)
        parent = actual
        if tick in parent_events:
            event = parent_events[tick]
            if event['selected_actions'] != joint[seat]:
                raise ValueError('recorded policy command disagrees with archived replay')
            other = list(joint); other[seat] = event['base_actions']
            parent = received(simulate(scratch,snap,other),seat)
        totals['actual_received'] += actual
        totals['unmodified_v5_parent_received_on_these_roots'] += parent
        offers = []
        actual_contact = False
        geometric = False
        for w in own:
            if not w.is_collector:
                continue
            played = joint[seat].get(w.id) or 'NONE'
            for m in MOVES:
                if m == played:
                    continue
                totals['single_collector_alternatives'] += 1
                dest = destination(sim,w.pos,m)
                geometric |= any(f.carbon>w.carbon and dest in foe_reach[f.id] for f in foes)
                possible = [f for f in foes if f.carbon>w.carbon and end[f.id]==dest]
                if not possible:
                    if negative is None:
                        example = next((f for f in foes if f.carbon>=20 and f.carbon>w.carbon and dest in foe_reach[f.id]),None)
                        if example:
                            negative = dict(seat=seat,step=tick,collector=w.id,collector_origin=w.pos,
                                            alternative=m,destination=dest,enemy=example.id,enemy_pre_cargo=example.carbon,
                                            enemy_origin=example.pos,enemy_recorded_move=joint[1-seat].get(example.id,'NONE'),
                                            enemy_actual_destination=end[example.id],transferred=0.,reason='Enemy could reach proposed square but went elsewhere.')
                    continue
                actual_contact = True
                sub = list(joint); sub[seat] = dict(joint[seat]); sub[seat][w.id] = None if m=='NONE' else m
                events = simulate(scratch,snap,sub)
                captured = []
                for foe in possible:
                    event = events.get(foe.id)
                    if not event or event[1] != seat:
                        continue
                    _,_,amount,(kind, owner) = event
                    # Require this collector to win, not another friendly collector or a friendly tree.
                    winner = scratch.workers.get(w.id)
                    if winner is None or winner.pos != dest or not (kind=='worker' and owner==w.id or kind=='cash' and owner==seat):
                        continue
                    captured.append(dict(enemy=foe.id,origin=foe.pos,recorded_move=joint[1-seat].get(foe.id) or 'NONE',amount=amount,
                                         credited_to='own_cash' if kind=='cash' else 'own_worker'))
                fixed_contacts.append(dict(step=tick,collector=w.id,own_pre_cargo=w.carbon,alternative=m,destination=dest,
                                           enemy_base=dest in enemy_bases,targets=[dict(id=f.id,pre_cargo=f.carbon,
                                           event=events.get(f.id)) for f in possible],attacker_survived=w.id in scratch.workers,
                                           actual_enemy_cargo_retained=sum(v['amount'] for v in captured)))
                if not captured:
                    continue
                friendly = [u.id for u in own if u.id!=w.id and end[u.id]==dest]
                threats = [f.id for f in foes if f.carbon<=w.carbon and dest in foe_reach[f.id]]
                deaths = [wid for wid,owner,_,_,_ in scratch.last_collisions if owner==seat and end.get(wid)==dest]
                offers.append(dict(collector=w.id,origin=w.pos,own_pre_cargo=w.carbon,played=played,alternative=m,destination=dest,
                                   target_cargo=captured,amount=sum(x['amount'] for x in captured),after_all_received=received(events,seat),
                                   friendly_planned_destinations=friendly,friendly_recruit_order_here=dest in recruits,enemy_base=dest in enemy_bases,
                                   possible_lighter_or_tied_enemy=threats,actual_friendly_collision_deaths=deaths,
                                   safe_public=not (friendly or threats or dest in enemy_bases or dest in recruits)))
        totals['public_geometric_contact_roots'] += geometric
        totals['fixed_orders_lighter_heavier_overlap_roots'] += actual_contact
        if offers:
            roots.append(dict(step=tick,actual_received=actual,parent_received=parent,offers=offers))
        sim.step(*joint)
    if arena.prior.hashed(sim.get_state()) != row['final_state_sha256']:
        raise ValueError('terminal hash mismatch')
    return dict(seat=seat,seed=row['seed'],steps=len(raw['actions']),raw_sha256=row['raw_sha256'],counts=totals,roots=roots,fixed_contacts=fixed_contacts,negative=negative)


def aggregate(games, minimum, safe):
    picked=[]
    for game in games:
        for row in game['roots']:
            options=[]
            for op in row['offers']:
                qualified=[v for v in op['target_cargo'] if v['amount']>=minimum]
                if qualified and (not safe or op['safe_public']) and op['after_all_received']>row['actual_received']:
                    options.append((sum(v['amount'] for v in qualified),op,qualified))
            if options:
                mass,op,targets=max(options,key=lambda v:v[0]);picked.append((game['seat'],row,op,mass,targets))
    return dict(roots=len(picked),by_seat=[sum(s==n for s,_,_,_,_ in picked) for n in (0,1)],
                sum_one_best_per_root=sum(x[3] for x in picked),
                immediate_own_cash=sum(t['amount'] for *_,ts in picked for t in ts if t['credited_to']=='own_cash'),
                initially_on_collector=sum(t['amount'] for *_,ts in picked for t in ts if t['credited_to']=='own_worker'),
                played_received_on_those_roots=sum(r['actual_received'] for _,r,*_ in picked),
                parent_received_on_those_roots=sum(r['parent_received'] for _,r,*_ in picked),
                strongest_example=(dict(seat=(e:=max(picked,key=lambda v:v[3]))[0],step=e[1]['step'],actual_received=e[1]['actual_received'],parent_received=e[1]['parent_received'],**e[2]) if picked else None))


if __name__=='__main__':
    run=HERE/'tree_exam_interception'; report=json.loads((run/'results.json').read_bytes()); row_pack=report['header']['pack_sha256']
    live_path=HERE/'interception_live.json'; live=json.loads(live_path.read_bytes())
    if live['ledger_sha256'] != arena.prior.digest(run/'results.json') or live['candidate_sha256']!=row_pack or not live['actual_deployed']:
        raise ValueError('parent policy behavior must be reproduced on the recorded trajectory')
    lookup={(g['seed'],g['seat']):g for g in live['games']}; games=[]
    for row in report['games']:
        events=lookup[row['seed'],row['candidate_seat']]
        if events['raw_sha256']!=row['raw_sha256']:
            raise ValueError('parent replay identity mismatch')
        games.append(census(run,row,{e['step']:e for e in events['events']},row_pack))
    if sorted((g['seed'],g['seat']) for g in games)!=[(620106811,0),(620106811,1)]:
        raise ValueError('expected both prespecified seats')
    out=dict(scope='Hindsight on the actually visited roots. Only one own collector may change its move. All other commands stay fixed for one transition. Sum of per-root maxima is not cash, a feasible joint trajectory, or a prospective prediction.',
             safety='Screen rejects friendly next-square occupancy or issued recruit on destination, enemy base, and any tied/lighter enemy collector publicly able to reach destination. Other future risks remain.',
             eligible='Alternative must improve the actual team positive enemy-cargo receipts, the altered collector must win, and target pre-collision cargo must exceed its own. >=20 applies per victim. Tree absorption after the collision may reduce carried cargo.',
             matches_sha256=arena.prior.digest(run/'results.json'),live_sha256=arena.prior.digest(live_path),script_sha256=arena.prior.digest(__file__),
             all=aggregate(games,0,False),gte20=aggregate(games,20,False),safe_all=aggregate(games,0,True),safe_gte20=aggregate(games,20,True),games=games)
    arena.prior.shared.atomic_json(HERE/'missed_interceptions.json',out)
