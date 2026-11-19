"""Attribute one recruit's actual life and first tree tenures on a sealed pair of action tapes."""
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STUDY = HERE.parent / 'endgame_queue_dev/study.py'
spec = importlib.util.spec_from_file_location('late_hire_official_transitions', STUDY)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
FastCarbon, read, sha, require = engine.FastCarbon, engine.read, engine.sha, engine.require


def near(actual, expected, label):
    require(math.isclose(actual, expected, abs_tol=2.e-7, rel_tol=1.e-10), f'{label}: {actual} != {expected}')


def positions(sim, before, counters, joint, events):
    result = {}
    for p, pl in enumerate(before['players']):
        for uid, (cell, load, role) in pl[2].items():
            action = joint[p].get(uid)
            if action in sim._move_nbr:
                cell = sim._move_nbr[action][cell]
                load *= 1 - sim.config['moveCost']
            result[uid] = (p, cell, load, role)
    for kind, p, _, *args in events:
        if kind in ('hire_collector', 'hire_planter'):
            uid = f'player-{p}-worker-{counters[p]}'
            counters[p] += 1
            require(uid not in result, 'new worker identity collision')
            result[uid] = (p, args[1], 0, kind.removeprefix('hire_').upper())
    return result


def collisions(sim, units):
    groups = defaultdict(list)
    for u, (_, cell, _, _) in units.items():
        groups[cell].append(u)
    expected, outcome = set(), []
    winners = {}
    for cell, group in groups.items():
        if len(group) < 2:
            continue
        collectors = [u for u in group if units[u][3]=='COLLECTOR']
        lightest = [u for u in collectors if units[u][2]==min(units[v][2] for v in collectors)]
        winner = lightest[0] if len(lightest)==1 else None
        winners[cell] = winner
        expected.update(u for u in group if u!=winner)
    require(expected == {r[0] for r in sim.last_collisions}, 'collisions vs destinations')
    for dead, owner, load, credited, winning_seat in sim.last_collisions:
        p, cell, cargo, role = units[dead]
        near(cargo,load,'victim cargo')
        winner = winners[cell]
        require(owner==p and winning_seat==(units[winner][0] if winner else None), 'wrong collision winner')
        outcome.append({'victim':dead, 'victim_seat':owner, 'victim_role':role, 'cargo':load, 'cell':cell,
                        'winner':winner, 'recipient_seat':credited, 'occupants':groups[cell],
                        'transfer':'collector_inventory' if winner else 'treasury' if credited is not None else 'ground'})
    return outcome


def tree_records(segments, active, before, after, sim, events, uid, seat, post):
    old = {tid:(p,cell,age) for p,pl in enumerate(before['players']) for tid,(cell,age) in pl[3].items()}
    new = {tid:(p,cell,age) for p,pl in enumerate(after['players']) for tid,(cell,age) in pl[3].items()}
    def credit(tid, p, cash, kind, timing):
        if tid not in segments:
            return
        record = segments[tid]
        book = 'first_tenure' if tid in active else 'after_first_tenure'
        require(book!='first_tenure' or p==seat, 'wrong first-tenure beneficiary')
        record[book][p][kind] += cash
        record['receipts'].append({'post':post, 'phase':timing,'seat':p,'kind':kind,'cash':cash,'tenure':book})
    # No-winner collision at a tree posts before subsequent claim/expiry checks below.
    for kind,p,cash,*args in events:
        if kind=='tree_collision_cash':
            tid=args[2]
            require(old[tid][0]==p and old[tid][2]<sim.config['treeLifespan'], 'pre-claim tree owner')
            credit(tid,p,cash,kind,'collision')
    for tid in list(active):
        if tid not in new or new[tid][0]!=seat:
            segments[tid]['first_tenure_ended_at'] = post
            segments[tid]['first_tenure_end_reason'] = 'expired' if tid not in new else 'seized_by_opponent'
            active.remove(tid)
    for kind,p,cash,*args in events:
        if kind not in ('plant','seize') or args[0]!=uid:
            continue
        cell=args[1]
        tree_ids=[tid for tid,(owner,pos,_) in new.items() if (owner,pos)==(seat,cell)]
        require(p==seat and len(tree_ids)==1,'missing focal tree')
        tid=tree_ids[0]
        require((kind=='plant' and tid not in old) or (kind=='seize' and old[tid][0]!=seat), 'acquisition identity')
        if tid not in segments:
            segments[tid]={'tree':tid,'cell':cell,'first_acquisition':kind,'first_acquired_at':post,
                           'first_tenure_ended_at':None,'first_tenure_end_reason':None,'acquisitions':[],
                           'first_tenure':[Counter(),Counter()],'after_first_tenure':[Counter(),Counter()],'receipts':[]}
            active.add(tid)
        segments[tid]['acquisitions'].append({'post':post,'kind':kind,'cash':cash})
    for kind,p,cash,*args in events:
        if kind=='tree_absorption':
            tid=args[0]
            require(new[tid][0]==p, 'post-claim tree owner')
            near(after['trees'][tid][1],cash,'official tree receipt')
            credit(tid,p,cash,kind,'absorption')


def run_arm(base_state, cfg, tail, seat, base, hire, score, expected_terminal):
    sim=FastCarbon(cfg).set_state(base_state)
    uid=hire['worker']; role=hire['role']
    start=sim.step_count
    require(uid not in sim.workers and sim.base_at[hire['base_cell']]==(seat,base),'focal identity at root')
    require(tail and tail[0]==hire['joint_action'],'incorrect first commands')
    money=Counter(); cargo=Counter(); losses=[]; kills=[]; ties=[]; life=[]; trees={}; first=set()
    events_by_seat=[Counter(),Counter()]; died_at=None; enemy_deaths=0; all_enemy_deaths=0
    for i,joint in enumerate(tail):
        before=sim.full_observation(); counters=sim.worker_counters
        opened=before['players'][seat][2].get(uid)
        result=engine.twin_step(sim,joint)
        after=sim.full_observation(); closed=after['players'][seat][2].get(uid)
        t=sim.step_count; events=result['events']; terminal=i==len(tail)-1
        require(result['status']==(['DONE','DONE'] if terminal else ['ACTIVE','ACTIVE']), 'unsealed stopping time')
        for p, flow in enumerate(result['flow']):
            events_by_seat[p].update(flow)
        units=positions(sim,before,counters,joint,events); crashed=collisions(sim,units)
        tree_records(trees,first,before,after,sim,events,uid,seat,t)
        enemy_deaths += len(result['deaths'][1-seat])
        all_enemy_deaths += sum(unit[0]==1-seat and wid not in sim.workers for wid,unit in units.items())
        if i==0:
            wage=[ev for ev in events if ev[:2]==['hire_'+role.lower(),seat] and ev[3]==base]
            require(len(wage)==1 and closed and (closed[0],closed[2])==(hire['base_cell'],role), 'selected hire did not survive its root')
            near(-wage[0][2],hire['hire_cost'],'wage'); money['hire_cost']=-wage[0][2]
        if uid not in units:
            require(closed is None and died_at is not None, 'focal identity reappeared')
            continue
        p,cell,moved_load,unit_role=units[uid]
        require(p==seat and role==unit_role,'focal identity changed')
        action=joint[seat].get(uid) or 'NONE'
        opening=float(opened[1]) if opened else 0.
        closing=float(closed[1]) if closed else 0.
        step_cargo=Counter(move_fee=opening-moved_load)
        won=[e for e in crashed if e['winner']==uid]
        for e in won:
            if e['victim_seat']!=seat:
                kills.append(dict(post=t,**e)); step_cargo['enemy_cargo_captured']+=e['cargo']
            else:
                step_cargo['friendly_cargo_received']+=e['cargo']
        dead=next((e for e in crashed if e['victim']==uid),None)
        if dead:
            step_cargo['collision_cargo_lost']=dead['cargo']
            losses.append(dict(post=t, **dead))
            if not dead['winner']:
                ties.extend(dict(post=t,**e) for e in crashed if e['cell']==cell and e['victim_seat']!=seat)
        deposits=[]; purchases=[]
        for kind,receiver,amount,*args in events:
            if not args or args[0]!=uid:
                continue
            if kind in ('own_delivery','opponent_worker_delivery_to_this_base'):
                money['own_base_deposits' if receiver==seat else 'enemy_base_deposits']+=amount
                step_cargo['deposited_to_either_base']+=amount
                deposits.append({'seat':receiver,'kind':kind,'cash':amount})
            elif kind in ('plant','seize'):
                money[kind+'_cost']-=amount; purchases.append({'action':kind,'cash':amount,'cell':cell})
            elif kind in ('tree_collision_cash','base_collision_cash'):
                money['focal_loss_collision_credit_'+('own' if receiver==seat else 'enemy')]+=amount
        if not dead and role=='COLLECTOR':
            recovered=moved_load+sum(e['cargo'] for e in won)
            if cell in sim.base_at:
                bank_seat=sim.base_at[cell][0]
                if bank_seat!=seat and recovered<=cfg['smeltCost']:
                    step_cargo['hostile_base_confiscation']=recovered
                elif recovered>=cfg['smeltCost']:
                    step_cargo['smelt_fee']=cfg['smeltCost']
            elif action=='NONE':
                prior_tree=[a for pl in before['players'] for pos,a in pl[3].values() if pos==cell]
                if not prior_tree or prior_tree[0]>=cfg['treeLifespan']:
                    ground=cfg['co2FrmWithered'] if prior_tree else before['carbon'][cell]
                    collectors=sum(w[2]=='COLLECTOR' for w in after['players'][seat][2].values())
                    step_cargo['harvested']=ground*max(0,cfg['initialCollectRate']-collectors*cfg['collectDecreaseRate'])
            pre_tax=(recovered+step_cargo['harvested']-step_cargo['deposited_to_either_base']-
                     step_cargo['smelt_fee']-step_cargo['hostile_base_confiscation'])
            old_trees={tid for pl in before['players'] for tid,(_,age) in pl[3].items() if age<cfg['treeLifespan']}
            threats=[tid for tid,(pos,_) in after['players'][1-seat][3].items()
                     if tid in old_trees and (cell==pos or cell in sim._nbr8[pos])]
            step_cargo['enemy_tree_tax']=len(threats)*cfg['collectorAbsorptionRate']*pre_tax
        balance=(opening+step_cargo['harvested']+step_cargo['enemy_cargo_captured']+step_cargo['friendly_cargo_received']
                 -step_cargo['deposited_to_either_base']-step_cargo['collision_cargo_lost']-step_cargo['move_fee']
                 -step_cargo['smelt_fee']-step_cargo['hostile_base_confiscation']-step_cargo['enemy_tree_tax'])
        near(balance,closing,f'worker cargo at {t}')
        cargo.update(step_cargo)
        if not closed:
            require(died_at is None and (dead or cell in sim.base_at), 'unexplained worker loss')
            died_at=t
            if not dead:
                losses.append({'post':t,'victim':uid,'cell':cell,'transfer':'hostile_base_removal','cargo':moved_load})
        life.append({'post':t,'order':action,'from':opened[0] if opened else None,'to':cell,'opening_cargo':opening,
                     'closing_cargo':closing,'survived':closed is not None,'flows':dict(step_cargo),'deposits':deposits,'purchases':purchases})
    require(result['score']==score and sim.step_count==expected_terminal, 'official terminal vs signed arm')
    first_income=sum(sum(tr['first_tenure'][seat].values()) for tr in trees.values())
    for tid in first:
        trees[tid]['first_tenure_end_reason']='owned_at_terminal'
    postings=money['own_base_deposits']+money['focal_loss_collision_credit_own']+first_income
    costs=money['hire_cost']+money['plant_cost']+money['seize_cost']
    return {'worker':uid,'role':role,'base':base,'born_at':start+1,'died_at':died_at,'alive_at_terminal':uid in sim.workers,
            'terminal_inventory':sim.workers[uid].carbon if uid in sim.workers else 0.,'own_focal_deposits':money['own_base_deposits'],
            'direct_enemy_kills':kills,'deaths_in_no_winner_collision_with_focal':ties,'focal_death':losses,
            'team_enemy_deaths_including_new_hires':all_enemy_deaths,'team_enemy_deaths_preexisting_workers':enemy_deaths,
            'enemy_deaths_not_direct_unique_wins_by_focal':all_enemy_deaths-len(kills),'cargo_flows':dict(cargo),'cash_flows':dict(money),
            'trees':trees,'first_tenure_tree_income':first_income,'first_tenure_income_by_acquisition':
            {k:sum(sum(tr['first_tenure'][seat].values()) for tr in trees.values() if tr['first_acquisition']==k) for k in ('plant','seize')},
            'first_tenure_tree_income_after_focal_death':sum(e['cash'] for tr in trees.values() for e in tr['receipts']
            if died_at is not None and e['post']>died_at and e['tenure']=='first_tenure'),
            'direct_cash_gross_including_first_tree_tenures':postings,'direct_cash_net_after_costs':postings-costs,
            'life':life,'team_cash_flows_root_to_end':events_by_seat,'terminal_cash':list(sim.cash),'official_score':result['score'],
            'terminal_step':sim.step_count}


def audit_case(case_dir):
    case_dir=Path(case_dir)
    config=read(case_dir/'live_protocol.json'); registration=read(case_dir.parent/'protocol.json')
    seed,seat=config['seed'],config['seat']; record=config['record']; root=config['root']
    registered,=[c for c in registration['cases'] if (c['seed'],c['seat'])==(seed,seat)]
    require(sha(case_dir.parent/'protocol.json')==config['study_protocol_sha256'] and
            all(registered[k]==v for k,v in record.items()) and all(registered[k]==v for k,v in root.items()),'case registration')
    raw_path=ROOT/record['source']; raw=read(raw_path)
    require(sha(raw_path)==record['source_sha256'] and raw_path.parent==ROOT/config['source_dir'], 'source tape')
    require((raw['seed'],raw['candidate_seat'])==(seed,seat), 'source game')
    sim=FastCarbon(raw['game_configuration']).reset(seed=seed)
    require(list(sim.carbon)==raw['initial_full_carbon'] and engine.digest(bytes(int(x) for x in sim.carbon))==raw['initial_terrain_u8_sha256'],'terrain')
    for joint in raw['actions'][:root['t']]:
        require(engine.twin_step(sim,joint,events=False)['status']==['ACTIVE','ACTIVE'],'source prefix ended')
    state=sim.get_state(); arms={}; receipts={}
    for arm in ('parent','alternative'):
        result_path=case_dir/f'live_{arm}.json'; result=read(result_path)
        tape=case_dir/f'live_{arm}_actions.json.gz'; tail=read(tape)
        require((result['arm'],result['seed'],result['seat'],result['root'])==(arm,seed,seat,root) and
                result['source_sha256']==registration['examiner_sha256'] and result['pack_sha256']==registration['parent_sha256']
                and result['action_tape']==tape.name and result['action_sha256']==sha(tape) and result['original_trajectory_sha256']==sha(raw_path)
                and result['protocol_sha256']==sha(case_dir/'live_protocol.json') and result['shared_init_sha256']==sha(case_dir/'live_shared_init.json')
                and result['statuses']==['DONE','DONE'] and result['search_calls']==len(tail)-1 and tail[0]==result['root_actions'], 'invalid signed arm')
        arms[arm]=run_arm(state,raw['game_configuration'],tail,seat,root['worker'],registered['root_check'][arm],result['score'],result['terminal_t'])
        receipts[arm]={'action_sha256':sha(tape),'result_sha256':sha(result_path)}
    return {'schema':'late-hire-focal-life-v1','seed':seed,'seat':seat,'root_t':root['t'],'arms':arms,'arm_receipts':receipts,
            'source_sha256':sha(raw_path),'study_protocol_sha256':config['study_protocol_sha256'],'case_protocol_sha256':sha(case_dir/'live_protocol.json'),
            'ledger_code_sha256':sha(Path(__file__)),'dual_transition_code_sha256':sha(STUDY),
            'scope':'Confirmed worker bank transfers and first uninterrupted tree-tenure receipts; direct unique-collector kills only. Transfers, inventory, team-wide cash and post-loss tree income are reported separately.'}


if __name__=='__main__':
    print(json.dumps(audit_case(Path(sys.argv[1])),ensure_ascii=False,allow_nan=False))
