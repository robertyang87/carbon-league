"""Recompute frozen training labels from both tapes, the official referee, and the cash ledger."""
from collections import Counter
import gzip
from pathlib import Path
import numpy as np
from collect import HERE,K,R

def digest(value):
    return K.digest(K.compact(value))

def replay(case, folder, shared, arm):
    card=case['card']; seat=case['seat'];other=1-seat;root=case['t']
    raw=R.load(R.ROOT/card['path']);path=folder/(arm+'.json.gz');ticket=R.load(path)
    assert ticket['binding']==shared['binding'] and ticket['root']==case
    assert ticket['root_receipt']==R.sha(folder/'root.json.gz') and ticket['state_sig']==shared['state_sig']
    assert digest(ticket['tape'])==ticket['tape_sha']
    prefix=raw['actions'][:root]; assert digest(prefix)==case['historical_prefix_sha']==shared['historical_prefix_joint_sha']
    actions=prefix+ticket['tape']
    assert actions[root]==K.physical(case[arm],shared['original'],seat)
    assert raw['actions'][root][other]==case['historical_enemy']==shared['archived_enemy']
    assert [x['step'] for x in shared['clock']['calls']]==[root]
    clock=ticket['clock']['calls']
    assert [c['step'] for c in clock]==list(range(root+1,len(actions)))
    assert ticket['host_at_end']['affinity']==shared['host']['affinity']==[ticket['binding']['cpu']]
    env=K.checked_env(raw);sim=R.FastCarbon(raw['game_configuration']).reset(seed=card['seed'])
    assert K.digest(bytes(map(int,sim.carbon)))==card['terrain']
    sim.cash=R.CashEvents(sim.cash); cash0=list(sim.cash)
    ledgers={g:[Counter(),Counter()] for g in ('prefix','root','suffix','post_root_to_terminal','whole')}
    trajectory=[];worker_history=[];transitions=[];uid=case['focal_uid']
    for i,joint in enumerate(actions):
        assert not env.done and sim.step_count==i
        obs=K.public(env,seat)
        public_hashes=[digest(obs),digest(K.public(env,other))]
        if i<=root: assert public_hashes==case['full_public_hashes'][i]
        else: assert public_hashes==ticket['suffix_public_sha'][i-root-1]
        if i>=root:
            position=[dict(cash=float(team[0]),workers=len(team[2]),trees=len(team[3]),
                           collector_cargo=sum(w[1] for w in team[2].values() if w[2]=='COLLECTOR')) for team in obs['players']]
            record=dict(t=i,pre_by_seat=position,joint=joint,public_sha=public_hashes,cash_events=[])
            transitions.append(record)
        sim.cash.events.clear();sim.step(*joint);env.step(joint)
        cash=[float(p[0]) for p in env.states[0].observation.players]
        np.testing.assert_allclose(cash,sim.cash,rtol=0,atol=.011)
        if i>=root:
            trajectory.append(cash)
            unit=sim.workers.get(uid)
            worker_history.append(None if unit is None else [int(unit.pos),float(unit.carbon)])
        groups=['whole','prefix'] if i<root else ['whole','post_root_to_terminal','root' if i==root else 'suffix']
        for kind,who,amount,*fields in sim.cash.events:
            components={'ground_rent':amount-fields[2],'enemy_cargo_tax':fields[2]} if kind=='tree_absorption' else {kind:amount}
            for group in groups:ledgers[group][who].update(components)
            if i>=root:record['cash_events'].append([who,kind,amount,*fields])
    official,cash,status=K.score(env)
    assert (official,cash,status)==(ticket['scores'],ticket['cash'],ticket['status']) and official[seat]-official[other]==ticket['margin']
    np.testing.assert_allclose([sum(a.values()) for a in ledgers['whole']],np.subtract(sim.cash,cash0),rtol=0,atol=.000001)
    times=[c['public_wall_seconds'] for c in shared['clock']['calls']+clock]
    return dict(label_role='incumbent' if arm=='parent' else 'on-support alternative',raw_result_sha256=R.sha(path),
                full_prefix_and_suffix_transitions=len(actions),official_score_by_seat=official,physical_cash_by_seat=cash,
                official_margin_for_candidate=ticket['margin'],status=status,cash_ledgers=ledgers,
                branch_elapsed_seconds=ticket['wall_s'],examiner_cpu_affinity=shared['host']['affinity'],
                original_root_call=shared['clock']['calls'][0],original_suffix_clock=ticket['clock']['summary'],
                original_public_wall_seconds=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),
                    p99=float(np.percentile(times,99)),maximum=max(times)),
                root_to_end_cash_by_seat=trajectory,root_to_end_focal_position_cargo=worker_history), transitions

def paired(case):
    folder=HERE/('root_'+case['root_id'][:16]);shared=R.load(folder/'root.json.gz')
    assert shared['snapshot']==K.digest(gzip.decompress((folder/'root_state.pickle.gz').read_bytes()))
    assert shared['actual_normal']=={'parent':case['parent'],'student':case['expected_free_student']}
    assert shared['binding']['batch']==R.sha(HERE/'batch.json')
    assert shared['binding']['source']==case['card']['sha256']
    pairs={a:replay(case,folder,shared,a) for a in ('parent','student')}
    arms={k:v[0] for k,v in pairs.items()};steps={k:v[1] for k,v in pairs.items()}
    tapes={a:R.load(folder/(a+'.json.gz'))['tape'] for a in arms}
    s=case['seat'];o=1-s;t=case['t'];arm_s,arm_p=arms['student'],arms['parent']
    n=min(len(v) for v in tapes.values())
    enemy_change=next((t+i for i in range(1,n) if tapes['student'][i][o]!=tapes['parent'][i][o]),None)
    h={}
    snapshots=(1,3,5,10,24,50,100)
    for k in snapshots:
        if n<k:continue
        c1,c0=[a['root_to_end_cash_by_seat'][k-1] for a in (arm_s,arm_p)]
        h[k]=dict(delta_own=c1[s]-c0[s],delta_enemy=c1[o]-c0[o],
            delta_cash_margin=(c1[s]-c1[o])-(c0[s]-c0[o]))
    components={}
    ledger=arm_p['cash_ledgers']['post_root_to_terminal']+arm_s['cash_ledgers']['post_root_to_terminal']
    for kind in sorted(set().union(*(c.keys() for c in ledger))):
        d0=ledger[2+s][kind]-ledger[s][kind];d1=ledger[2+o][kind]-ledger[o][kind]
        components[kind]=dict(delta_own=d0,delta_enemy=d1,delta_cash_margin=d0-d1)
    delta=arm_s['official_margin_for_candidate']-arm_p['official_margin_for_candidate']
    # Fixed operational turning points. For each horizon retain both directions, and obtain the real
    # actions and payments in both worlds. This describes trajectories and cannot localize causation.
    gaps=[0.]+[(x[s]-x[o])-(y[s]-y[o]) for x,y in zip(arm_s['root_to_end_cash_by_seat'],arm_p['root_to_end_cash_by_seat'])]
    turning={}
    for width in (3,5,10):
        if n<width:continue
        moves=np.asarray(gaps[width:])-np.asarray(gaps[:-width])
        selected={}
        for direction,choice in [('rise',int(np.argmax(moves))),('fall',int(np.argmin(moves)))]:
            selected[direction]=dict(start_step=t+choice,last_step=t+choice+width-1,change_in_paired_cash_gap=float(moves[choice]),
                    initial_paired_gap=gaps[choice],closing_paired_gap=gaps[choice+width],
                    prior_actions_and_events_by_arm={a:steps[a][choice:choice+width] for a in steps})
        turning[width]=selected
    return dict(seed=case['card']['seed'],terrain=case['card']['terrain'],seat=s,step=t,uid=case['focal_uid'],role=case['role'],
               commands=[case['parent'].get(case['focal_uid'],'NONE'),case['student'].get(case['focal_uid'],'NONE')],
               root_id=case['root_id'],shared_receipt_sha=R.sha(folder/'root.json.gz'),shared_snapshot_sha=shared['snapshot'],
               archived_root_enemy=shared['archived_enemy'],live_root_enemy=shared['original'],
               same_archived_and_live_root_enemy=shared['archived_enemy']==shared['original'],
               first_later_enemy_joint_difference=enemy_change,official_margin_change_proposal_minus_incumbent=delta,
               deltas_at_horizons=h,cash_components_proposal_minus_incumbent=components,largest_rise_and_fall_by_window=turning,arms=arms)

def main():
    cases=R.load(HERE/'batch.json')['roots'];assert len(cases)==4
    interrupt=R.load(HERE/'interruption.json')
    assert interrupt['batch']==R.sha(HERE/'batch.json') and interrupt['collector']==R.sha(HERE/'collect.py')
    for case,record in zip(cases,interrupt['roots'],strict=True):
        folder=HERE/('root_'+case['root_id'][:16])
        assert R.sha(folder/'root.json.gz')==record['root'] and R.sha(folder/'root_state.pickle.gz')==record['stored_state_blob']
        for arm,sha in record['completed_preserved'].items():
            assert R.sha(folder/(arm+'.json.gz'))==sha
        for arm in record['missing_to_resume']:
            ticket=R.load(folder/(arm+'.json.gz'))
            assert ticket['state_sig']==record['state_sig'] and ticket['root_receipt']==record['root']
    rows=[paired(c) for c in cases];ys=[r['official_margin_change_proposal_minus_incumbent'] for r in rows]
    R.write(HERE/'labels.json.gz',dict(frozen_batch=R.sha(HERE/'batch.json'),release=R.sha(HERE/'collection_authorized.json'),
            independent_settlement_code=R.sha(Path(__file__)),recovery_evidence=R.sha(HERE/'interruption.json'),six_preinterruption_receipts_unchanged=True,roots=rows,all_four_labels_in_frozen_order=ys,
            above_zero=sum(y>0 for y in ys),zero=sum(y==0 for y in ys),below_zero=sum(y<0 for y in ys),
            independent_full_games_won_claimed=0,interpretation='One paired original-opponent continuation per training-only proposal, with a common enemy move at the root and fresh natural responses afterwards. Includes wall-clock and sampling variability. No estimate of whole-model strength or conditional expectation per action.'))
    # Small loss-input sidecar. Keep the terminal labels separate from inference features and
    # signed public prefix. These dependent single-draw labels have no estimated precision.
    examples=[]
    for c,r in zip(cases,rows,strict=True):
        arm_p=r['arms']['parent'];arm_s=r['arms']['student'];source=c['card']
        examples.append(dict(split='train',root_id=c['root_id'],prefix_actions_sha256=c['historical_prefix_sha'],
            signed_observation_sha256=c['public_sha256'],source=source['path'],source_sha256=source['sha256'],terrain=source['terrain'],
            seed=source['seed'],seat=c['seat'],step=c['t'],uid=c['focal_uid'],role=c['role'],parent=c['parent'],proposal=c['student'],
            proposal_prior=c['policy_probability'],prior_rank=c['policy_rank'],visible_input_contrast_terrain_carbon=c['all_signed_contrasts'],
            true_candidate_argmax=c['expected_free_student'],label_metadata_opponent_root_after_decision=r['live_root_enemy'],label_pair_replicates=1,
            parent_official_scores=arm_p['official_score_by_seat'],proposal_official_scores=arm_s['official_score_by_seat'],
            delta_official_margin=r['official_margin_change_proposal_minus_incumbent'],parent_receipt=arm_p['raw_result_sha256'],
            proposal_receipt=arm_s['raw_result_sha256']))
    R.write(HERE/'train_pairs.json',dict(format='sapling-historical-action-contrast-v1',batch=R.sha(HERE/'batch.json'),
        settlement=R.sha(HERE/'labels.json.gz'),use='Training targets only; terminal scores, opponent simultaneous root move and recovered trajectory are forbidden as inputs to the acting policy.',examples=examples))

if __name__=='__main__':main()
