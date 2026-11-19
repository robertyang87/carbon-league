"""Independent official-environment and FastCarbon ledger replay of already completed natural branches."""
from collections import Counter
from pathlib import Path
import numpy as np
from run import HERE,P,R,game_raw,digest,compact,provenance,score,public,clock_result

def close(a,b):return np.allclose(a,b,atol=.011,rtol=0)

def replay(case,shared,arm):
    folder=HERE/f"{case['seed']}_{case['seat']}"
    raw=game_raw(case);reg=R.load(P);ticket=R.load(folder/(arm+'.json.gz'))
    assert ticket['identity']==shared['identity'] and ticket['shared_sha']==R.sha(folder/'shared.json.gz')
    assert ticket['state_sha']==shared['state_sha'] and ticket['snapshot_sha']==shared['snapshot_sha']
    assert ticket['case_terrain']==raw['initial_terrain_u8_sha256'] and ticket['host_at_start']['affinity']==ticket['host_at_end']['affinity']
    assert digest(compact(shared['prefix']))==shared['prefix_sha'] and digest(compact(ticket['tape']))==ticket['tape_sha']
    root=shared['root'];s=case['seat'];o=1-s;troot=root['t'];actions=shared['prefix']+ticket['tape']
    assert len(shared['prefix'])==troot and ticket['root_t']==troot and ticket['root_joint'][o]==root['enemy'] and ticket['root_joint'][s]==root[arm]
    assert shared['clock']['calls'][-1]['step']==troot and [v['step'] for v in ticket['clock']['calls']]==list(range(troot+1,len(actions)))
    assert len(shared['trace'])==troot+1 and len(actions)==len(shared['clock']['calls'])+len(ticket['clock']['calls'])
    original_early=[x for x in shared['trace'] if 24<=x['t']<=128 and x['changed']]
    assert len(original_early)==1 and original_early[0]['t']==troot
    env=R.live.begin(raw);sim=R.FastCarbon(raw['game_configuration']).reset(seed=case['seed'])
    assert digest(bytes(map(int,sim.carbon)))==case['terrain'];sim.cash=R.CashEvents(sim.cash)
    starting=list(sim.cash);focal={};owned={}
    ledger={k:[Counter(),Counter()] for k in ('whole','prefix','root','after','root_to_terminal')}
    for t,joint in enumerate(actions):
        assert not env.done and sim.step_count==t
        actual=[digest(compact(public(env,s))),digest(compact(public(env,o)))]
        if t<=troot:
            rr=shared['trace'][t];assert actual==[rr['own_sha'],rr['enemy_sha']]
            if t<troot:
                assert joint[s]==rr['parent'] and joint[o]==rr['enemy']
        else:assert actual==ticket['suffix_observations'][t-troot-1]
        if t==troot:
            for uid,d in root['changed'].items():
                obs=sim.obs(s);w=obs['players'][s][2][uid];pos=w[0]
                focal[uid]=dict(role=w[2],cell=pos,cargo=w[1],action=joint[s].get(uid,'NONE'),wallets_pre=list(sim.cash),
                    visible_neighbors=[obs['carbon'][q] for q in sim._nbr4[pos]],root_cash_events=[],purchased=None)
        sim.cash.events.clear();sim.step(*joint);env.step(joint)
        if t==troot:
            for uid,entry in focal.items():
                entry['survived_first_joint']=uid in sim.workers
                entry['position_after_first_joint']=sim.workers[uid].pos if uid in sim.workers else None
                for e in sim.cash.events:
                    if len(e)>4 and e[0] not in ('tree_absorption','rounding') and e[3]==uid:
                        entry['root_cash_events'].append(list(e))
                        if e[0] in ('plant','seize'):
                            tree=next(z for z in sim.trees.values() if z.pos==e[4] and z.player==s)
                            entry['purchased']=dict(id=tree.id,cell=tree.pos,cost=-e[2],ground=0.,tax=0.,held_through_t=t,first_loss_at=None)
                            owned[tree.id]=entry['purchased']
        for tid,hold in owned.items():
            tree=sim.trees.get(tid)
            if hold['first_loss_at'] is None:
                if tree is None or tree.player!=s:hold['first_loss_at']=t
                else:hold['held_through_t']=t

        assert close(sim.cash,[p[0] for p in env.states[0].observation.players])
        for kind,who,amount,*fields in sim.cash.events:
            if kind=='tree_absorption' and fields[0] in owned and who==s and owned[fields[0]]['first_loss_at'] is None:
                owned[fields[0]]['ground']+=amount-fields[2];owned[fields[0]]['tax']+=fields[2]
            values={'ground_rent':amount-fields[2],'enemy_cargo_tax':fields[2]} if kind=='tree_absorption' else {kind:amount}
            for group in (['whole','prefix'] if t<troot else ['whole','root_to_terminal','root' if t==troot else 'after']):
                ledger[group][who].update(values)
    unrounded=[sum(c.values()) for c in ledger['whole']]
    assert np.max(np.abs(np.subtract(sim.cash,starting)-unrounded))<1e-6
    scores,cash,status=score(env);assert scores==ticket['official_scores'] and cash==ticket['physical_cash'] and status==ticket['status']
    assert scores[s]-scores[o]==ticket['official_margin']
    return dict(arm=arm,seed=case['seed'],seat=s,terrain=case['terrain'],cpu=shared['cpu'],transitions=len(actions),
                root=troot,root_changed=root['changed'],root_orders=ticket['root_joint'],original_root_clock=shared['clock']['calls'][-1],
                shared_prefix_calls=len(shared['clock']['calls']),remaining_original_calls=len(ticket['clock']['calls']),
                prefix_clock=shared['clock']['summary'],suffix_clock=ticket['clock']['summary'],whole_arm_elapsed_s=ticket['wall_s'],
                score=scores,terminal_status=status,terminal_cash=cash,official_margin=scores[s]-scores[o],ledger=ledger,
                root_workers=focal,receipt=R.sha(folder/(arm+'.json.gz')),shared=R.sha(folder/'shared.json.gz'))

def verify_zero(case,shared):
    src=game_raw(case);env=R.live.begin(src);s=case['seat'];o=1-s
    assert not any(24<=r['t']<=128 and r['changed'] for r in shared['trace'])
    assert digest(compact(shared['prefix']))==shared['prefix_sha']
    assert len(shared['prefix'])==len(shared['trace'])==len(shared['clock']['calls'])
    for t,(trace,joint,call) in enumerate(zip(shared['trace'],shared['prefix'],shared['clock']['calls'],strict=True)):
        assert t==trace['t']==call['step']
        assert digest(compact(public(env,s)))==trace['own_sha']
        assert digest(compact(public(env,o)))==trace['enemy_sha']
        assert joint[s]==trace['parent'] and joint[o]==trace['enemy']
        env.step(joint)
    assert env.done or shared['trace'][-1]['t']==128
    return dict(seed=case['seed'],seat=s,changed=False,n_prefix=len(shared['trace']),first_t=0,last_t=shared['trace'][-1]['t'],
                per_step_clock=shared['clock']['summary'],prefix_sha=shared['prefix_sha'],sha=R.sha(HERE/f"{case['seed']}_{s}"/'shared.json.gz'))

def main():
    reg=R.load(P);ids=provenance();cards=[]
    for case in reg['cases']:
        folder=HERE/f"{case['seed']}_{case['seat']}";shared=R.load(folder/'shared.json.gz')
        assert shared['identity']==ids and shared['case']==case
        if shared['root'] is None:
            cards.append(verify_zero(case,shared))
            continue
        arms={arm:replay(case,shared,arm) for arm in case['branch_order']}
        data={arm:R.load(folder/(arm+'.json.gz')) for arm in ('parent','student')}
        s=case['seat'];o=1-s;A=arms['student'];B=arms['parent'];t0=shared['root']['t']
        n=min(map(lambda a:len(a['tape']),data.values()))
        first_enemy_divergence=next((t0+i for i in range(1,n) if data['parent']['tape'][i][o]!=data['student']['tape'][i][o]),None)
        net={}
        keys=set().union(*(x.keys() for arm in arms.values() for x in arm['ledger']['root_to_terminal']))
        for kind in keys:
            def diff(arm):
                l=arm['ledger']['root_to_terminal'];return l[s][kind]-l[o][kind]
            net[kind]=float(diff(A)-diff(B))
        cards.append(dict(seed=case['seed'],seat=s,changed=True,root_t=t0,student_minus_parent_margin=A['official_margin']-B['official_margin'],
                          first_later_enemy_command_difference=first_enemy_divergence,component_net_own_minus_rival_student_minus_parent=net,
                          opposing_rng_at_root_identical=True,arms=arms))
    R.write(HERE/'result.json',dict(protocol=R.sha(P),qualification=R.sha(HERE/'qualified.json'),settlement_program=R.sha(Path(__file__)),
        maps=cards,original_full_map_wins_claimed=0,qualified_for_ladder=False,
        conditional_claim='Single worker-command divergence step, naturally responding archived-map opponent with one paired starting state per map; no population inference.'))

if __name__=='__main__':main()
