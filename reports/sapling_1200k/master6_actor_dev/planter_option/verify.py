"""Replay every executed branch independently and reconcile the root charge and first tree tenure."""
from collections import Counter
import math
from pathlib import Path

from responses import BASE,HERE,read,retain,sha,setup
from screen import FastCarbon,CashEvents
from branches import begin,clean,public,utility

def branch(reg,arm):
    root=read(HERE/reg['root'])
    case=HERE/f"response_{reg['seed']}_{reg['seat']}"
    receipt=read(case/f'live_{arm}.json')
    log=case/receipt['action_tape']
    tail=read(log)
    tape=read(HERE.parents[3]/reg['source'])
    if sha(log.read_bytes())!=receipt['action_sha256'] or sha((case/'live_protocol.json').read_bytes())!=receipt['protocol_sha256']:
        raise ValueError('changed response or root')
    t0,s=reg['t'],reg['seat']
    commands=tape['actions'][:t0]+tail
    if len(commands)!=receipt['terminal_t'] or receipt['search_calls']!=len(tail)-1:
        raise ValueError('incomplete responsive continuation')
    env=begin(tape)
    sim=FastCarbon(tape['game_configuration']).reset(seed=reg['seed'])
    sim.cash=CashEvents(sim.cash)
    agg=[Counter(),Counter()]
    uid,cell=reg['worker'],reg['cell']
    debit=0.;first_income=0.;tree_id=None;ended=None;tracked=0;root_log=[]
    for k,pair in enumerate(commands):
        if env.done or sim.step_count!=k:
            raise ValueError('continuation overran terminal or began at incorrect step')
        if k==t0:
            joint=root['root_parent_joint'] if arm=='parent' else root['root_option_joint']
            if clean(pair[s])!=joint or clean(pair[1-s])!=clean(root['root_opponent_fixed']):
                raise ValueError('root submitted commands differ')
            for name in ('player','step','players','carbon','trees'):
                if public(env,s)[name]!=root['public_at_exercise'][name]:
                    raise ValueError('public root differs from parent ' + name)
        sim.cash.events.clear()
        sim.step(*pair)
        env.step(pair)
        # FastCarbon's .obs is canonical; compare visible assets/cash at every state.
        if sim.obs(s)['players']!=env.states[0].observation.players:
            raise ValueError(f'official and independent economic replay disagree at {k}')
        if k<t0:
            continue
        events=sim.cash.events
        for e in events:
            agg[e[1]][e[0]]+=e[2]
        if k==t0:
            root_log=[e for e in events if len(e)>4 and e[0] in ('plant','seize') and e[1]==s and e[3]==uid and e[4]==cell]
            if len(root_log)>1:
                raise ValueError('multiple charged transactions from one planter')
            debit=sum(-e[2] for e in root_log)
            if debit:
                tree_id=next(t.id for t in sim.trees.values() if t.pos==cell and t.player==s)
        if tree_id and ended is None:
            for e in events:
                if e[0]=='tree_absorption' and e[1]==s and e[3]==tree_id:
                    first_income+=e[2]
                elif e[0]=='tree_collision_cash' and e[1]==s and e[5]==tree_id:
                    first_income+=e[2]
            if tree_id not in sim.trees or sim.trees[tree_id].player!=s:
                ended=k+1
            else:
                tracked+=1
    results=[v.reward for v in env.states]
    statuses=[v.status for v in env.states]
    if not env.done or statuses!=['DONE','DONE'] or statuses!=receipt['statuses'] or results!=receipt['score']:
        raise ValueError('official terminal label differs')
    margin=results[s]-results[1-s]
    if margin!=receipt['margin'] or not math.isclose(utility(results,s),receipt['u']):
        raise ValueError('receipt score differs')
    return {'official_scores_by_seat':results,'margin':margin,'U':receipt['u'],'root_debit':debit,'root_ledger_entries':root_log,
            'root_quote':root['quote']['price'],'first_tenure_income':first_income,'first_tenure_net':first_income-debit,
            'originating_tree':tree_id,'first_tenure_ended_at_t':ended,'held_after_t_count':tracked,'remaining_cash_events':agg,
            'statuses':statuses,'terminal_t':receipt['terminal_t'],'original_search_calls':receipt['search_calls'],
            'multi_candidate_comparisons':receipt['search_completed'],'single_candidate':receipt['search_single'],
            'shared_initial_rng_sha256':receipt['shared_init_sha256'],'elapsed':receipt['elapsed'],'commands_sha256':sha(log.read_bytes())},tail

def main():
    rows=[]
    for reg in setup():
        p,t0=branch(reg,'parent')
        a,t1=branch(reg,'alternative')
        if (p['shared_initial_rng_sha256']!=a['shared_initial_rng_sha256'] or p['root_debit']!=0 or a['root_debit']<=0 or
                not math.isclose(a['root_debit'],a['root_quote'],abs_tol=1e-7)):
            paid=False
        else:
            paid=True
        opponent=1-reg['seat']
        response=next((reg['t']+i for i,(old,new) in enumerate(zip(t0,t1)) if clean(old[opponent])!=clean(new[opponent])),None)
        rows.append({'seed':reg['seed'],'seat':reg['seat'],'t':reg['t'],'worker':reg['worker'],'parent':p,'option':a,
                     'exact_option_purchase':paid,'delta_official_margin':a['margin']-p['margin'],'delta_U':a['U']-p['U'],
                     'first_different_actual_opponent_command':response})
    expanded=all(x['exact_option_purchase'] and x['delta_official_margin']>0 for x in rows)
    retain(HERE/'response_results.json',{'cases':rows,'eligible_for_registered_whole_game_transfer':expanded,
                                        'conditional_only':True,'followup_gate_sha256':sha((HERE/'followup_gate.json').read_bytes())})
    print([(r['t'],r['option']['root_debit'],r['option']['first_tenure_net'],r['delta_official_margin']) for r in rows],expanded)

if __name__=='__main__':
    main()
