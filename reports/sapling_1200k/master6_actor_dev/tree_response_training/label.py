"""Verify each true-response arm, and label pay minus defer with whole-game score and separate cash channels."""
from collections import Counter
import math
from statistics import mean
from pathlib import Path
from pair_games import records
from select_roots import BASE,HERE,PLAN,ROOT,clean,load,retain,sha
from branches import begin,public,utility
from turning_point_ledger import CashEvents, FastCarbon


def arm(reg,name):
    wd=HERE/reg['case_directory']
    root=load(wd/'root.json.gz');raw=load(ROOT/reg['source'])
    rec=load(wd/f'live_{name}.json');actions_file=wd/rec['action_tape'];tail=load(actions_file)
    s,t,uid,cell=reg['seat'],reg['step'],reg['uid'],reg['cell']
    if (sha(actions_file)!=rec['action_sha256'] or sha(wd/'live_protocol.json')!=rec['protocol_sha256']
            or sha(wd/'live_shared_init.json')!=rec['shared_init_sha256'] or rec['arm']!=name
            or rec['pack_sha256']!=PLAN['parent_pack_sha256'] or rec['search_calls']!=len(tail)-1):
        raise ValueError('invalid response identity or missing live opponent call')
    env=begin(raw);sim=FastCarbon(raw['game_configuration']).reset(seed=reg['seed']);sim.cash=CashEvents(sim.cash)
    money=[Counter(),Counter()];root_debit=0.;orig_tree=None;first_ended_at=None;first_tree_income=0.
    root_alive=None;initial_cash=None
    for i,pair in enumerate(raw['actions'][:t]+tail):
        if env.done or sim.step_count!=i:
            raise ValueError('unexpected end of episode')
        if i==t:
            before=public(env,s)
            if any(before[k]!=root['public_at_root'][k] for k in ('players','trees','carbon','step','player')):
                raise ValueError('different public state')
            expected=root['v1'] if name=='parent' else root['alternative']
            if clean(pair[s])!=expected or clean(pair[1-s])!=clean(raw['actions'][t][1-s]):
                raise ValueError('different intervention or opponent has looked at the candidate')
            initial_cash=list(sim.cash)
        sim.cash.events.clear()
        env.step(pair);sim.step(*pair)
        if env.states[0].observation.players!=sim.obs(s)['players']:
            raise ValueError(f'dual interpreter disagreement at transition {i}')
        if i<t:
            continue
        entries=sim.cash.events
        for e in entries:
            money[e[1]][e[0]]+=e[2]
        if i==t:
            root_alive=uid in sim.workers
            pays=[e for e in entries if e[0] in ('plant','seize') and e[1]==s and e[3]==uid and e[4]==cell]
            if len(pays)>1:
                raise ValueError('one worker cannot pay twice at root')
            if pays:
                root_debit=-pays[0][2]
                orig_tree=next(x.id for x in sim.trees.values() if x.pos==cell and x.player==s)
        if orig_tree is not None and first_ended_at is None:
            for e in entries:
                if e[1]==s and ((e[0]=='tree_absorption' and e[3]==orig_tree) or (e[0]=='tree_collision_cash' and e[5]==orig_tree)):
                    first_tree_income+=e[2]
            if orig_tree not in sim.trees or sim.trees[orig_tree].player!=s:
                first_ended_at=i+1
    scores=[v.reward for v in env.states];statuses=[v.status for v in env.states]
    if initial_cash is None or not env.done or scores!=rec['score'] or statuses!=['DONE','DONE'] or statuses!=rec['statuses']:
        raise ValueError('source/terminal official score mismatch')
    if rec['terminal_t']!=t+len(tail) or rec['margin']!=scores[s]-scores[1-s] or not math.isclose(rec['u'],utility(scores,s)):
        raise ValueError('official utility mismatch')
    for k in (0,1):
        if not math.isclose(sum(money[k].values())+initial_cash[k],sim.cash[k],abs_tol=1e-6):
            raise ValueError('cash ledger does not reconcile')
    return {'native_arm':name,'scores_by_physical_seat':scores,'official_margin':scores[s]-scores[1-s],'U':rec['u'],
            'wdl':'win' if scores[s]>scores[1-s] else 'loss' if scores[s]<scores[1-s] else 'draw',
            'root_debit':root_debit,'quoted':reg['price'],'root_worker_survived':root_alive,'originating_tree_id':orig_tree,
            'first_tenure_ended_at':first_ended_at,'first_tenure_income':first_tree_income,'first_tenure_net':first_tree_income-root_debit,
            'cash_events_from_root_by_physical_seat':money,'official_terminal_t':rec['terminal_t'],
            'statuses':statuses,'examiner_calls':rec['search_calls'],'multiple_candidate_searches':rec['search_completed'],
            'search_aborts':rec['search_aborted'],'elapsed':rec['elapsed'],'actual_action_tape_sha256':rec['action_sha256'],
            'shared_stream_sha256':rec['shared_init_sha256']},tail


def label(reg):
    pairs={name:arm(reg,name) for name in ('parent','alternative')}
    buy=reg['paid_arm'];wait='alternative' if buy=='parent' else 'parent'
    a,act_a=pairs[buy];b,act_b=pairs[wait];s=reg['seat']
    if a['shared_stream_sha256']!=b['shared_stream_sha256'] or b['root_debit']!=0:
        raise ValueError('not a paired buy/defer case')
    if a['root_debit'] and not math.isclose(a['root_debit'],reg['price'],abs_tol=1e-7):
        raise ValueError('prospective quote different from actually charged price')
    first=next((reg['step']+j for j,(pa,pb) in enumerate(zip(act_a,act_b)) if clean(pa[1-s])!=clean(pb[1-s])),None)
    score_a,score_b=a['scores_by_physical_seat'],b['scores_by_physical_seat']
    delta_own,delta_rival=score_a[s]-score_b[s],score_a[1-s]-score_b[1-s]
    delta_margin=a['official_margin']-b['official_margin']
    if not math.isclose(delta_margin,delta_own-delta_rival,rel_tol=1e-12,abs_tol=1e-8):
        raise ValueError('pay-minus-defer sign/convention mismatch')
    return {'seed':reg['seed'],'seat':s,'t':reg['step'],'window':reg['window'],'focal_uid':reg['uid'],'cell':reg['cell'],
            'root_sha256':reg['root_sha256'],'features':reg['case_directory']+'/root.json.gz','observed_stratum':reg['source_stratum'],
            'requested_stratum':reg['expected_stratum'],'v1_action':reg['old'],'alternative_action':reg['new'],'pay_is_v1':buy=='parent',
            'pay_intent_filled':bool(a['root_debit']),'delta_own_official_score':delta_own,'delta_rival_official_score':delta_rival,
            'target_delta_official_margin':delta_margin,'delta_U':a['U']-b['U'],'first_different_opponent_command':first,'pay':a,'defer':b}


def finish():
    selected=records()
    rows=[]
    for reg in selected:
        r=label(reg)
        retain(HERE/reg['case_directory']/'training_label.json',r)
        rows.append(r)
    covered=sorted({r['seed'] for r in rows})
    summary={s:{'pairs':sum(r['seed']==s for r in rows),'mean_local_effect':mean(r['target_delta_official_margin'] for r in rows if r['seed']==s)}
             for s in covered}
    out={'schema':'tree-response-training-terminal-v1','protocol_sha256':sha(HERE/'protocol.json'),'selection_sha256':sha(HERE/'selection.json'),
         'pairs':rows,'unique_source_maps':len(covered),'map_summary':summary,'paid':sum(r['pay_intent_filled'] for r in rows),
         'positive_labels':sum(r['target_delta_official_margin']>0 for r in rows),'negative_labels':sum(r['target_delta_official_margin']<0 for r in rows),
         'context_counts':Counter(''.join(map(str,r['observed_stratum'])) for r in rows),'trained_head':False,'reserved_development_games':0,
         'fit_reason':'Two fixed training maps. No adequate across-map training support; independent held-out maps were not opened.'}
    retain(HERE/'training_labels.json',out)
    return out

if __name__=='__main__':
    doc=finish()
    print({k:doc[k] for k in ('map_summary','paid','positive_labels','negative_labels')},flush=True)
