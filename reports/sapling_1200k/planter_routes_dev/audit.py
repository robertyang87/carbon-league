"""Reproduce all route decisions, realized next-cell net earnings and paired phase accounts."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(BASE))
import tree_audited_game as arena
from turning_point_ledger import CashEvents


def check(run,game,attribution):
    packed=(run/game['raw_file']).read_bytes()
    if hashlib.sha256(packed).hexdigest()!=game['raw_sha256']: raise ValueError('tape changed')
    raw=json.loads(gzip.decompress(packed));seat=raw['candidate_seat']
    if attribution:
        cand=ROOT/'data/weights/sapling_1200k/planter_routes_it50_v3.py'
        parent=cand.with_name('hybrid_it50_v3.py')
        if arena.prior.digest(cand)!=raw['pack_sha256'] or arena.prior.digest(parent)!=json.loads((HERE/'plan.json').read_bytes())['parent_sha256']:
            raise ValueError('actor/counterfactual identity')
        live=arena.prior.shared.module(cand,'routes_live')
        ablated=arena.prior.shared.module(cand,'routes_without_sapling')
        original=arena.prior.shared.module(parent,'routes_parent')
        ablated._HYBRID['student_weight']=0.
    sim=arena.exam.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);sim.cash=CashEvents(sim.cash)
    counts=collections.Counter();changes=[];actions={};active={};trace={}
    mids=[collections.Counter(),collections.Counter()];all_cash=[collections.Counter(),collections.Counter()]
    for t,orders in enumerate(raw['actions']):
        if attribution:
            obs=sim.obs(seat);joint=live.agent(obs,sim.config);normal=original.agent(obs,sim.config);no_student=ablated.agent(obs,sim.config)
            if joint!=orders[seat]: raise ValueError(f'saved command does not reproduce {t}')
            def differences(ref):return {k for k in joint.keys()|ref.keys() if joint.get(k)!=ref.get(k)}
            adv=differences(normal);contrib=differences(no_student)
            counts['roots']+=1;counts['changed_without_sapling_roots']+=bool(contrib);counts['changed_without_sapling_orders']+=len(contrib)
            if bool(adv)!=bool(live._ROUTE_LAST):raise ValueError('independent change census mismatch')
            if adv:
                row=dict(live._ROUTE_LAST);uid=row['worker']
                if adv!={uid} or not joint.get(uid) or not normal.get(uid) or sim.workers[uid].is_collector:raise ValueError('unintended root rewrite')
                row.update(official_command=joint[uid],parent_command=normal[uid],purchased_next_action=False,first_tenure_net=0.)
                actions[t+1,uid,row['proposed_cell']]=row
                changes.append(row)
        sim.cash.events=[];sim.step(*orders)
        ids={tree.pos:tid for tid,tree in sim.trees.items()}
        for kind,s,delta,*detail in sim.cash.events:
            all_cash[s][kind]+=delta
            if 100<=t<200:mids[s][kind]+=delta
            if kind in ('plant','seize'):
                uid,cell=detail;tid=ids[cell]
                if tid in active:
                    active.pop(tid)['end_reason']='seized'
                if s==seat and (t,uid,cell) in actions:
                    item=actions[t,uid,cell];active[tid]=item
                    item.update(purchased_next_action=True,paid_kind=kind,paid_cost=-delta,tree_id=tid,
                                cash_receipts=0.,collision_cash=0.,enemy_worker_absorption=0.,first_tenure_net=delta,paid_turn=t+1)
            elif kind in ('tree_absorption','tree_collision_cash'):
                tid=detail[0] if kind=='tree_absorption' else detail[2]
                if tid in active:
                    item=active[tid]
                    if s!=seat:raise ValueError('credited wrong tree owner')
                    item['cash_receipts']+=delta;item['first_tenure_net']+=delta
                    if kind=='tree_collision_cash':item['collision_cash']+=delta
                    else:item['enemy_worker_absorption']+=detail[2]
                    item['last_income_turn']=t+1
        for tid in list(active):
            if tid not in sim.trees:active.pop(tid)['end_reason']='expired'
        if t+1 in (100,200,len(raw['actions'])):
            trace[t+1]={'cash':list(sim.cash),'trees':[sum(tree.player==s for tree in sim.trees.values()) for s in (0,1)]}
    for item in active.values():item['end_reason']='terminal'
    if arena.prior.hashed(sim.get_state())!=game['final_state_sha256']:raise ValueError('final state mismatch')
    return dict(seed=raw['seed'],seat=seat,raw_sha256=game['raw_sha256'],official_scores=game['official_score_by_seat'],
                margin=game['margin'],valid=game['valid'],steps=game['actual_transitions'],counts=dict(counts),changes=changes,
                funnel=dict(live._ROUTE_COUNTS) if attribution else {},trace=trace,mid_cash=mids,whole_game_cash=all_cash)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('which',choices=['candidate','parent']);p.add_argument('seat',type=int,choices=(0,1));args=p.parse_args()
    run=HERE/args.which;result=json.loads((run/'results.json').read_bytes())
    if result['status']!='complete' or sorted((r['seed'],r['candidate_seat']) for r in result['games'])!=[(621456920,0),(621456920,1)]:
        raise ValueError('wrong or incomplete run')
    g=next(g for g in result['games'] if g['candidate_seat']==args.seat)
    out=check(run,g,args.which=='candidate')
    (HERE/(f'{args.which}_economics_seat{args.seat}.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
