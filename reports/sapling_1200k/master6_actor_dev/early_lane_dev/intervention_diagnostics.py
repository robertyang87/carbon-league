"""Decompose every registered one-step change; future positions refer only to factual v1 routes."""
from collections import Counter
import json
from pathlib import Path
import sys

from gate import load_games
from train import HERE, batch, check, distance, extract

sys.path.insert(0,str(HERE.parent/'return_denial_training'))
from ledger import one


def outcome(obs,pay,t,h,s,uid):
    w=obs[t+h]['players'][s][2].get(uid)
    paid=any(uid in f for f in pay[t:t+h])
    return dict(cell=None if w is None else w[0],cargo=None if w is None else w[1],
                died=w is None,unloaded=paid,low=w is not None and w[1]<100,
                label=-1 if w is None or w[1]<100 or paid else w[0])


def game(spec,validation):
    src=extract.ROOT/spec['raw_path']
    check(extract.sha(src)==validation['source_sha256']==spec['raw_sha256'],'source digest')
    raw=extract.load(src);cfg=raw['game_configuration'];seat=spec['seat']
    sim=extract.FastCarbon(cfg).reset(seed=spec['seed']); shadow=extract.FastCarbon(cfg)
    picked={r['t']:r for r in validation['decisions'] if r['issued']}
    hist=[];payments=[];rows=[]
    for t,joint in enumerate(raw['actions']):
        hist.append(sim.obs(seat))
        if t in picked:
            r=picked[t];c=r['candidate'];uid=c['own_uid'];tar=c['target_uid'];snap=sim.get_state()
            before=sim.full_observation()
            cmds=[o.copy() for o in joint]
            check((joint[seat].get(uid) or 'NONE')==c['old'],'source mover order')
            if c['order']=='NONE':cmds[seat].pop(uid,None)
            else:cmds[seat][uid]=c['order']
            arms=[]
            for j in (joint,cmds):
                shadow.set_state(snap);after,ev=extract.step(shadow,before,cfg,j)
                ledger=one(before,after,shadow,ev,seat,uid,tar,j[seat],j[1-seat],cfg)
                focal=after['players'][seat][2].get(uid);enemy=after['players'][1-seat][2].get(tar)
                arms.append(dict(own_ledger=ledger,own_uid_after=focal,target_after=enemy,
                            same_destination=bool(focal and enemy and focal[0]==enemy[0]),
                            own_deposit=sum(e[2] for e in ev if e[0]=='own_delivery' and e[1]==seat and e[3]==uid),
                            collision_target=ledger['enemy_collision']))
            old,new=arms; o=old['own_ledger'];n=new['own_ledger'];g=r['registered_single_step_delta']
            check(abs(old['own_deposit']-g['normal_delivery_v1'])<1e-6 and abs(new['own_deposit']-g['normal_delivery_alternative'])<1e-6 and
                  (new['own_uid_after'] is not None)==g['mover_survived'] and (new['target_after'] is not None)==g['target_survived'],'frozen one-step result')
            observed_order=joint[1-seat].get(tar) or 'NONE'
            en0=before['players'][1-seat][2][tar][0]
            dest=shadow._move_nbr[observed_order][en0] if observed_order!='NONE' else en0
            diff={k:n[k][seat]-o[k][seat] for k in ('real_new_harvest','move_cost_burn','cash_change','cross_to_worker','collision_cross_received')}
            diff['cash_margin']=diff['cash_change']-(n['cash_change'][1-seat]-o['cash_change'][1-seat])
            diff['focal_own_deposit']=new['own_deposit']-old['own_deposit']
            rows.append(dict(t=t,signature=r['signature'],oid=uid,target=tar,cells=c['cells'],old=c['old'],alternative=c['order'],
                         own_root=c['own_origin'],old_destination=c['old_cell'],new_destination=c['cell'],enemy_root=en0,
                         observed_enemy_order=observed_order,observed_enemy_destination=dest,
                         prospective_colocation=(c['cell']==dest),baseline_colocation=(c['old_cell']==dest),delta=diff,arms=arms))
        sim.cash=extract.CashEvents(sim.cash)
        sim.step(*joint)
        payments.append({e[3] for e in sim.cash.events if e[0] in ('own_delivery','opponent_worker_delivery_to_this_base') and e[2]>0})
    hist.append(sim.obs(seat))
    check([round(v,2) for v in sim.cash]==spec['scores'],'terminal mismatch')
    for i in rows:
        t=i['t'];r=picked[t];w=i['target'];u=i['oid']
        paths=[]
        for h in (3,5):
            y=outcome(hist,payments,t,h,1-seat,w)
            endpoint=i['cells'][int(h==5)]
            check(y['label']==r['original_trace_endpoints'][int(h==5)],'route label')
            ys=[hist[t+d]['players'][1-seat][2].get(w) for d in range(1,h+1)]
            xs=[hist[t+d]['players'][seat][2].get(u) for d in range(1,h+1)]
            positions=[p[0] if p is not None else None for p in ys]
            gaps=[distance(p[0],q[0]) for p,q in zip(xs,ys) if p and q]
            if y['label']==endpoint: why='eligible_target_on_time'
            elif y['label']==-1:why='other_class'
            elif endpoint in positions:why='visited_earlier'
            else:why='did_not_visit_by_horizon'
            paths.append(dict(h=h,outcome=y,selected_cell=endpoint,source_path=positions,
                    own_source_path=[None if p is None else p[0] for p in xs],
                    min_factual_carriers_distance=min(gaps) if gaps else None,reason=why,
                    alternative_first_move_reach_any_initial_factual_contact=any(p is not None and distance(i['new_destination'],p[0])<=d-1 for d,p in enumerate(ys,1))))
        i['future']=paths
    return dict(seed=spec['seed'],seat=seat,source_sha256=spec['raw_sha256'],validation_sha256=validation['artifact_sha256'],rows=rows)


def main():
    games=[]
    for group in (4,5):
        by={(r['seed'],r['seat']):r for r in load_games(group)}
        for g in batch(group):
            x=game(g,by[g['seed'],g['seat']]);games.append(x);print('checked',g['seed'],g['seat'],len(x['rows']),flush=True)
    allrows=[r for g in games for r in g['rows']];check(len(allrows)==41 and len({r['signature'] for r in allrows})==41,'issue population')
    hits=Counter();timing={str(h):Counter() for h in (3,5)};closest={str(h):Counter() for h in (3,5)}
    for r in allrows:
        hits[sum(w['reason']=='eligible_target_on_time' for w in r['future'])]+=1
        for w in r['future']:
            timing[str(w['h'])][w['reason']]+=1
            closest[str(w['h'])][str(w['min_factual_carriers_distance'])]+=1
    summary=dict(issued=len(allrows),same_step_same_destination_old=sum(r['baseline_colocation'] for r in allrows),
                 same_step_same_destination_alternative=sum(r['prospective_colocation'] for r in allrows),
                 changes_by_eligible_target_hits={str(k):hits[k] for k in (0,1,2)},factual_path_categories=timing,
                 closest_actual_v1_separation=closest,
                 alternative_reachable_target_site_by_factual_path={str(h):sum(r['future'][i]['alternative_first_move_reach_any_initial_factual_contact'] for r in allrows) for i,h in enumerate((3,5))},
                 delta_sums={k:round(sum(r['delta'][k] for r in allrows),6) for k in allrows[0]['delta']},
                 with_less_harvest=sum(r['delta']['real_new_harvest']< -1e-6 for r in allrows),
                 with_more_harvest=sum(r['delta']['real_new_harvest']>1e-6 for r in allrows))
    extract.write(HERE/'intervention_diagnostics.json',dict(summary=summary,games=games,
          interpretation='Both arms were checked with Board and FastCarbon for one step. Original occupied cell, target survival or deposit through h, and carrier-to-carrier minimum distance after t+1 are v1 observations only. Alternative reachability after its first step ignores all later hazards, banking, competition and opponent reaction. Income and cargo are separate units; their sums describe these 41 roots, including overlapping paths on four maps, not match-level income.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
