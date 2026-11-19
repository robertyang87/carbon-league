"""Aggregate terminal cash and reachable cargo, and quantify what the prescribed single-deviation family can fix."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import check as c
sys.path.insert(0,str(c.ACTOR.parent))
from turning_point_ledger import CashEvents


def ledger(sim,snapshot,commands,enemy,s):
    sim.set_state(snapshot);sim.cash=CashEvents(sim.cash);initial=list(sim.cash);orders=[None,None];orders[s]=commands;orders[1-s]=enemy;sim.step(*orders)
    net=Counter()
    for k,side,amount,*_ in sim.cash.events:net[k]+=(1 if side==s else -1)*amount
    assert abs(sum(net.values())-(sim.cash[s]-sim.cash[1-s]-initial[s]+initial[1-s]))<=1.e-5
    return dict(net)


def outcome(rows):
    slots=[x['roots'][-1] for x in rows];v=np.array([q['actual_one_step_cash_gain'] for q in slots]);o=np.array([q['oracle_max_one_step_gain'] for q in slots])
    return dict(games=len(rows),maps=len(set(x['seed'] for x in rows)),single_deviation_options_including_parent=int(sum(len(q['actions']) for q in slots)),
        changed=int(sum(q['chosen']!=0 for q in slots)),positive=int((v>1.e-5).sum()),negative=int((v< -1.e-5).sum()),gain_mean=float(v.mean()),gain_max=float(v.max()),
        hindsight_max=float(o.max()),hindsight_positive=int((o>1.e-5).sum()),hindsight_new_wins=sum(1 for x,n in zip(rows,o) if x['reported_margin']<0 and x['reported_margin']+n>0),
        actual_new_wins=sum(1 for x,n in zip(rows,v) if x['reported_margin']<0 and x['reported_margin']+n>0),
        by_turn=[dict(t=296+i,options=sum(len(x['roots'][i]['actions']) for x in rows),changed=sum(1 for x in rows if x['roots'][i]['chosen']),
            realized_immediate_cash_delta_average=float(np.mean([x['roots'][i]['actual_one_step_cash_gain'] for x in rows]))) for i in range(3)])


def main():
    rows=[c.read(c.HERE/'per_game'/f"{r['seed']}_{r['seat']}.json.gz") for r in c.P["games"]]
    selected=[r for r in rows if 'actions' in r['roots'][-1]];fresh=[r for r in selected if r['seed'] not in c.P['prior_last_turn_actors_only_comparison_maps']]
    official=c.read(c.HERE/'official_final.json.gz')
    assert len(official['rows'])==len(selected)
    money=[]; sources=[]
    for r in rows:
        g=c.read(c.ROOT/r['source']);sim=c.FastCarbon(g['game_configuration']).reset(seed=r['seed']);s=r['seat']
        for joint in g['actions'][:-1]:sim.step(*joint)
        z=r['roots'][-1];op=g['actions'][-1][1-s];shot=sim.get_state();base=ledger(sim,shot,g['actions'][-1][s],op,s)
        if 'chosen' in z and z['chosen']:
            a=z['actions'][z['chosen']];test=ledger(sim,shot,a['command'],op,s)
            diff={k:test.get(k,0)-base.get(k,0) for k in sorted(set(base)|set(test))}
            assert abs(sum(diff.values())-z['actual_one_step_cash_gain'])<1.e-5
            money.append(dict(seed=r['seed'],seat=s,delta=z['actual_one_step_cash_gain'],focal=a['focal'],net_cash_breakdown=diff))
        sources.append(dict(seed=r['seed'],seat=s,terminal_actual_margin=r['reported_margin'],final_unbanked_own=r['terminal_unbanked'][0],
            horizons=[dict(t=q['t'],turns_remaining=q['remaining'],own_cargo=q['cargo_total'][0],own_cargo_geometrically_reachable=q['cargo_reachable'][0]) for q in r['roots']],last_turn_realized_net_cash_sources=base))
    missed=[r for r in selected if r['reported_margin']<0 and r['reported_margin']+r['roots'][-1]['oracle_max_one_step_gain']>=0]
    summary=dict(protocol_sha256=c.sha(c.HERE/'protocol.json'),all_complete_games=len(rows),all_maps=len(set(r['seed'] for r in rows)),
        average_final_unbanked_own=float(np.mean([r['terminal_unbanked'][0] for r in rows])),
        cargo_average_by_horizon=[dict(t=296+i,total_own=float(np.mean([r['roots'][i]['cargo_total'][0] for r in rows])),
           geometrically_reachable=float(np.mean([r['roots'][i]['cargo_reachable'][0] for r in rows]))) for i in range(3)],
        all_eligible=outcome(selected),previous_terminal_actor_panel_excluded=outcome(fresh),public_changed_final_cash_evidence=money,
        terminal_options_officially_verified=sum(z['options'] for z in official['rows']),additional_native_response_gate=bool(missed),additional_native_response_reason="No eligible actual loss can be recovered even by a hindsight best-of-enumerated action on its own last step." if not missed else "Review all criteria before native branch.",
        rows=sources)
    c.write(c.HERE/'summary.json',summary)


if __name__=="__main__":main()
