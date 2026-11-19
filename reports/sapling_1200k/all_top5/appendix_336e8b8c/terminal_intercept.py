"""Replay-side audit of simultaneous large late deliveries. No policies or new games."""
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP.parent/'master6_actor_dev/tree_response_training/planter_survival'))
import extract
sys.path.insert(0,str(TOP))
import fetch
from appendix_cbd6d545.contrasts import load
from economy_extract import distances

DIST=distances(15)
MOVES=('NONE','UP','RIGHT','DOWN','LEFT')
DEST=[[15*((p//15+dy)%15)+(p%15+dx)%15 for dx,dy in ((0,0),(0,-1),(1,0),(0,1),(-1,0))] for p in range(225)]


def dt(o,home):
    return min(DIST[o][b] for b in home)


def digest(row):
    return {k:row[k] for k in ('id','score_by_seat','seat_names','sha256')}


def run(g,tape):
    replay=fetch.CACHE/g['path'];assert fetch.sha(replay.read_bytes())==g['sha256']==tape['replay_sha256']
    env=load(replay);fs=env['steps'];cfg=env['configuration'];L=len(fs)-1
    want=[]
    for r in tape['turns']:
        for s in (0,1):
            group=[(u,v) for k,u,v in r[3] if k==s and v>=100]
            if L-19<=r[0]<=L and len(group)>=2:want.append((r[0],s,group))
    if not want:return {**digest(g),'events':[]}
    sim=extract.FastCarbon(cfg).load_observation(fs[0][0]['observation']);snap={}; cmds=lambda t:[a['action'] or {} for a in fs[t+1]]
    entries=[]
    for t in range(L):
        if any(1<=p-t<=5 for p,_,_ in want):snap[t]=sim.get_state()
        sim.step(*cmds(t))
        if sim._players_obs()!=fs[t+1][0]['observation']['players']:raise ValueError('replay reconstruction')
    sim2=extract.FastCarbon(cfg)
    for P,bank,group in want:
        defense=1-bank;home=set(fs[P][0]['observation']['players'][bank][1].values())
        probes=[]
        for target,amount in group:
            lead=[]
            for lag in (5,3,2,1):
                t=P-lag; rec=fs[t][0]['observation']['players']; movers=rec[defense][2]; rival=rec[bank][2]
                carrier=rival.get(target)
                if not carrier:
                    lead.append(dict(lag=lag,absent=True,options=[],can_stage=False));continue
                H=cmds(t);un=H[bank].get(target) or 'NONE';dst=DEST[carrier[0]][MOVES.index(un)]
                ww=fs[t+1][0]['observation']['players'][bank][2].get(target)
                if not ww or ww[0]!=dst:raise ValueError('bank target disappeared unexpectedly')
                root_price=carrier[1]*(1-cfg['moveCost']) if un!='NONE' else carrier[1]
                last=[fs[z][0]['observation']['players'][bank][2].get(target) for z in range(max(0,t-3),t+1)]
                homes={p for p in rec[defense][1].values()}
                # Earliest geometric interception before credit, even with perfect knowledge of
                # the recipient's subsequent factual movement. Optimistic: future hazards ignored.
                staged=[]; candidates=[]
                for u,(start,carry,role) in movers.items():
                    if role!='COLLECTOR' or carry>=carrier[1]:continue
                    base_order=H[defense].get(u) or 'NONE'; base_cell=DEST[start][MOVES.index(base_order)]
                    for j in range(1,lag):
                        path=fs[t+j][0]['observation']['players'][bank][2].get(target)
                        if path and path[0] not in home and DIST[start][path[0]]<=j:
                            staged.append([u,j,path[0],carry]);break
                    if dst not in DEST[start]:continue
                    move=MOVES[DEST[start].index(dst)]
                    if move==base_order:continue
                    post=carry*(1-cfg['moveCost']) if move!='NONE' else carry
                    if post>=root_price:continue
                    if any(uid!=u and DEST[w[0]][MOVES.index(H[defense].get(uid) or 'NONE')]==dst for uid,w in movers.items()):continue
                    # Reserve any recruitment command on this cell conservatively.
                    if any(pos==dst and H[defense].get(b) in ('RECCOLLECTOR','RECPLANTER') for b,pos in rec[defense][1].items()):continue
                    alt=[dict(j) for j in H]
                    if move=='NONE':alt[defense].pop(u,None)
                    else:alt[defense][u]=move
                    sim2.set_state(snap[t]);state=sim2.full_observation();after,events=extract.step(sim2,state,cfg,alt)
                    enemy_paid=sum(e[2] for e in events if e[0] in ('own_delivery','opponent_worker_delivery_to_this_base') and e[1]==bank)
                    base_paid=sum(tape['turns'][t][2][bank][:2])
                    picked=next((v for v in sim2.last_collisions if v[0]==target),None)
                    root_gain=(after['players'][defense][0]-after['players'][bank][0])-(tape['turns'][t][1][defense]-tape['turns'][t][1][bank])
                    lost=any(s==defense and worker==u and val>0 for s,worker,val in tape['turns'][t][3])
                    # Continue only as a mechanical sensitivity check; count missing explicit
                    # IDs and all living-roster departures, including implicit NONE choices.
                    missing=0;roster_changed=0
                    ownbank=next((e[2] for e in events if e[0]=='own_delivery' and e[1]==defense and e[3]==u),0.)
                    branch_alive=sim2.workers.get(u);first_held=branch_alive.carbon if branch_alive else None
                    if picked and picked[4]==defense:
                        for k in range(t+1,L):
                            j=cmds(k);actual=fs[k][0]['observation']['players']
                            legal=[set(sim2.player_workers[i])|{bid for bid,_ in sim2.bases[i]} for i in (0,1)]
                            missing+=sum(len(set(j[i])-legal[i]) for i in (0,1))
                            roster_changed+=int(any(set(actual[i][2])!=set(sim2.player_workers[i]) for i in (0,1)))
                            # The sensitivity trajectory ends at an altered extinction.
                            if any(not sim2.player_workers[z] and sim2.cash[z]<cfg['recCollectorCost'] for z in (0,1)):break
                            sim2.cash=extract.CashEvents(sim2.cash);sim2.step(*j)
                            ownbank+=sum(ev[2] for ev in sim2.cash.events if ev[0]=='own_delivery' and ev[1]==defense and ev[3]==u)
                    end_diff=(sim2.cash[defense]-sim2.cash[bank])-(g['score_by_seat'][defense]-g['score_by_seat'][bank]) if sim2.step_count==L else None
                    candidates.append(dict(own_uid=u,from_cell=start,focal_cargo=carry,old=base_order,alternative=move,cell=dst,
                        transfer=0 if not picked or picked[4]!=defense else round(picked[2],5),
                        at_receiver_base=dst in home, immediate_money_margin_delta=round(root_gain,4),
                        opponent_all_bank_delta=round(enemy_paid-base_paid,5),overrode_own_paying_order=lost,
                        winning_collector_holding_after_first=first_held,open_loop_focal_future_own_bank=round(ownbank,5),
                        open_loop_terminal_cash_margin_delta=None if end_diff is None else round(end_diff,4),
                        missing_recorded_order_ids=missing,active_roster_different_turns=roster_changed))
                lead.append(dict(lag=lag,t=t,target_cell=carrier[0],target_cargo=carrier[1],distance_to_home=dt(carrier[0],home),
                        past_visible_home_distances=[None if w is None else dt(w[0],home) for w in last],
                        actual_next_order=un,actual_next_cell=dst,optimistic_staging_paths=staged,can_stage=bool(staged),options=candidates))
            probes.append(dict(uid=target,actual_credit=amount,contacts=lead))
        entries.append(dict(t=P,payer_seat=bank,targets=probes))
    return {**digest(g),'events':entries}


def main():
    data=load(HERE/'turn_cash.json.gz');idx={x['id']:x for x in data['games']}
    games=[run(g,idx[g['id']]) for g in load(HERE/'manifest.json')['games']]
    events=[e for g in games for e in g['events']];units=[a for e in events for a in e['targets']]
    summary={}
    for lag in (5,3,2,1):
        phases=[next(r for r in a['contacts'] if r['lag']==lag) for a in units]
        opts=[o for p in phases for o in p['options']]
        summary[lag]=dict(deliveries=len(phases),stage_before_own_base=sum(p['can_stage'] for p in phases),
                at_least_one_true_contact_option=sum(bool(p['options']) for p in phases),
                with_true_transfer=sum(any(o['transfer']>0 for o in p['options']) for p in phases),
                with_positive_immediate_cash_margin=sum(any(o['immediate_money_margin_delta']>.011 for o in p['options']) for p in phases),
                with_open_loop_positive=sum(any((o['open_loop_terminal_cash_margin_delta'] or 0)>.011 for o in p['options']) for p in phases),
                tested_single_order_changes=len(opts))
    result=dict(scope='All ten games, both receiving seats, final 20 signed transitions. At least two distinct collectors must each make a normal own-base deposit >=100 in one transition. Working backwards from actual large receipts intentionally uses their future path; this is an optimistic feasibility check, not a rule, online accuracy or causal policy value. Branches test all currently strictly lighter collectors that can enter the observed next target cell with one changed action; avoid friendly occupied/reserved cells. The observed alternative focal action and opponent joint order are applied by both Board and FastCarbon. In the optional open-loop continuation source orders are reused; a rival worker removed by capture cannot respond, and its old commands are ignored. No future policy was evaluated. A horizon at which no single-step collision occurs does not exclude nontrivial staging or coordinated two-worker play.',
        games=games, simultaneous_receipt_events=len(events),exposed_individual_deliveries=len(units),map_count=len(games),
        games_with_events=sum(bool(g['events']) for g in games),summary=summary)
    (HERE/'terminal_intercept.json').write_text(json.dumps(result,indent=1,ensure_ascii=False,allow_nan=False)+'\n');print(summary)


if __name__=='__main__':main()
