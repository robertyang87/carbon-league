"""Bounded hindsight interception on already played double-delivery paths; no fresh policies."""
from collections import Counter
from pathlib import Path

from terminal_intercept import HERE, DIST, DEST, MOVES, extract, fetch, load, dt
import json


def paths(p,goal,room,lead=()):
    if room==0:
        if p==goal:yield lead
        return
    for move,q in zip(MOVES,DEST[p]):
        if DIST[q][goal]<room:yield from paths(q,goal,room-1,lead+(move,))


def evaluate(env,bank,pay,target,uid,path,snap,strict=False):
    fs,cfg=env['steps'],env['configuration'];end=len(fs)-1; start=pay-5;defense=1-bank
    sim=extract.FastCarbon(cfg);sim.set_state(snap)
    home={p for _,p in sim.bases[defense]};enemy_bases={p for _,p in sim.bases[bank]}
    banked=enemy_deposit=0.;hits=False;win_at=None;held=0.;unknown=changed=overrides=0;turns=[]
    for t in range(start,end):
        joint=[a['action'] or {} for a in fs[t+1]];focal=sim.workers.get(uid);orig=joint[defense].get(uid) or 'NONE'
        if not hits and not focal:return None
        legal=[set(sim.player_workers[s])|{b for b,_ in sim.bases[s]} for s in (0,1)]
        unknown+=sum(len(set(joint[s])-legal[s]) for s in (0,1))
        recorded=fs[t][0]['observation']['players']
        changed+=int(any(set(recorded[s][2])!=set(sim.player_workers[s]) for s in (0,1)))
        commands=[dict(a) for a in joint]
        order=orig; mode='factual'
        if t-start<len(path) and not hits:order=path[t-start];mode='oracle_stage'
        elif not hits:return None
        elif focal and banked<=0:
            # Nearest own base, one plausible open move; deterministic by move order.
            mode='public_return'
            reserved={DEST[sim.workers[v].pos][MOVES.index(joint[defense].get(v) or 'NONE')]
                      for v in sim.player_workers[defense] if v!=uid}
            reserved|={p for b,p in sim.bases[defense] if joint[defense].get(b) in ('RECPLANTER','RECCOLLECTOR')}
            options=[(dt(q,home),i,m) for i,(m,q) in enumerate(zip(MOVES,DEST[focal.pos])) if q not in reserved|enemy_bases]
            if options:order=min(options)[2]
        if mode=='oracle_stage':
            q=DEST[focal.pos][MOVES.index(order)]
            reserved={DEST[sim.workers[u].pos][MOVES.index(joint[defense].get(u) or 'NONE')] for u in sim.player_workers[defense] if u!=uid}
            reserved|={p for b,p in sim.bases[defense] if joint[defense].get(b) in ('RECPLANTER','RECCOLLECTOR')}
            if q in reserved|enemy_bases:return None
        if order!=orig:
            overrides+=1
            if order=='NONE':commands[defense].pop(uid,None)
            else:commands[defense][uid]=order
        if strict:
            after,ev=extract.step(sim,sim.full_observation(),cfg,commands)
        else:
            sim.cash=extract.CashEvents(sim.cash);sim.step(*commands);ev=sim.cash.events
        lost=next((x for x in sim.last_collisions if x[0]==target and x[4]==defense),None)
        focal=sim.workers.get(uid)
        if not hits and lost and focal:
            actual=fs[t+1][0]['observation']['players'][bank][2].get(target)
            # Require that this focal collector picks up the target on the actual route.
            if actual and focal.pos==actual[0] and focal.pos not in enemy_bases and lost[2]>0:
                hits=True;win_at=t+1;held=lost[2]
        deposits=[(e[1],e[3],e[2]) for e in ev if e[0]=='own_delivery']
        if hits:banked+=sum(v for s,u,v in deposits if s==defense and u==uid)
        enemy_deposit+=sum(v for s,u,v in deposits if s==bank and u==target)
        turns.append([t,mode,orig,order,None if focal is None else focal.pos,deposits])
        if any(not sim.player_workers[s] and sim.cash[s]<cfg['recCollectorCost'] for s in (0,1)):break
    if not hits:return None
    scores=fs[-1][0]['observation']['players'];native=[s[0] for s in scores]
    if sim.step_count!=end:return None
    dm=round((sim.cash[defense]-sim.cash[bank])-(native[defense]-native[bank]),4)
    return dict(target=target,runner=uid,path=path,hits_at=win_at,target_cargo_transferred=round(held,5),
                runner_own_bank_since_contact=round(banked,5),observed_target_paid_in_branch=round(enemy_deposit,5),
                terminal_margin_delta=dm,terminal_cash_by_seat=list(sim.cash),counterfactual_order_count=overrides,
                explicit_absent_order_ids=unknown,turns_with_different_rosters=changed,trace=turns)


def main():
    one=load(HERE/'terminal_intercept.json'); manifest={r['id']:r for r in load(HERE/'manifest.json')['games']}
    records=[]
    for g in one['games']:
        if not g['events']:continue
        env=load(fetch.CACHE/manifest[g['id']]['path']);fs=env['steps'];cfg=env['configuration']
        native=extract.FastCarbon(cfg).load_observation(fs[0][0]['observation']);states={}
        requests={e['t']-5 for e in g['events']}
        for t,frame in enumerate(fs[1:]):
            if t in requests:states[t]=native.get_state()
            native.step(*[a['action'] or {} for a in frame])
        for e in g['events']:
            pay=e['t'];start=pay-5;bank=e['payer_seat'];de=1-bank
            pl=fs[start][0]['observation']['players'];homes=set(pl[bank][1].values())
            for item in e['targets']:
                target=item['uid']; w=pl[bank][2].get(target); attempts=set()
                if w:
                    for uid,(p,c,k) in pl[de][2].items():
                        if k!='COLLECTOR' or c>=w[1]:continue
                        for j in range(1,5):
                            wj=fs[start+j][0]['observation']['players'][bank][2].get(target)
                            if not wj or wj[0] in homes or DIST[p][wj[0]]>j:continue
                            attempts.update((uid,path) for path in paths(p,wj[0],j))
                successes=[]
                for uid,path in sorted(attempts):
                    r=evaluate(env,bank,pay,target,uid,path,states[start])
                    if r is not None:successes.append(r)
                # Keep the best hindsight outcome and raw counts for all paths, not a prospective selector.
                best=max(successes,key=lambda r:r['terminal_margin_delta']) if successes else None
                if best:
                    audit=evaluate(env,bank,pay,target,best['runner'],best['path'],states[start],True)
                    if audit!=best:raise ValueError('selected alternate diverged between engines')
                records.append(dict(id=g['id'],depositor_seat=bank,pay_at=pay,target=target,actual_credit=item['actual_credit'],
                        originally_eligible_paths=len(attempts),successful_paths=len(successes),
                        positive_paths=sum(r['terminal_margin_delta']>.01 for r in successes),
                        paths_with_own_bank=sum(r['runner_own_bank_since_contact']>.01 for r in successes),
                        min_cash_margin=min((r['terminal_margin_delta'] for r in successes),default=None),
                        max_cash_margin=best['terminal_margin_delta'] if best else None,
                        best_oracle_path=best))
    summary=dict(deliveries=len(records),ones_with_geometric_path=sum(bool(r['originally_eligible_paths']) for r in records),
                 with_true_target_grab=sum(bool(r['successful_paths']) for r in records),
                 with_eventual_runner_bank=sum(bool(r['paths_with_own_bank']) for r in records),
                 with_any_positive_open_loop_cash_margin=sum(bool(r['positive_paths']) for r in records))
    doc=dict(summary=summary,records=records,limits='Oracle uses the same worker actual future cells from an existing match to stage a single initially lighter collector starting five turns before an actual double deposit. Other workers replay original orders. Before capture reserve friendly moves and commanded hires. After capture head to nearest free own base until first positive receipt; further own-bank totals can mix original, stolen and new cargo. Check each highest-margin successful branch again with Board and FastCarbon through horizon. Altered living rosters and invalid later recorded orders counted; effects are not a responsive opponent, guaranteed surplus or a learned policy.')
    (HERE/'staging.json').write_text(json.dumps(doc,ensure_ascii=False,indent=1)+'\n')
    print(summary)


if __name__=='__main__':main()
