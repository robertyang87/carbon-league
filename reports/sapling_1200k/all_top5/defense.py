"""Replay-based two-seat accounting of collisions, defensive choices, and stolen cargo."""
from collections import Counter,defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DIR=((0,0),(0,-1),(1,0),(0,1),(-1,0))
DEST=[[((p%15+x)%15)+15*((p//15+y)%15) for x,y in DIR] for p in range(225)]
DR=lambda a,b:min(abs(a-b),15-abs(a-b))
DIST=[[DR(p%15,q%15)+DR(p//15,q//15) for q in range(225)] for p in range(225)]
RING=[{((p%15+x)%15)+15*((p//15+y)%15) for x in (-1,0,1) for y in (-1,0,1)} for p in range(225)]
MOVE={None:0,'NONE':0,'UP':1,'RIGHT':2,'DOWN':3,'LEFT':4}
RECRUIT={'RECCOLLECTOR':'COLLECTOR','RECPLANTER':'PLANTER'}


def audit(game):
    cfg=game['configuration'];frames=game['steps']
    if cfg['moveCost'] or cfg['smeltCost'] or cfg['size']!=15:
        raise ValueError('unsupported movement/smelting costs or board')
    sums=[Counter(),Counter()];lots=[];refund=[];threats=[];tags={};trees_robbed=[]
    for tick in range(len(frames)-1):
        ob=frames[tick][0]['observation'];aft=frames[tick+1][0]['observation']
        if ob['step']!=tick or aft['step']!=tick+1:
            raise ValueError('observation step gap')
        old=ob['players'];new=aft['players']
        act=[x['action'] or {} for x in frames[tick+1]]
        live={};groups=defaultdict(list);next_workers={};home={}
        old_trees={};grown={};taxable=[set(),set()]
        for s in (0,1):
            z=sums[s];z['ticks']+=1
            for wid,(p,c,k) in old[s][2].items():
                dest=DEST[p][MOVE[act[s].get(wid)]]
                live[wid]=(s,c,k);groups[dest].append(wid)
            for wid,(p,c,k) in new[s][2].items():next_workers[wid]=(s,p,c,k)
            for tid,(p,a) in old[s][3].items():
                old_trees[tid]=(s,p,a)
                if a<cfg['treeLifespan']:grown[p]=s
            for _,p in old[s][1].items():home[p]=s
        collector_n=sum(x[2]=='COLLECTOR' for x in live.values())
        prices={'COLLECTOR':cfg['recCollectorCost']+collector_n*cfg['recCollectorIncreaseCost'],
                'PLANTER':cfg['recPlanterCost']+(len(live)-collector_n)*cfg['recPlanterIncreaseCost']}
        for s in (0,1):
            z=sums[s];balance=old[s][0];n=len(old[s][2])
            for bid,p in old[s][1].items():
                kind=RECRUIT.get(act[s].get(bid))
                if kind is None or balance<prices[kind] or n>=cfg['workerLimit']:continue
                balance-=prices[kind];n+=1
                newborn=next((uid for uid,(seat,pos,_,typ) in next_workers.items() if uid not in live and seat==s and typ==kind and pos==p),f'new:{s}:{bid}:{tick}')
                live[newborn]=(s,0.,kind);groups[p].append(newborn);z['recruits']+=1
            foes=[(p,c) for p,c,k in old[1-s][2].values() if k=='COLLECTOR']
            for wid,(p,c,kind) in old[s][2].items():
                if kind!='COLLECTOR' or c<50:continue
                z['loaded_decisions']+=1
                near=[q for q,ec in foes if ec<=c and DIST[q][p]<=2]
                if not near:continue
                hostile={d for q in near for d in DEST[q]}
                if p not in hostile:continue
                z['loaded_stay_under_threat']+=1
                safe=[q for q in DEST[p] if q not in hostile and home.get(q)!=1-s]
                if not safe:continue
                z['loaded_stay_under_threat_with_safe_move']+=1
                end=DEST[p][MOVE[act[s].get(wid)]]
                z['chose_geometrically_safe_move']+=end in safe
                if end in safe:
                    ex=(float(c),tick,s,wid,p,end)
                    if len(threats)<2:threats.append(ex)
                    elif c>min(threats)[0]:threats[threats.index(min(threats))]=ex
            for tid,(p,age) in new[s][3].items():
                # Seized trees keep their IDs and obtain this step's tax for the new owner.
                # Only newly planted trees skip absorption in the rules.
                if tid in old_trees:taxable[s].add(p)
                if tid in old_trees and old_trees[tid][0]!=s:
                    z['enemy_trees_seized']+=1;sums[1-s]['trees_lost_to_enemy']+=1
                    trees_robbed.append([tick+1,s,tid,p,age])
        wanted={};routes={};ground={};fresh=[]
        for pos,group in groups.items():
            if len(group)==1:win=group[0]
            else:
                collectors=[uid for uid in group if live[uid][2]=='COLLECTOR']
                lo=min((live[u][1] for u in collectors),default=None)
                least=[u for u in collectors if live[u][1]==lo]
                win=least[0] if len(least)==1 else None
            mixed=len({live[u][0] for u in group})>1
            win_seat=live[win][0] if win is not None else None
            deaths=[u for u in group if u!=win]
            if mixed:
                for z in sums:z['mixed_collision_squares']+=1
            gross=sum(live[u][1] for u in group)
            if win is not None:
                route=('worker',win)
                if live[win][2]=='COLLECTOR' and pos in home:
                    route=('cash',home[pos])
                    if home[pos]!=win_seat:
                        if gross>0:
                            sums[win_seat]['enemy_base_donation_deposits']+=1
                            sums[win_seat]['all_enemy_base_donations']+=gross
                        else:
                            sums[win_seat]['deaths_empty_at_enemy_base']+=1
                    if home[pos]!=win_seat and gross==0:win_alive=False
                    else:win_alive=True
                else:win_alive=True
                if win_alive:wanted[win]=(win_seat,pos,live[win][2])
                routes[win]=route
            elif pos in grown:route=('cash',grown[pos])
            elif pos in home:route=('cash',home[pos])
            else:
                route=('ground',None)
                expired=any(p==pos and a>=cfg['treeLifespan'] for _,p,a in old_trees.values())
                ground[pos]=cfg['co2FrmWithered'] if expired else ob['full_carbon'][pos]
            credited=live[route[1]][0] if route[0]=='worker' else route[1]
            for victim in deaths:
                s,mass,kind=live[victim];z=sums[s];routes[victim]=route
                z['collision_deaths']+=1;z['mixed_deaths' if mixed else 'same_side_deaths']+=1
                z['collector_deaths' if kind=='COLLECTOR' else 'planter_deaths']+=1
                if mass<=0:continue
                if mixed:z['positive_victims_in_mixed_collision']+=1
                if route[0]=='ground':
                    dropped=min(mass,max(0.,cfg['maxCellCarbon']-ground[pos]));ground[pos]+=dropped
                    if mixed:
                        z['mixed_cargo_left_on_ground']+=dropped
                        z['mixed_cargo_destroyed_by_cell_cap']+=mass-dropped
                if win_seat==1-s and credited==s:
                    sums[1-s]['enemy_base_refund_at_contact']+=mass
                    refund.append([tick+1,1-s,victim,win,pos,mass])
                if credited!=1-s:continue
                acquired_by_collector=win_seat==1-s
                method='collector' if acquired_by_collector else ('enemy_base_donation_friendly_merge' if win is not None else ('tree' if pos in grown else 'base'))
                index=len(lots)
                lots.append(dict(t=tick+1,beneficiary=1-s,victim=victim,hunter=win,pos=pos,mass=mass,
                                 method=method,first_route=route[0],fate=Counter(),last={}))
                fresh.append((index,mass,route));sums[1-s]['acquisitions_all_methods']+=1
                sums[1-s]['enemy_cargo_acquired']+=mass;sums[1-s]['acquired_'+method]+=mass
        if set(wanted)!=set(next_workers) or any((side,pos,kind)!=(next_workers[u][0],next_workers[u][1],next_workers[u][3]) for u,(side,pos,kind) in wanted.items()):
            raise ValueError(f'next roster differs at {tick+1}')
        moved=defaultdict(dict)
        def resolve(ix,m,route):
            e=lots[ix];s=e['beneficiary'];k,who=route
            if k=='worker' and next_workers[who][0]==s:
                moved[who][ix]=moved[who].get(ix,0.)+m;return
            fate=('banked' if s==who else 'opponent_cash') if k=='cash' else ('opponent_worker' if k=='worker' else 'ground_or_destroyed')
            e['fate'][fate]+=m;e['last'][fate]=tick+1
        for uid,bundle in tags.items():
            for ix,m in bundle.items():resolve(ix,m,routes[uid])
        for ix,m,route in fresh:resolve(ix,m,route)
        for uid,bundle in moved.items():
            s,p,c,_=next_workers[uid]
            fraction=min(1.,len(RING[p]&taxable[1-s])*cfg['collectorAbsorptionRate'])
            for ix,m in bundle.items():
                if fraction:
                    paid=m*fraction;lots[ix]['fate']['tax_paid_to_enemy_trees']+=paid
                    lots[ix]['last']['tax_paid_to_enemy_trees']=tick+1;bundle[ix]-=paid
            if sum(bundle.values())>c+1e-6:raise ValueError(f'tags exceed actual cargo at {tick+1}')
        tags=moved
    for uid,bundle in tags.items():
        for ix,m in bundle.items():lots[ix]['fate']['carried_at_terminal']+=m
    for e in lots:
        if abs(sum(e['fate'].values())-e['mass'])>1e-6:raise ValueError('unbalanced lot')
        z=sums[e['beneficiary']]
        for k,v in e['fate'].items():
            z['stolen_'+k]+=v
            if e['method']=='collector':z['collector_stolen_'+k]+=v
    key=max((e for e in lots if e['method']=='collector'),key=lambda e:(e['fate'].get('banked',0),e['mass']),default=None)
    trace=[]
    if key:
        t=key['t'];s=key['beneficiary']
        for d in (-10,-5,-3,-1,0,1,3,5):
            i=t+d
            if not 0<=i<len(frames):continue
            ps=frames[i][0]['observation']['players'];h=ps[s][2].get(key['hunter']);v=ps[1-s][2].get(key['victim'])
            trace.append([d,h,v,DIST[h[0]][v[0]] if h and v else None])
    return dict(transitions=len(frames)-1,seat_totals=sums,lots=lots,enemy_base_refunds=refund,
                top_safe_escapes=sorted(threats,reverse=True),tree_seizures=trees_robbed,biggest_banked_trace=trace,
                trace_lot=(None if key is None else [key['t'],key['victim'],key['hunter']]))
