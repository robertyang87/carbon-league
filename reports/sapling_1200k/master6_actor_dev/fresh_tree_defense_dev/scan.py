"""Enumerate every new-tree first loss and replay an auditable bounded defense from the same root."""
from collections import Counter,defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
sys.path.insert(0,str(ACTOR/'canopy_tempo_dev'))
from action_audit import sources
from study import ROOT,read
from policy import MOVES,DEST,DIST
sys.path.insert(0,str(ACTOR.parent))
from turning_point_ledger import CashEvents,FastCarbon

ECO=ACTOR/'tree_response_training/ten_map_economy/tenures.json.gz'
HORIZONS=(2,3,5)


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def nearest(pos,rows):
    return min((DIST[pos][w[0]] for w in rows),default=99)


def public(obs, tree, h):
    s=obs['player'];me,other=obs['players'][s],obs['players'][1-s]; p=tree['pos']
    defenders=sorted((DIST[p][pos],cargo,uid,pos) for uid,(pos,cargo,k) in me[2].items() if k=='COLLECTOR')
    threats=sorted((DIST[p][pos],uid,pos) for uid,(pos,c,k) in other[2].items() if k=='PLANTER')
    dist,uid,loc=(threats or [(99,None,None)])[0]
    return {'threat_distance':dist,'threat_uid':uid,'defender':list(defenders[0]) if defenders else None,
            'geometrically_reachable':bool(defenders and defenders[0][0]<=h),'public_threat':dist<h}


def defend(obs,parent_order,uid,target):
    s=obs['player'];me=obs['players'][s];enemy=obs['players'][1-s]
    w=me[2].get(uid)
    if w is None:
        return None,'defender_absent'
    pos,cargo,_=w
    remaining=DIST[pos][target]
    actual=parent_order.get(uid,'NONE')
    other_dest={DEST[location][MOVES.index(parent_order.get(wid,'NONE'))] for wid,(location,_,_) in me[2].items() if wid!=uid}
    other_dest.update(location for bid,location in me[1].items() if parent_order.get(bid) in ('RECCOLLECTOR','RECPLANTER'))
    threatening=[v for v in enemy[2].values() if v[2]=='COLLECTOR' and v[1]<=cargo]
    for a in dict.fromkeys([actual,*MOVES]):
        q=DEST[pos][MOVES.index(a)]
        if DIST[target][q] != max(0,remaining-1):continue
        if q in other_dest or nearest(q,threatening)<=1:continue
        return a,'safe_shortest'
    return actual,'all_shortest_screened'


def simulate(game, tree, horizon, before, uid, changed):
    s=game['seat'];target=tree['pos'];tid=tree['tid'];lost_step=tree['end']
    t0=lost_step-horizon; limit=min(len(game['tape']),lost_step+5)
    sim=FastCarbon(game['cfg']);sim.set_state(before)
    sim.cash=CashEvents(sim.cash)
    channels=[Counter(),Counter()]; moves=[]; missing=Counter()
    focal_deposits=0.;own_tree_turns=0;window=None
    for t in range(t0,limit):
        obs=sim.obs(s);joint=[dict(cmd) for cmd in game['tape'][t]]
        if changed and t<lost_step and tid in sim.trees and sim.trees[tid].player==s:
            cmd,reason=defend(obs,joint[s],uid,target)
            if cmd is not None:
                old=joint[s].pop(uid,'NONE')
                if cmd!='NONE':joint[s][uid]=cmd
                moves.append([t,old,cmd,reason])
        for side,orders in enumerate(joint):
            living=sim.workers.keys() | set(obs['players'][side][1])
            missing[side]+=len(set(orders)-living)
        sim.cash.events.clear();sim.step(*joint)
        for kind,owner,value,*extra in sim.cash.events:
            channels[owner][kind]+=value
            if kind=='own_delivery' and owner==s and extra[0]==uid:focal_deposits+=value
        held=tid in sim.trees and sim.trees[tid].player==s
        own_tree_turns+=held
        if t+1==lost_step:
            window=dict(held=held,cash=list(sim.cash),focal_alive=uid in sim.workers,
                        focal_cargo=(sim.workers[uid].carbon if uid in sim.workers else None))
    return dict(at_original_seizure=window,post=limit,held_at_end=tid in sim.trees and sim.trees[tid].player==s,
                cash=list(sim.cash),cash_channels=[dict(c) for c in channels],own_tree_turns=own_tree_turns,
                focal_deposits=focal_deposits,focal_alive=uid in sim.workers,
                focal_cargo=(sim.workers[uid].carbon if uid in sim.workers else 0.),
                onboard_cargo=[sum(w.carbon for w in sim.workers.values() if w.player==owner and w.is_collector) for owner in (0,1)],
                moves=moves,missing=[missing[0],missing[1]])


def game_scan(source, tenure):
    s=source['seat'];p=ROOT/source['raw_path'];tape=read(p)
    if sha(p)!=source['raw_sha256'] or sha(p)!=tenure['raw_sha256']:
        raise ValueError('source identity')
    holdings=[r for r in tenure['holdings'] if r['owner']==s and r['origin']=='plant' and 51<=r['start']<=250]
    takeovers={(r['tid'],r['start']):r for r in tenure['holdings'] if r['owner']==1-s and r['origin']=='seize'}
    rivals=[r for r in tenure['holdings'] if r['owner']==1-s and r['origin']=='plant' and 51<=r['start']<=250]
    game={'seat':s,'tape':tape['actions'],'cfg':tape['game_configuration']}
    sim=FastCarbon(game['cfg']).reset(seed=source['seed'])
    if sim.carbon!=tape['initial_full_carbon']:raise ValueError('incorrect opening')
    time_queries=defaultdict(list); missed=[]; results=[]
    for r in holdings:
        if r['exit']!='seized_by_opponent':continue
        for h in HORIZONS:
            if r['end']-h < r['start']:
                missed.append({'tid':r['tid'],'end':r['end'],'start':r['start'],'horizon':h,'tree_not_yet_owned':True})
            else:time_queries[r['end']-h].append((r,h))
    alarms={str(h):Counter() for h in HORIZONS};history=[]
    for t, joint in enumerate(game['tape']):
        obs=sim.obs(s)
        history.append({u:v[0] for u,v in obs['players'][1-s][2].items() if v[2]=='PLANTER'})
        for r in holdings:
            if not r['start']<=t<r['end']:continue
            # Only complete future windows; a non-seizure exit before H is a real negative.
            for h in HORIZONS:
                if t+h>len(game['tape']):continue
                state=public(obs,r,h)
                loss=r['exit']=='seized_by_opponent' and r['end']<=t+h
                key='alert' if state['public_threat'] else 'no_alert'
                alarms[str(h)][key]+=1;alarms[str(h)][key+'_loss_by_horizon']+=loss
                if state['public_threat'] and state['geometrically_reachable']:
                    alarms[str(h)]['alert_and_reachable']+=1;alarms[str(h)]['alert_and_reachable_loss']+=loss
        for r,h in time_queries.get(t,[]):
            state=public(obs,r,h)
            row={'seed':source['seed'],'seat':s,'tid':r['tid'],'origin_step':r['start'],'original_loss_post':r['end'],
                 'root':t,'horizon':h,'tree_pos':r['pos'],'actual_purchase_cost':r['price'],
                 'three_prior_threat_positions':[entry.get(state['threat_uid']) for entry in history[-4:]],**state}
            take=takeovers[r['tid'],r['end']]
            row.update(actual_next_enemy_tenure_turns=take['preturns'], actual_next_enemy_tenure_gross=take['gross']+take['collision'],
                       actual_next_enemy_tenure_net=take['gross']+take['collision']-take['price'])
            if state['geometrically_reachable']:
                uid=state['defender'][2]; snap=sim.get_state()
                # get_state returns independent lists of primitives. Both arms start with exact original worker/tree counters.
                factual=simulate(game,r,h,snap,uid,False); alt=simulate(game,r,h,snap,uid,True)
                if factual['at_original_seizure']['held'] or factual['missing']!=[0,0]:
                    raise ValueError('factual branch did not reproduce seizure')
                row.update(parent=factual,defense=alt,
                           margin_delta=round(alt['cash'][s]-alt['cash'][1-s]-factual['cash'][s]+factual['cash'][1-s],5),
                           delivery_delta=round(alt['focal_deposits']-factual['focal_deposits'],5),
                           cargo_delta=round(alt['focal_cargo']-factual['focal_cargo'],5))
            results.append(row)
        sim.step(*joint)
    if list(sim.cash)!=source['scores']:
        raise ValueError('recorded official cash mismatch')
    return {'seed':source['seed'],'seat':s,'raw_sha256':source['raw_sha256'],'own_mid_fresh':len(holdings),
            'own_seized':sum(r['exit']=='seized_by_opponent' for r in holdings),'opponent_mid_fresh':len(rivals),
            'opponent_seized':sum(r['exit']=='seized_by_opponent' for r in rivals),
            'threat_opportunities':alarms,'not_yet_owned':missed,'roots':results}


def main():
    tenures={(g['seed'],g['seat']):g for g in read(ECO)}
    games=[game_scan(s,tenures[s['seed'],s['seat']]) for group,s in sources() if group=='training_source']
    if len(games)!=20:raise ValueError('incomplete map-seat cohort')
    agg={}
    for h in HORIZONS:
        rows=[r for g in games for r in g['roots'] if r['horizon']==h]
        compared=[r for r in rows if 'defense' in r]
        blocked=[r for r in compared if r['defense']['at_original_seizure']['held']]
        alert=Counter();[alert.update(g['threat_opportunities'][str(h)]) for g in games]
        agg[h]=dict(total_seizures=sum(g['own_seized'] for g in games),already_owned=len(rows),geom_reachable=len(compared),
                    protected_at_loss=len(blocked),protected_still_at_plus5=sum(r['defense']['held_at_end'] for r in blocked),
                    public_threat_and_reachable=sum(r['public_threat'] for r in compared),
                    forced_actions=sum(old!=new for r in compared for _,old,new,_ in r['defense']['moves']),
                    aggregate_open_loop_margin_delta=sum(r['margin_delta'] for r in compared),
                    timely_defenses_with_worse_margin=sum(r['margin_delta']<0 for r in blocked),
                    timely_defenses_with_own_lost_delivery=sum(r['delivery_delta']<-.01 for r in blocked),
                    warning_census=alert)
    data={'protocol_sha256':sha(HERE/'protocol.json'),'code_sha256':sha(Path(__file__)),'full_cohort_sha256':sha(ECO),
          'games':games,'summary':agg}
    blob=json.dumps(data,separators=(',',':'),allow_nan=False).encode()
    (HERE/'responses.json.gz').write_bytes(gzip.compress(blob,mtime=0))
    print(agg,flush=True)


if __name__=='__main__':main()
