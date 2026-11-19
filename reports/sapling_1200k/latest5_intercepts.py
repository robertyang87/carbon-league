"""Trace collision cargo and its ultimate cash disposition on verified ladder replays."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'code'))
sys.path.insert(0,str(HERE.parent/'master_family/deep_review'))
from cnet2_feat import DEST,DIST,MOVES
from fastsim import FastCarbon
from recover_public import canonical

DEST=DEST.tolist()
DIST=DIST.tolist()
MOVE={m:i for i,m in enumerate(MOVES)}
ROUND=lambda x:round(x,5)


class Economics:
    def __init__(self):
        source,start=inspect.getsourcelines(FastCarbon.step)
        self.code=FastCarbon.step.__code__
        self.hooks={}
        for name,needle in [('hit','if winner_is_collector:'),('deposit','self.cash[p] += w.carbon - smelt'),
                            ('tax','w.carbon = max(w.carbon - red, 0)')]:
            lines=[start+k for k,s in enumerate(source) if s.strip()==needle]
            if len(lines)!=1:
                raise ValueError((name,lines))
            self.hooks[lines[0]]=name

    def trace(self,f,event,arg):
        if f.f_code is not self.code:
            return None
        if event!='line' or f.f_lineno not in self.hooks:
            return self.trace
        x=f.f_locals
        worker=x['w']
        hook=self.hooks[f.f_lineno]
        if hook=='hit':
            winner=x['winner']
            tile=x['pos']
            recipients={v.player for v in x['group']}
            base=x['self'].base_at.get(tile)
            if winner is not None and winner.is_collector:
                destination=('cash',base[0]) if base else ('worker',winner.id)
                method='collector'
            elif tile in x['tree_at']:
                destination=('cash',x['tree_at'][tile].player)
                method='tree'
            elif base:
                destination=('cash',base[0]);method='base'
            else:
                destination=('ground',None);method='ground'
            self.hits[worker.id]=dict(owner=worker.player,amount=worker.carbon,pos=tile,method=method,route=destination,
                                      winner=None if winner is None else winner.id,winning_seat=None if winner is None else winner.player,
                                      mixed=len(recipients)==2,base=base[0] if base else None,
                                      dropped=min(worker.carbon,max(0,x['cfg']['maxCellCarbon']-x['carbon'][tile])) if method=='ground' else 0.)
        elif hook=='deposit' and worker.carbon>0:
            self.deposits[worker.id]=(worker.player,x['p'],worker.carbon)
        elif hook=='tax' and worker.carbon>0:
            self.levy[worker.id]=min(1.,x['red']/worker.carbon)
        return self.trace

    def step(self,sim,commands):
        self.hits={};self.deposits={};self.levy={}
        previous=sys.gettrace()
        try:
            sys.settrace(self.trace)
            sim.step(*commands)
        finally:
            sys.settrace(previous)


def audit(game,meta):
    frames=game['steps']
    cfg=game['configuration']
    if cfg['smeltCost']!=0 or cfg['moveCost']!=0 or cfg['size']!=15:
        raise ValueError('unrecognized charge/geometry')
    sim=FastCarbon(cfg).load_observation(frames[0][0]['observation'])
    econ=Economics();summary=[Counter(),Counter()];events=[];refunds=[];active=defaultdict(dict)
    # Each tag is owned by the first receiving seat until it banks, loses or ends holding that lot.
    for tick,frame in enumerate(frames[1:]):
        actions=[a['action'] or {} for a in frame]
        src=frames[tick][0]['observation']
        prev=src['players']
        for s in (0,1):
            z=summary[s]
            for hid,(hp,hc,hkind) in prev[s][2].items():
                if hkind!='COLLECTOR':
                    continue
                for uid,(p,c,k) in prev[1-s][2].items():
                    if k!='COLLECTOR' or c<50 or hc>=c or DIST[hp][p]>2:
                        continue
                    z['lighter_vs_loaded_reachable_pairs']+=1
                    actual=DEST[hp][MOVE[actions[s].get(hid) or 'NONE']]
                    z['chosen_landing_in_enemy_reach']+=actual in DEST[p]
                    z['chosen_landing_met_enemy']+=actual==DEST[p][MOVE[actions[1-s].get(uid) or 'NONE']]
            for uid,(pos,cargo,kind) in prev[s][2].items():
                if kind!='COLLECTOR' or cargo<50:
                    continue
                z['loaded_decisions']+=1
                foes=[q for q,c,k in prev[1-s][2].values() if k=='COLLECTOR' and c<=cargo]
                threatened={d for q in foes for d in DEST[q]}
                if pos not in threatened:
                    continue
                z['loaded_stay_under_threat']+=1
                homes=set(prev[1-s][1].values())
                safe=[p for p in DEST[pos] if p not in threatened|homes]
                if safe:
                    z['loaded_stay_under_threat_with_safe_move']+=1
                    dest=DEST[pos][MOVE[actions[s].get(uid) or 'NONE']]
                    z['chose_geometrically_safe_move']+=dest in safe
        econ.step(sim,actions)
        truth=frame[0]['observation'];seen=sim.full_observation()
        if (seen['players']!=truth['players'] or seen['trees']!=truth['trees'] or seen['step']!=truth['step']
                or seen['carbon']!=truth['full_carbon']):
            raise ValueError(f"engine mismatch {meta['id']}:{tick+1}")
        for seat in (0,1):
            summary[seat]['ticks']+=1
            mixed_sites={h['pos'] for h in econ.hits.values() if h['mixed']}
            summary[seat]['mixed_collision_squares']+=len(mixed_sites)
        kept=defaultdict(dict)
        def settle(index,mass,route):
            known=events[index];owner=known['beneficiary']
            kind,who=route
            if kind=='worker' and sim.workers[who].player==owner:
                kept[who][index]=kept[who].get(index,0)+mass
            else:
                fate=('banked' if who==owner else 'opponent_cash') if kind=='cash' else ('opponent_worker' if kind=='worker' else 'ground_or_destroyed')
                known['fate'][fate]+=mass
                known['fate_last_transition'][fate]=tick+1

        for uid,tags in active.items():
            if uid in econ.hits:
                route=econ.hits[uid]['route']
            elif uid in econ.deposits:
                route=('cash',econ.deposits[uid][1])
            elif uid in sim.workers:
                route=('worker',uid)
            else:
                raise ValueError(f'vanished tagged holder {uid}')
            for index,mass in tags.items():
                settle(index,mass,route)
        for uid,(worker_owner,recipient,amount) in econ.deposits.items():
            if worker_owner!=recipient:
                summary[worker_owner]['all_enemy_base_donations']+=amount
                summary[worker_owner]['enemy_base_donation_deposits']+=1
        for victim,h in econ.hits.items():
            if h['amount']<=0:
                continue
            victim_seat=h['owner'];benefit=1-victim_seat;kind,who=h['route']
            if h['mixed']:
                summary[victim_seat]['positive_victims_in_mixed_collision']+=1
            attacker=h['winning_seat']==benefit and h['method']=='collector'
            recipient=(sim.workers[who].player if kind=='worker' else who if kind=='cash' else None)
            if not attacker and recipient!=benefit:
                if h['mixed'] and kind=='ground':
                    summary[victim_seat]['mixed_cargo_left_on_ground']+=h['dropped']
                    summary[victim_seat]['mixed_cargo_destroyed_by_cell_cap']+=h['amount']-h['dropped']
                if h['mixed'] and recipient==victim_seat:
                    summary[victim_seat]['own_value_recovered_at_mixed_collision']+=h['amount']
                continue
            if attacker:
                summary[benefit]['collector_take_attempt_resolved']+=1
                if recipient==victim_seat:
                    summary[benefit]['enemy_base_refund_at_contact']+=h['amount']
                    refunds.append(dict(t=tick+1,beneficiary=benefit,victim=victim,winner=h['winner'],pos=h['pos'],
                                        type='enemy_base_refund',mass=h['amount'],actual_received=0.,fate={'opponent_cash':h['amount']},
                                        fate_last_transition={'opponent_cash':tick+1}))
            if recipient!=benefit:
                continue
            victim_before=prev[victim_seat][2].get(victim)
            if not victim_before:
                raise ValueError('positive unobserved victim')
            idx=len(events)
            method='collector' if attacker else ('enemy_base_donation_friendly_merge' if h['method']=='collector' else h['method'])
            event=dict(t=tick+1,beneficiary=benefit,victim=victim,victim_origin=victim_before[0],mass=h['amount'],
                       pos=h['pos'],winner=h['winner'],type=method,initial_kind=kind,mixed=h['mixed'],fate=Counter(),fate_last_transition={})
            if attacker:
                hunter=prev[benefit][2].get(h['winner'])
                p,c,_=victim_before
                hostile={d for q,wc,k in prev[benefit][2].values() if k=='COLLECTOR' and wc<=c for d in DEST[q]}
                safe=[d for d in DEST[p] if d not in hostile and d not in prev[benefit][1].values()]
                event['approach']={'hunter_pre':hunter,'target_pre':victim_before,
                                   'hunter_move':actions[benefit].get(h['winner']) or 'NONE','target_move':actions[victim_seat].get(victim) or 'NONE',
                                   'safe_victim_landings':safe,'attacker_origin_distance':DIST[p][hunter[0]] if hunter else None,
                                   'threatening_lighter_or_equal_collectors':sum(k=='COLLECTOR' and wc<=c and DIST[p][q]<=2 for q,wc,k in prev[benefit][2].values())}
            events.append(event)
            summary[benefit]['acquisitions_all_methods']+=1
            summary[benefit]['enemy_cargo_acquired']+=h['amount']
            summary[benefit]['acquired_'+method]+=h['amount']
            settle(idx,h['amount'],h['route'])
        for uid,tags in kept.items():
            pct=econ.levy.get(uid,0.)
            for index,mass in list(tags.items()):
                if pct:
                    tax=mass*pct;events[index]['fate']['tax_paid_to_enemy_trees']+=tax
                    events[index]['fate_last_transition']['tax_paid_to_enemy_trees']=tick+1
                    tags[index]-=tax
            if sum(tags.values())>sim.workers[uid].carbon+1e-6:
                raise ValueError('overallocated lots')
        active=kept
    for uid,tags in active.items():
        for idx,mass in tags.items():
            events[idx]['fate']['carried_at_terminal']+=mass
    for e in events:
        if abs(sum(e['fate'].values())-e['mass'])>1e-6:
            raise ValueError('lot did not balance')
        z=summary[e['beneficiary']]
        for k,v in e['fate'].items():
            z['stolen_'+k]+=v
            if e['type']=='collector':
                z['collector_stolen_'+k]+=v
    return {'id':meta['id'],'cohort':meta['cohort'],'seed':meta['seed'],'names':meta['seat_names'],'rewards':meta['score_by_seat'],
            'blob_sha256':meta['sha256'],'canonical_sha256':meta['canonical_sha256'],'n_steps':len(frames)-1,
            'seat_totals':[{k:ROUND(v) for k,v in z.items()} for z in summary],'lots':events,'refunded_contacts':refunds}


def trace_example(game,event):
    frames=game['steps'];out=[]
    # t denotes the state AFTER contact; trace through its deposit or the next five states.
    for k in range(max(0,event['t']-10),min(len(frames),max(event['t']+6,event['fate_last_transition'].get('banked',event['t'])+1))):
        state=frames[k][0]['observation']['players']
        s=event['beneficiary'];w=state[s][2].get(event['winner']);v=state[1-s][2].get(event['victim'])
        action=[(entry['action'] or {}) for entry in frames[k+1]] if k+1<len(frames) else [{},{}]
        out.append(dict(state=k,relative=k-event['t'],hunter=w,target=v,hunter_move=action[s].get(event['winner']) if w else None,
                        target_move=action[1-s].get(event['victim']) if v else None,
                        dist=DIST[w[0]][v[0]] if w and v else None,cash=[p[0] for p in state]))
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=HERE/'latest5_inventory.json')
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();rows=[];games={}
    manifest=json.loads(a.manifest.read_bytes())
    for m in manifest['games']:
        blob=(a.cache/m['path']).read_bytes()
        game=json.loads(gzip.decompress(blob))
        if hashlib.sha256(blob).hexdigest()!=m['sha256'] or hashlib.sha256(canonical(game)).hexdigest()!=m['canonical_sha256'] or m['id'] in games:
            raise ValueError(m['id'])
        if game['id']!=m['environment_uuid'] or game['rewards']!=m['score_by_seat']:
            raise ValueError('replay identity')
        rows.append(audit(game,m));games[m['id']]=game
    examples=[(g,e) for g in rows for e in g['lots'] if e['type']=='collector' and e['initial_kind']=='worker']
    result={'definitions':{'time':'transition t produces state t','lots':'Gross per possession transfer; recapturing a previously lost lot begins a new episode. A captured lot is released when it passes to an enemy worker, enemy cash, or unclaimed ground. Enemy tree tax reduces carried lots pro rata using the exact simulator deduction. Cash credited upon collision and subsequent banking are distinguished.',
                           'defensive_denominator':'collector with >=50 cargo; exposed when staying is reachable by an enemy collector of lower or equal cargo; safe landing excludes enemy bases and such enemy reach, without assuming future enemy commands.'},
            'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'games':rows}
    if examples:
        g,e=max(examples,key=lambda p:(p[1]['fate'].get('banked',0),p[1]['mass']))
        result['example']={'game':g['id'],'lot':e,'trace':trace_example(games[g['id']],e)}
    result['mutual_examples']=[]
    for g in rows:
        if not g['cohort'].endswith('_mutual'):
            continue
        choices=[e for e in g['lots'] if e['type']=='collector' and e['initial_kind']=='worker']
        choices=choices or g['refunded_contacts']
        if choices:
            picked=max(choices,key=lambda e:(e['fate'].get('banked',0),e['mass']))
            result['mutual_examples'].append({'game':g['id'],'lot':picked,'trace':trace_example(games[g['id']],picked)})
    a.out.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')


if __name__=='__main__':
    main()
