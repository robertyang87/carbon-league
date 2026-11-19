"""Compare immediate alternatives to deployed commands and retain actual future receipts separately."""
from collections import Counter,defaultdict
import math, sys, statistics, gzip, json
from review import HERE, EXP, reconcile

def outcome_delta(a,b,metric):
    if metric=='paid':return int(a['bought'])-int(b['bought'])
    if metric=='saved':return int(a['survived'])-int(b['survived'])
    if metric=='liquid':return a['credited']-b['credited']
    if metric=='cargo_and_credited':return a['held']+a['credited']-b['held']-b['credited']
    raise ValueError(metric)

def sign(v):return 'plus' if v>1e-4 else 'minus' if v < -1e-4 else 'ties'

def main():
    metrics={'deposit_now':'liquid','laden_near_home':'liquid','loaded_under_threat':'saved',
             'visible_safe_harvest':'cargo_and_credited','safe_rich_plant_now':'paid'}
    first_seen=set();first_worker=defaultdict(Counter);counts=defaultdict(Counter);details=defaultdict(Counter);units=defaultdict(set);samples=defaultdict(list);source={};game_results=[];verified=0
    for f in sorted(HERE.glob('rank_*.json.gz')):
        v=reconcile.read(f);seed=v['seed'];seat=v['seat'];arm=v['arm'];rdir=EXP/'direct_original/accounts' if seed in (621758345,621758346) else HERE
        g=reconcile.read(rdir/f'{arm}_{seed}_{seat}.json');assert g['raw_sha256']==v['raw_sha256']
        traced=reconcile.read(rdir/f'{arm}_{seed}_{seat}_turns.json.gz')
        card={k:g[k] for k in ('seed','terrain','seat','path')}
        # Independent replay verifies the factual event of every alternative-action fork.
        _,steps,ev,states,orders=reconcile.process(card,events=True)
        at=defaultdict(list)
        for e in ev:at[e['t']].append(e)
        chain={'raw_sha256':v['raw_sha256'],'seed':seed,'seat':seat,'arm':arm,'rank_file':f.name,
               'steps':[dict(t=i,joint_commands=joint,cash_by_physical_seat=steps[i]['balances'],
                       components_by_physical_seat=steps[i]['accounts']) for i,joint in enumerate(orders)]}
        cp=HERE/f'account_steps_{arm}_{seed}_{seat}.json.gz'
        cp.write_bytes(gzip.compress(json.dumps(chain,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0))
        terrain=seed
        source[str(seed)+'/'+str(seat)+'/'+arm]=v['counts']
        for q in v['opportunities']:
            t=q['t'];uid=q['uid'];sent=q['sent'];factual=q['one_step'][sent]
            assert sent==(orders[t][seat].get(uid) or 'STAY')
            bank=sum(e['value'] for e in at[t] if e['kind']=='own_delivery' and e['recipient']==seat and e['ident'][0]==uid)
            buy=any(e['kind']=='plant' and e['recipient']==seat and e['ident'][0]==uid for e in at[t])
            bal=steps[t]['balances']
            assert abs(bank-factual['credited'])<1e-4 and buy==factual['bought'] and abs((bal[seat]-bal[1-seat])-factual['cashmargin'])<.001
            if t+1<len(states):
                nxt=states[t+1][seat]['players'][seat][2].get(uid)
                assert bool(nxt)==factual['survived'] and (not nxt or abs(nxt[1]-factual['held'])<1e-4)
            verified+=1
            hist20=sum(e['value'] for u in range(t,min(t+20,len(steps))) for e in at[u] if e['kind']=='own_delivery' and e['recipient']==seat and e['ident'][0]==uid)
            ph=[i for i in traced['safe_rich_purchases_t90_189'] if i['side']=='own' and i['uid']==uid and t<=i['t']<t+4]
            actual_plant4=any(e['kind']=='plant' and e['recipient']==seat and e['ident'][0]==uid for u in range(t,min(t+4,len(steps))) for e in at[u])
            for tag in q['tags']:
                metric=metrics[tag]
                for direction in ('sapling','it50'):
                    chosen=q[direction]
                    if chosen==sent:continue
                    # The opportunity was formed for student vs sent. Where all
                    # three differ, only that direction has its own geometry tested.
                    if direction=='it50' and q['sapling']!=sent:continue
                    name=(arm,str(terrain),tag,direction)
                    counts[name]['roots']+=1
                    def target(cell,cmd):
                        dx,dy={'STAY':(0,0),'UP':(0,-1),'RIGHT':(1,0),'DOWN':(0,1),'LEFT':(-1,0)}[cmd]
                        return ((cell//15+dy)%15)*15+(cell%15+dx)%15
                    roster=states[t][seat]['players'][seat][2]
                    other_targets={target(state[0],orders[t][seat].get(w) or 'STAY') for w,state in roster.items() if w!=uid}
                    if target(q['pre']['cell'],chosen) in other_targets:
                        counts[name]['potential_team_conflict']+=1;continue
                    alt=q['one_step'][chosen];m=outcome_delta(alt,factual,metric)
                    counts[name][sign(m)]+=1;counts[name]['delta']+=m
                    counts[name]['deployed_is_it50_'+sign(m)]+=int(sent==q['it50'])
                    identity=(v['raw_sha256'],uid,tag,direction)
                    if identity not in first_seen:
                        first_seen.add(identity);first_worker[name][sign(m)]+=1
                    subtype=('to_stay' if chosen=='STAY' else 'from_stay' if sent=='STAY' else 'between_moves')
                    fk=(arm,str(terrain),tag,direction,sign(m),subtype)
                    ctr=details[fk];ctr['n']+=1;ctr['bank_within20']+=int(hist20>0);ctr['bank200_within20']+=int(hist20>=200)
                    ctr['actual_plant4']+=int(actual_plant4);ctr['actual_safe_recoup4']+=int(any(tr['net25']>0 for tr in ph));ctr['delta']+=m
                    units[fk].add((seat,uid))
                    if abs(m)>1e-4:
                        samples[tag,direction,sign(m)].append(dict(seed=seed,seat=seat,arm=arm,t=t,uid=uid,factual_order=sent,alternative=chosen,both_network_tops=[q['it50'],q['sapling']],pre=q['pre'],actual=factual,alternative_state=alt,
                            actual_future20_ordinary_deposit=hist20,actual_next_enemy_orders=q['opponent_commands'],raw=g['raw_sha256']))
        econ={}
        for who in ('own','opponent'):
            data=g['sides'][who];safe=[tr for tr in traced['safe_rich_purchases_t90_189'] if tr['side']==who]
            econ[who]=dict(cash=data['terminal_score'],bank_all=data['normal_deliveries'],bank_late=data['late_t200_298'],soil_mid=data['mid_soil_t100_199'],
                           safe_paid=len(safe),safe_recoup=sum(i['net25']>0 for i in safe))
        game_results.append(dict(seed=seed,seat=seat,arm=arm,margin=g['sides']['own']['terminal_score']-g['sides']['opponent']['terminal_score'],outcome=g['result'],steps=g['steps'],cash=econ))
    reconcile.save(HERE/'network_differences.json',{'games':source,'opportunities_with_observed_next_state_verified':verified,'by_arm_map_public_type_direction':{'/'.join(k):i for k,i in counts.items()},'metric':metrics,'first_eligible_per_replay_worker':{'/'.join(k):i for k,i in first_worker.items()},'direction_outcome_histories':{'/'.join(k):dict(i,unique_replay_worker=len(units[k])) for k,i in details.items()}})
    reconcile.save(HERE/'four_map_economy.json',game_results)
    # Preserve distinct positive and negative examples per map and arm. Future figures only cover the recorded action.
    selected={}
    for k,v in samples.items():
        earliest={}
        for a in sorted(v,key=lambda z:(z['seed'],z['arm'],z['t'],z['seat'])):earliest.setdefault((a['seed'],a['arm']),a)
        selected['/'.join(k)]=list(earliest.values())
    reconcile.save(HERE/'network_examples.json',selected)
    print(len(source),sum(i['verified_root'] for i in source.values()),verified,flush=True)
if __name__=='__main__':main()
