"""Aggregate once per map/seat/cell/tick, retaining sole enemy-versus-own encounters separately."""
import collections,statistics
from audit import HERE,read,write

def summarize(games):
    by={};maps={}
    for cohort,arm in sorted({(g['card']['cohort'],g['card']['arm']) for g in games}):
        group=[g for g in games if (g['card']['cohort'],g['card']['arm'])==(cohort,arm)]
        union=collections.Counter();margins=[]
        for g in group:
            c=g['card'];s=c['seat'];margins.append(c['score'][s]-c['score'][1-s])
            for e in g['events']:union[e['category']]+=1
        key=f'{cohort}/{arm}'
        by[key]={'games':len(group),'maps':len({g['card']['seed'] for g in group}),'unique_cells_t_by_type':union,'mean_own_terminal_margin':statistics.mean(margins),'sides':{}}
        for tag in ('pure','mixed_stack','duel'):
            for role in ('own','opponent'):
                path=f'{tag}/{role}';a=collections.Counter();in_maps=set();where=collections.Counter();bymap=collections.defaultdict(collections.Counter);route=collections.Counter();source=collections.Counter();lag=collections.Counter()
                for g in group:
                    c=g['card'];p=c['seat'] if role=='own' else 1-c['seat'];base={64,160} if p==0 else {70,154}
                    seen=False
                    for e in g['events']:
                        if e['category']!=tag or str(p) not in e['team_counts']:continue
                        seen=True;in_maps.add(c['seed']);a['cells']+=1;where[p]+=1;lower=bymap[(c['seed'],c['seat'])]
                        lower['cells']+=1
                        a['group_includes_recruit']+=int(any(w['new'] for w in e['members']))
                        a['on_own_base']+=int(e['cell'] in base);a['t200_or_later']+=int(e['t']>=200)
                        for v in e['victims']:
                            if v['owner']!=p:continue
                            a['dead_'+v['role']]+=1;a['pre_settlement_cargo']+=v['cargo'];a['laden_deaths']+=int(v['cargo']>0);a['ground_after_cap']+=v['ground'];a['clipped']+=v['cap_loss']
                            dest='own' if v['recipient']==p else 'enemy' if v['recipient'] is not None else 'loose'
                            route[dest+'/'+v['destination']]+=v['cargo'];lower['cargo']+=v['cargo']
                            if v['cargo']>0:
                                if v['trace_end']=='cash':
                                    typ=('own' if v['trace_recipient']==p else 'enemy')+'_cash'
                                    a['strict_'+typ]+=v['cargo'];a['strict_'+typ+'_transfers']+=1
                                    if typ=='own_cash':lag[v['trace_end_t']-e['t']]+=1
                                else:typ=v['trace_end']
                                source[typ]+=v['cargo']
                    a['games_with_cells']+=int(seen)
                assert abs(a['pre_settlement_cargo']-sum(route.values()))<1e-4
                assert abs(a['pre_settlement_cargo']-sum(source.values()))<1e-4
                by[key]['sides'][path]={'stats':a,'distinct_maps':len(in_maps),'physical_seats_events':where,'immediate_route_gross':route,'strict_traced_gross':source,'ticks_to_cashed_by_original_owner':lag}
                maps[key+'/'+path]={f'{seed}/{seat}':ctr for (seed,seat),ctr in bymap.items()}
    write(HERE/'summary.json',dict(cohorts=by,per_seed_and_exam_seat=maps))
    return by
if __name__=='__main__':summarize(read(HERE/'collisions.json.gz'))
