"""Compare the timing and identity of the largest five-turn deterioration within each paired response."""
from collections import Counter
from screen import HERE,ROOT,FastCarbon,CashEvents,read,retain,sha


def states(reg,arm):
    raw=read(ROOT/reg['source'])
    case=HERE/f"response_{reg['seed']}_{reg['seat']}"
    receipt=read(case/f'live_{arm}.json')
    path=case/receipt['action_tape']
    if sha(path.read_bytes())!=receipt['action_sha256']:
        raise ValueError('branch tape binding')
    cmds=raw['actions'][:reg['t']]+read(path)
    sim=FastCarbon(raw['game_configuration']).reset(seed=reg['seed'])
    sim.cash=CashEvents(sim.cash);rows=[]
    for t,joint in enumerate(cmds):
        before=list(sim.cash);evs=sim.cash.events;evs.clear()
        sim.step(*joint)
        rows.append({'t':t,'before':before,'after':list(sim.cash),'events':list(evs)})
    if rows[-1]['after']!=receipt['score']:
        raise ValueError('window diagnostic requires normal cash-score terminal')
    return rows


def describe(rows,s,left,right):
    entries=[Counter(),Counter()];largest=[]
    for r in rows[left:right]:
        for ev in r['events']:
            entries[ev[1]][ev[0]]+=ev[2]
            if ev[0]=='own_delivery':
                largest.append({'t':r['t'],'seat':ev[1],'uid':ev[3],'cell':ev[4],'amount':ev[2]})
    ranked=sorted(largest,key=lambda e:(-e['amount'],e['t'],e['uid']))
    return {'cash_before':rows[left]['before'],'cash_after':rows[right-1]['after'],'ledger_by_seat':entries,
            'largest_worker_deliveries':ranked[:5]}


def main():
    registered=read(HERE/'mechanism.json')['selected_roots']
    records=[]
    for reg in registered:
        s=reg['seat'];p=states(reg,'parent');a=states(reg,'alternative');t0=reg['t']
        def delta(t):
            return (a[t]['before'][s]-a[t]['before'][1-s])-(p[t]['before'][s]-p[t]['before'][1-s])
        def swing(start):
            end=start+4
            margin=(a[end]['after'][s]-a[end]['after'][1-s])-(p[end]['after'][s]-p[end]['after'][1-s])
            return margin-delta(start)
        start=min(range(t0,min(len(a),len(p))-4),key=swing)
        records.append({'seed':reg['seed'],'seat':s,'worst_5_transition_start':start,'end_inclusive':start+4,'change_in_paired_margin':swing(start),
                        'prior_five':{arm:describe(rows,s,max(t0,start-5),start) if start>t0 else None for arm,rows in [('parent',p),('option',a)]},
                        'within_five':{arm:describe(rows,s,start,start+5) for arm,rows in [('parent',p),('option',a)]}})
    retain(HERE/'cash_windows.json',{'definition':'five consecutive transitions minimizing change in option minus parent live-cash margin; largest by realized hindsight','roots':records})
    print([(r['seed'],r['worst_5_transition_start'],r['change_in_paired_margin']) for r in records])

if __name__=='__main__':
    main()
