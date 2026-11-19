"""Find the last crossing into a 500-point deficit and trace both seats' preceding cash and actions."""
from actual_rescues import HERE,sr,CashEvents
from collections import Counter
def explain(directory):
    results=[]
    for r in sr.load(directory/'scores.json'):
        seat=r['seat'];path=directory/'games'/r['arm']/f"raw_{r['seed']}_{seat}.json.gz"
        raw=sr.load(path);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=r['seed'])
        sim.cash=CashEvents(sim.cash);rows=[];last=None
        for t,cmds in enumerate(raw['actions']):
            pre=sim.obs(seat)['players'];before=list(sim.cash);sim.cash.events.clear();sim.step(*cmds)
            if before[seat]-before[1-seat]>-500>=sim.cash[seat]-sim.cash[1-seat]:last=t
            rows.append(dict(t=t,players=pre,commands=cmds,events=list(sim.cash.events),after=list(sim.cash)))
        assert last is not None and r['margin']<=-500
        windows={}
        for n in (3,5,10,30):
            subset=rows[max(0,last-n+1):last+1];flow=[Counter(),Counter()]
            for step in subset:
                for k,s,v,*_ in step['events']:flow[s][k]+=v
            windows[str(n)]=flow
        results.append(dict(seed=r['seed'],seat=seat,arm=r['arm'],final_margin=r['margin'],last_crossing=last,
                            windows=windows,trace=rows[max(0,last-10):last+1]))
    return results
if __name__=='__main__':
    results=explain(HERE.parent/'bank_under_pressure_dev')+explain(HERE)
    sr.retain(HERE/'deficit_crossings.json.gz',results)
    for x in results:
        s=x['seat'];f=x['windows']['10']
        print(x['seed'],x['arm'],s,x['last_crossing'],[(k,round(f[s].get(k,0)-f[1-s].get(k,0))) for k in ('own_delivery','tree_absorption','plant','seize')])
