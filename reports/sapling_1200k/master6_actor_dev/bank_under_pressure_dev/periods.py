"""Locate sustained sources of the score gap without conditioning on winning or losing plays."""
from audit import sr,HERE
from turning_point_ledger import CashEvents
from collections import Counter
def compare(path,seat):
    raw=sr.load(path);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    sim.cash=CashEvents(sim.cash);result=[];net=Counter();start=0;start_diff=sim.cash[seat]-sim.cash[1-seat]
    for t,pair in enumerate(raw['actions']):
        sim.cash.events.clear();sim.step(*pair)
        for k,s,v,*meta in sim.cash.events:net[k]+=(1 if s==seat else -1)*v
        if (t+1)%50==0 or t+1==len(raw['actions']):
            diff=sim.cash[seat]-sim.cash[1-seat]
            assert abs(sum(net.values())-diff+start_diff)<1e-5
            result.append(dict(start=start,end=t+1,net=dict(net),margin_change=diff-start_diff,cash_margin=diff))
            net=Counter();start=t+1;start_diff=diff
    return dict(seed=raw['seed'],seat=seat,source=str(path.relative_to(sr.ROOT)),windows=result)
if __name__=='__main__':
    hist=[compare(sr.ROOT/c['source'],c['seat']) for c in sr.load(HERE/'audit.json')]
    sr.retain(HERE/'development_periods.json',hist)
    if (HERE/'scores.json').exists():
        sr.retain(HERE/'paired_periods.json',[compare(HERE/'games'/r['arm']/f"raw_{r['seed']}_{r['seat']}.json.gz",r['seat']) for r in sr.load(HERE/'scores.json')])
    for i in range(6):
        w=[h['windows'][i] for h in hist]
        keys=('own_delivery','tree_absorption','plant','seize')
        print(w[0]['start'],w[0]['end'],{k:round(sum(row['net'].get(k,0) for row in w)/len(w),1) for k in keys},round(sum(row['margin_change'] for row in w)/len(w),1))
