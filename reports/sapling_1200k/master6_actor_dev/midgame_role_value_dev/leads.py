"""Trace whether new-map candidates still had a recoverable cash lead when their action first changed."""
from data import HERE,sr
from collections import Counter
from turning_point_ledger import CashEvents
def one(path,seat):
    raw=sr.load(path);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);sim.cash=CashEvents(sim.cash)
    track=[0.];events=[]
    for cmd in raw['actions']:
        sim.cash.events.clear();sim.step(*cmd)
        track.append(round(sim.cash[seat]-sim.cash[1-seat],2));events.append(list(sim.cash.events))
    windows={}
    for m in (0,250,500):
        indices=[i for i,v in enumerate(track) if i and v>=m]
        if not indices:windows[str(m)]=None;continue
        t=max(indices);n=min(t+10,len(events));flows=[Counter(),Counter()]
        for e in events[t:n]:
            for k,s,v,*_ in e:flows[s][k]+=v
        windows[str(m)]=dict(last_t=t,at=track[t],after=track[n],until=n,flows=[dict(v) for v in flows])
    return dict(seed=raw['seed'],seat=seat,path=str(path.relative_to(sr.ROOT)),peak=max(track),peak_t=track.index(max(track)),
                samples=[track[t] for t in (50,100,150,200,250,299) if t<len(track)],last_leads=windows)
def run():
    paths=[]
    for case in sr.load(HERE.parent/'harvest_contact_dev/coverage.json.gz'):paths.append((sr.ROOT/case['source'],case['seat']))
    for folder in (HERE.parent/'harvest_contact_dev',HERE):
        for arm in ('parent','candidate'):
            paths.extend((p,int(p.name.split('_')[-1].split('.')[0])) for p in sorted((folder/'games'/arm).glob('raw_*.json.gz')))
    games=[one(p,s) for p,s in paths];sr.retain(HERE/'lead_history.json.gz',games)
    for r in games[-8:]:print(r['seed'],r['seat'],r['path'].split('/')[-2],r['peak'],r['samples'])
if __name__=='__main__':run()
