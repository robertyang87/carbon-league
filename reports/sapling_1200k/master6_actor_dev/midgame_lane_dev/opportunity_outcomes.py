"""For each eligible root, track both identified workers' positions and returns for ten turns."""
from coverage import sr,HERE,CashEvents
def trace():
    out=[]
    for c in sr.load(HERE/'coverage.json.gz'):
        raw=sr.load(sr.ROOT/c['source']);seat=c['seat']
        starts={r['t']:r for r in c['roots']};sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
        sim.cash=CashEvents(sim.cash);traces=[]
        for t,pair in enumerate(raw['actions'][:211]):
            if t in starts:
                r=starts[t];traces.append(dict(root=r['t'],worker=r['worker'],target=r['target'],history=[]))
            watch=[v for v in traces if t-v['root']<10]
            sim.cash.events.clear();sim.step(*pair)
            for v in watch:
                moves={k:dict(command=pair[s].get(k),cell=sim.workers[k].pos if k in sim.workers else None,
                              carried=sim.workers[k].carbon if k in sim.workers else None) for k,s in [(v['worker'],seat),(v['target'],1-seat)]}
                payments=[e for e in sim.cash.events if e[0]=='own_delivery' and e[3] in moves]
                v['history'].append(dict(t=t,workers=moves,payments=payments))
        out.append(dict(seed=c['seed'],seat=seat,observed=traces))
    sr.retain(HERE/'carrier_paths.json.gz',out)
if __name__=='__main__':trace()
