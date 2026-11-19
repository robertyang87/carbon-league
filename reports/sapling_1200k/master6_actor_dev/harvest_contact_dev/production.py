"""Attribute mineral throughput to live collector turns, collection frequency and yield per productive stay."""
from collections import Counter
import inspect,textwrap
from scan import sr,HERE

def measuring_engine():
    method=sr.FastCarbon.step
    marks=[('w.carbon += delta', "self.harvest[w.player].append((w.id,pos,delta))"),
           ('self.cash[p] += w.carbon - smelt',"self.banked[p].append((w.id,w.carbon-smelt))")]
    source=textwrap.dedent(inspect.getsource(method))
    for line,record in marks:
        assert source.count(line)==1
        indent=next(x[:len(x)-len(x.lstrip())] for x in source.splitlines() if x.strip()==line)
        source=source.replace(indent+line,indent+record+'\n'+indent+line)
    scope=dict(method.__globals__);exec(source,scope)
    return type('ProductionTrace',(sr.FastCarbon,),{'step':scope['step']})

def analyze(path,s,engine):
    raw=sr.load(path);sim=engine(raw['game_configuration']).reset(seed=raw['seed'])
    period=[Counter(),Counter()];rows=[]
    for t,pair in enumerate(raw['actions']):
        sim.harvest=[[],[]];sim.banked=[[],[]]
        for w in sim.workers.values():
            if w.is_collector:
                period[w.player]['collector_turns']+=1
                period[w.player]['stay_orders']+=not pair[w.player].get(w.id)
        sim.step(*pair)
        for side in (0,1):
            p=period[side];lots=sim.harvest[side]
            p['productive_stays']+=len(lots);p['harvested']+=sum(v for _,_,v in lots)
            p['bank_received']+=sum(v for _,v in sim.banked[side])
            p['cargo_taken_by_opponent']+=sum(e[2] for e in sim.last_collisions if e[1]==side and e[4]==1-side)
        if t+1 in (50,100,150,200,250,len(raw['actions'])):
            rows.append(dict(end=t+1,own=period[s],opponent=period[1-s],cash=list(sim.cash)))
            period=[Counter(),Counter()]
    assert list(sim.cash)==raw['official_score_by_seat'] or any(raw['extinct_by_seat'])
    return dict(seed=raw['seed'],seat=s,source=str(path.relative_to(sr.ROOT)),periods=rows)
def main():
    engine=measuring_engine()
    games=[analyze(sr.ROOT/c['source'],c['seat'],engine) for c in sr.load(HERE/'coverage.json.gz')]
    sr.retain(HERE/'production.json.gz',games)
    for i in range(6):
        sums=[sum((Counter(g['periods'][i][side]) for g in games),Counter()) for side in ('own','opponent')]
        print(games[0]['periods'][i]['end'],[(p['collector_turns'],p['productive_stays'],round(p['harvested']/p['productive_stays'],2),round(p['harvested']/10,1)) for p in sums])
if __name__=='__main__':main()
