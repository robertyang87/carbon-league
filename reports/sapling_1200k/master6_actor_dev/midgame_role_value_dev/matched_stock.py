"""Decompose the collector inventory of matched opponents into surviving arrivals and departures."""
from collections import defaultdict
from data import sr,HERE
from elite_census import TOP,fetch
import numpy as np
def run():
    policy=('claude-AlphaCarbon-Master5s','claude-AlphaCarbon-Master6')
    keys=[{(r['opponent'],r['opponent_seat']) for r in x['strata']} for x in sr.load(HERE/'matched.json')]
    cache={};groups=defaultdict(lambda:[[],[]])
    for m in [TOP/'manifest.json',*sorted(TOP.glob('appendix_*/manifest.json'))]:
        for card in sr.load(m)['games']:
            for seat,opp in enumerate(card['seat_names']):
                faced=card['seat_names'][1-seat]
                if faced not in policy or (opp,seat) not in set.union(*keys):continue
                raw=(fetch.CACHE/card['path']).read_bytes()
                assert fetch.sha(raw)==card['sha256']
                import gzip,json
                env=json.loads(gzip.decompress(raw));frames=env['steps'];v=[]
                for f in frames:
                    p=f[0]['observation']['players'][seat]
                    v.append((set(k for k,w in p[2].items() if w[2]=='COLLECTOR'),p[0],len(p[2])))
                for phase in range(6):
                    start=phase*50;end=min(start+50,299)
                    if end>=len(v) or (opp,seat) not in keys[phase]:continue
                    change=[(len(v[t+1][0]-v[t][0]),len(v[t][0]-v[t+1][0])) for t in range(start,end)]
                    births,deaths=np.sum(change,axis=0)
                    assert len(v[start][0])+births-deaths==len(v[end][0])
                    no_wage=sum(v[t][1]<env['configuration']['recCollectorCost'] for t in range(start,end))
                    row=[len(v[start][0]),float(births),float(deaths),len(v[end][0]),no_wage]
                    groups[(phase,opp,seat)][policy.index(faced)].append(row)
                    cache.setdefault(card['id'],dict(seed=card['seed'],opponent=opp,seat=seat,faced=faced,phases=[]))['phases'].append(row)
    out=[]
    for phase in range(6):
        by_name=defaultdict(list)
        for (j,name,seat),v in groups.items():
            if j==phase:
                assert min(map(len,v))>=2
                by_name[name].append(np.mean(v[1],axis=0)-np.mean(v[0],axis=0))
        diff=np.mean([np.mean(v,axis=0) for v in by_name.values()],axis=0)
        out.append(dict(phase=phase,opponents=len(by_name),difference=diff.tolist()))
    sr.retain(HERE/'matched_stock.json.gz',dict(fields=['start','new_alive','old_gone','end','turns_cash_below_base_wage'],games=cache,summary=out))
    for v in out:print(v['phase'],v['difference'])
if __name__=='__main__':run()
