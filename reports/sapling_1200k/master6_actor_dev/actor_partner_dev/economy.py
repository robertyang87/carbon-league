"""Close both accounts and inspect the public positioning before the largest ten-step lead swings."""
from collections import Counter
import gzip
import json
import sys
from build import BASE,HERE,digest,out
from run import require
sys.path.insert(0,str(BASE))
from turning_point_ledger import CashEvents,FastCarbon


def load(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def audit(path,expected):
    require(digest(path)==expected,'tape hash changed')
    tape=load(path);seat=tape['candidate_seat'];sim=FastCarbon(tape['game_configuration']).reset(seed=tape['seed'])
    initial=list(sim.cash);sim.cash=CashEvents(initial);leads=[];snapshots=[];events=[]
    def snapshot():
        by=[]
        for s in (seat,1-seat):
            workers=[w for w in sim.workers.values() if w.player==s]
            by.append(dict(cash=sim.cash[s],collectors=sum(w.is_collector for w in workers),planters=sum(not w.is_collector for w in workers),
                cargo=sum(w.carbon for w in workers),trees=sum(tree.player==s for tree in sim.trees.values())))
        leads.append(sim.cash[seat]-sim.cash[1-seat]);snapshots.append(by)
    snapshot()
    for joint in tape['actions']:
        sim.cash.events.clear();sim.step(*joint);led=[Counter(),Counter()]
        for kind,s,change,*_ in sim.cash.events:
            led[s][kind]+=change
        events.append(led);snapshot()
    def ledger(lo,hi,s):
        c=Counter()
        for pair in events[lo:hi]:c.update(pair[s])
        return dict(c)
    n=len(events);sides=[ledger(0,n,s) for s in range(2)]
    require(all(abs(initial[s]+sum(sides[s].values())-sim.cash[s])<1e-6 for s in range(2)),'account does not recover terminal cash')
    windows={}
    for label,order in [('largest_gain',max),('largest_loss',min)]:
        start=order(range(n-9),key=lambda i:leads[i+10]-leads[i]);end=start+10
        windows[label]=dict(start=start,end=end,change=leads[end]-leads[start],lead_before=leads[start],lead_after=leads[end],
            own_cash_events=ledger(start,end,seat),opponent_cash_events=ledger(start,end,1-seat),
            prior_public_snapshots=[dict(t=t,own=snapshots[t][0],opponent=snapshots[t][1]) for t in sorted({max(0,start-i) for i in (0,3,5,10)})])
    return dict(seed=tape['seed'],seat=seat,raw_sha256=expected,terminal_step=n,final_cash=list(sim.cash),official_score=tape['official_score_by_seat'],
                by_physical_seat=sides,windows=windows)


def run():
    c=load(HERE/'comparison.json');plan=load(HERE/'protocol.json');rows=[]
    for g in c['games']:
        seed=g['seed'];s=g['seat'];name=f'raw_{seed}_{s}.json.gz'
        arms={arm:audit(folder/name,g[arm+'_sha256']) for arm,folder in [('candidate',HERE/'native/candidate'),
             ('parent',HERE.parent/plan['parent_source'][str(seed)])]}
        rows.append(dict(seed=seed,seat=s,arms=arms))
    out(HERE/'income.json',dict(comparison_sha256=digest(HERE/'comparison.json'),code_sha256=digest(HERE/'economy.py'),cases=rows))
    print('historical-map cash reconciled',len(rows)*2)


if __name__=='__main__':run()
